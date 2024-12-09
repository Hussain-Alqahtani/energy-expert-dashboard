import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from streamlit_option_menu import option_menu

# Configuration
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

class DataLoader:
    @staticmethod
    def load_and_process_data():
        """Load and process the CSV data with enhanced features"""
        try:
            df = pd.read_csv("energy_data.csv")
            
            # Basic datetime processing
            df['date'] = pd.to_datetime(df['date'])
            df['hour'] = df['hour'].astype(float)
            df['day_of_week'] = df['date'].dt.day_name()
            df['week'] = df['date'].dt.isocalendar().week
            df['month'] = df['date'].dt.month
            df['month_name'] = df['date'].dt.strftime('%B')
            df['year'] = df['date'].dt.year
            
            # Monthly aggregation
            df['monthly_usage'] = df.groupby(['year', 'month'])['usage'].transform('sum')
            df['monthly_avg'] = df.groupby(['year', 'month'])['usage'].transform('mean')
            
            # Enhanced time features
            df['quarter'] = df['date'].dt.quarter
            df['day_type'] = df.apply(
                lambda x: 'Holiday' if x['is_holiday'] else 
                ('Weekend' if x['is_weekend'] else 'Weekday'), axis=1
            )
            
            # Weather categorization
            df['temp_category'] = pd.qcut(
                df['temp'], 
                q=5, 
                labels=['Very Cold', 'Cold', 'Moderate', 'Warm', 'Hot']
            )
            
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

class DataFilter:
    @staticmethod
    def get_city_date_range(df, selected_cities):
        """Determine min and max dates based on selected cities."""
        try:
            filtered_df = df[df['city'].isin(selected_cities)]
            min_date = filtered_df['date'].min().date()
            max_date = filtered_df['date'].max().date()
            return min_date, max_date
        except Exception as e:
            st.error(f"Error determining date range: {str(e)}")
            return df['date'].min().date(), df['date'].max().date()

    @staticmethod
    def apply_filters(df, start_date, end_date, cities, appliances):
        """Apply selected filters to the dataframe."""
        try:
            mask = (
                (df['date'].dt.date >= start_date) &
                (df['date'].dt.date <= end_date) &
                (df['city'].isin(cities)) &
                (df['appliance'].isin(appliances))
            )
            return df[mask]
        except Exception as e:
            st.error(f"Error applying filters: {str(e)}")
            return df


class InsightGenerator:
    def __init__(self, df):
        self.df = df
        
    def generate_context(self, filtered_df):
        """Generate rich context from filtered data."""
        context = []
        for city in filtered_df['city'].unique():
            city_data = filtered_df[filtered_df['city'] == city]
            
            for appliance in city_data['appliance'].unique():
                app_data = city_data[city_data['appliance'] == appliance]
                
                # Monthly analysis
                monthly_usage = app_data.groupby(['year', 'month_name'])['usage'].agg([
                    ('total_usage', 'sum'),
                    ('average_usage', 'mean')
                ]).round(3)
                
                # Sort months chronologically
                month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                               'July', 'August', 'September', 'October', 'November', 'December']
                monthly_usage = monthly_usage.reset_index()
                monthly_usage['month_order'] = monthly_usage['month_name'].map(
                    {month: idx for idx, month in enumerate(month_order)}
                )
                monthly_usage = monthly_usage.sort_values(['year', 'month_order'])
                
                # Peak months and yearly trends
                peak_month = monthly_usage.loc[monthly_usage['total_usage'].idxmax()]
                yearly_totals = app_data.groupby('year')['usage'].sum()
                
                # Seasonal analysis
                season_usage = app_data.groupby('season')['usage'].mean().round(3).to_dict()
                
                # Weekend vs Weekday usage
                weekend_usage = app_data.groupby('is_weekend')['usage'].mean().round(3).to_dict()
                
                # Holiday usage
                holiday_usage = app_data.groupby('is_holiday')['usage'].mean().round(3).to_dict()
                
                # Outliers analysis
                outlier_count = app_data['outlier'].sum()
                outlier_percentage = (outlier_count / len(app_data)) * 100
                
                # Weather analysis
                weather_summary = {
                    "Average Temperature (°C)": app_data["temp"].mean().round(2),
                    "Average Wind Speed (m/s)": app_data["wspd"].mean().round(2),
                    "Average Humidity (%)": app_data["rhum"].mean().round(2),
                    "Average Pressure (hPa)": app_data["pres"].mean().round(2)
                }

                # Add insights to context
                context.append(f"""
                City: {city}, Appliance: {appliance}

                Monthly Usage:
                - {monthly_usage.to_dict('records')}

                Peak Month:
                - {peak_month['month_name']} {peak_month['year']}: {peak_month['total_usage']} kWh

                Yearly Totals:
                - {yearly_totals.to_dict()}

                Seasonal Usage Averages:
                - {season_usage}

                Weekend vs Weekday Usage:
                - {weekend_usage}

                Holiday Usage Patterns:
                - {holiday_usage}

                Outlier Information:
                - Total Outliers: {outlier_count}
                - Outlier Percentage: {outlier_percentage:.2f}%

                Weather Summary:
                - {weather_summary}
                """)
        
        return "\n".join(context)

    def initialize_qa_chain(self):
        """Initialize the QA chain with enhanced context"""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(self.generate_context(self.df))
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        vector_store = Chroma.from_texts(texts, embeddings)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1,
            max_output_tokens=2048
        )
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
        )

    def get_analysis(self, question, filtered_df, appliances, cities, date_range):
        """Generate analysis based on the question and filtered data."""
        try:
            # Ensure context is generated for the filtered data
            context = self.generate_context(filtered_df)

            # Explicitly include city details in the query
            city_details = ", ".join(cities) if cities else "all cities"
            query = f"""
            Analyze energy usage for {', '.join(appliances)} in {city_details} 
            from {date_range[0]} to {date_range[1]}.
            
            Relevant data and patterns:
            {context}
            
            User question: {question}
            
            Please provide a detailed analysis focusing on the specific question asked.
            Include exact numbers, months, and years when available.
            Highlight trends for the specified city or cities.
            """
            
            # Initialize QA chain
            qa_chain = self.initialize_qa_chain()
            response = qa_chain.run(query)
            
            return response
        except Exception as e:
            raise Exception(f"Analysis generation failed: {str(e)}")

def main():
    st.set_page_config(layout="wide")
    st.title("🏠 Smart Energy Analysis Dashboard")
    
    # Load and initialize data
    df = DataLoader.load_and_process_data()
    if df is None:
        st.error("Failed to load data. Please check your data file.")
        return
    
    # Initialize components
    insight_gen = InsightGenerator(df)
    
    # Sidebar filters
    with st.sidebar:
        # City filter: Show all cities as default
        cities = st.multiselect(
            "Select Cities", 
            options=sorted(df['city'].unique()),  # Available options
            default=sorted(df['city'].unique())   # Default to all cities
        )

        # Get the dynamic date range based on selected cities
        min_date, max_date = DataFilter.get_city_date_range(df, cities)
        
        # Date range picker
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),  # Default range for selected cities
            key="date_range"
        )
        
        # Appliance filter: Show all appliances as default
        appliances = st.multiselect(
            "Select Appliances",
            options=sorted(df['appliance'].unique()),  # Available options
            default=sorted(df['appliance'].unique())   # Default to all appliances
        )
    
    # Apply filters
    filtered_df = DataFilter.apply_filters(
        df, date_range[0], date_range[1], cities, appliances
    )
    
    # Display visualizations and insights
    if not filtered_df.empty:
        # Tabs for different sections
        tabs = st.tabs([
        "Overview and Key Insights",
        "Energy Usage Trends",
        "Outliers Analysis",
        "AI Insights"
        ])
        
        # Overview and Key Insights Section
        with tabs[0]:
            st.header("Overview and Key Insights")

            # KPI Cards (Row 1)
            st.markdown("### **_Key Metrics_**")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            with kpi_col1:
                avg_daily_consumption = filtered_df[filtered_df["appliance"] == "grid"].groupby("day")["usage"].mean().mean()
                st.metric("Avg. Daily Consumption (Grid)", f"{avg_daily_consumption:.2f} kWh")
            with kpi_col2:
                num_appliances = filtered_df["appliance"].nunique()
                st.metric("Number of Appliances", num_appliances)
            with kpi_col3:
                avg_outliers = filtered_df["outlier"].mean()
                st.metric("Avg. Outliers Count", f"{avg_outliers:.2f}")
            with kpi_col4:
                total_data_points = len(filtered_df)
                st.metric("Data Points", total_data_points)

            # Second Row Charts
            st.markdown("### **_Monthly and Appliance Insights_**")
            row2_col1, row2_col2, row2_col3 = st.columns([1, 1, 1])


            # Chart 1: Grid Usage (Last Month vs Previous Month)
            with row2_col1:
                # Determine the unique months in the filtered dataset
                unique_months = filtered_df["date"].dt.to_period("M").unique()
                
                if len(unique_months) >= 2:
                    # Sort months and select the last two
                    sorted_months = sorted(unique_months)
                    last_month = sorted_months[-1]
                    prev_month = sorted_months[-2]
                    
                    # Filter data for the last two months
                    last_month_data = filtered_df[filtered_df["date"].dt.to_period("M") == last_month]
                    prev_month_data = filtered_df[filtered_df["date"].dt.to_period("M") == prev_month]
                    
                    # Extract day of the month for a common X-axis
                    last_month_data["day_of_month"] = last_month_data["date"].dt.day
                    prev_month_data["day_of_month"] = prev_month_data["date"].dt.day

                    # Label the months for the legend
                    last_month_data["label"] = f"{last_month.start_time.strftime('%B %Y')}"
                    prev_month_data["label"] = f"{prev_month.start_time.strftime('%B %Y')}"

                    # Combine data for plotting
                    combined_data = pd.concat([last_month_data, prev_month_data])
                    combined_data = combined_data.groupby(["day_of_month", "label"])["usage"].sum().reset_index()

                    # Create the line chart
                    fig_usage = px.line(
                        combined_data,
                        x="day_of_month",
                        y="usage",
                        color="label",
                        title="Grid Usage: Last Month vs Previous Month",
                        labels={"day_of_month": "Day of Month", "usage": "Total Usage (kWh)", "label": "Month"},
                    )
                    fig_usage.update_layout(
                        xaxis=dict(title="Day of Month"),
                        yaxis=dict(title="Total Usage (kWh)"),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_usage, use_container_width=True)
                else:
                    st.warning("Not enough data to compare two months. Please adjust your date range.")
                
            # Chart 2: Top 5 Appliances by Total Monthly Usage
            with row2_col2:
                appliance_usage = filtered_df[filtered_df["appliance"] != "grid"].groupby("appliance")["usage"].sum().reset_index()
                top_appliances = appliance_usage.sort_values(by="usage", ascending=False).head(5)
                fig_appliances = px.bar(
                    top_appliances,
                    x="usage",
                    y="appliance",
                    orientation="h",
                    title="Top 5 Appliances (Monthly Avg Usage)",
                    labels={"appliance": "Appliance", "usage": "Total Usage (kWh)"},
                )
                fig_appliances.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_appliances, use_container_width=True)

            # Chart 3: Weekend Effect on Grid Consumption
            with row2_col3:
                weekend_usage = filtered_df.groupby("is_weekend")["usage"].mean().reset_index()
                fig_weekend = px.bar(
                    weekend_usage,
                    x="is_weekend",
                    y="usage",
                    title="Weekend Effect on Grid Consumption",
                    labels={"is_weekend": "Weekend", "usage": "Daily Avg Usage (kWh)"},
                )
                fig_weekend.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_weekend, use_container_width=True)

            # Third and Fourth Row Chart
            st.markdown("### **_Total Grid Consumption by City (Smoothed)_**")
            filtered_df["day"] = filtered_df["date"].dt.date
            daily_city_usage = (
                filtered_df[filtered_df["appliance"] == "grid"]
                .groupby(["day", "city"])["usage"]
                .mean()
                .reset_index()
            )

            fig_city_usage = px.line(
                daily_city_usage,
                x="day",
                y="usage",
                color="city",
                title="Total Grid Consumption by City (Daily Average)",
                labels={"day": "Date", "usage": "Daily Avg Usage (kWh)", "city": "City"},
            )
            fig_city_usage.update_layout(
                xaxis=dict(title="Date"),
                yaxis=dict(title="Daily Avg Usage (kWh)"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_city_usage, use_container_width=True)


        # Energy Usage Trends
        with tabs[1]:
            st.header("Energy Usage Trends")

            # Hourly Usage Line Plot (Subplots by Appliance)
            st.markdown("### **_Hourly Usage Trends_**")

            # Filter data for the selected appliances
            hourly_filtered_df = filtered_df[filtered_df["appliance"].isin(appliances)]

            # Create subplots for each appliance
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            # Create a subplot figure with one row per appliance
            fig_hourly = make_subplots(
                rows=len(appliances),
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,  # Compact spacing between subplots
                subplot_titles=[f"Hourly Usage for {appliance.capitalize()}" for appliance in appliances]
            )

            for idx, appliance in enumerate(appliances):
                appliance_data = hourly_filtered_df[hourly_filtered_df["appliance"] == appliance]
                avg_hourly_usage = appliance_data.groupby("hour")["usage"].mean().reset_index()

                fig_hourly.add_trace(
                    go.Scatter(
                        x=avg_hourly_usage["hour"],
                        y=avg_hourly_usage["usage"],
                        mode="lines",
                        name=appliance.capitalize()
                    ),
                    row=idx + 1,
                    col=1
                )

            fig_hourly.update_layout(
                height=250 * len(appliances),  # Adjust height dynamically
                title="Hourly Usage Trends by Appliance",
                xaxis=dict(title="Hour of Day", tickmode="linear", dtick=1),  # Increment by 1
                yaxis_title="Average Usage (kWh)",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False  # Disable legend since titles show appliance names
            )

            st.plotly_chart(fig_hourly, use_container_width=True, key="hourly_chart_subplots")

            # Heatmap: Appliances vs Usage
            st.markdown("### **_Appliances vs Usage Heatmap_**")

            # Adjust X-axis based on the global filtered data
            heatmap_data = filtered_df.groupby(["appliance", "hour"])["usage"].mean().reset_index()

            # Create the heatmap
            fig_heatmap = px.density_heatmap(
                heatmap_data,
                x="hour",
                y="appliance",
                z="usage",
                color_continuous_scale="Viridis",
                title=f"Appliances vs Hour (Usage)",
                labels={"hour": "Hour", "appliance": "Appliance", "usage": "Avg Usage (kWh)"},
            )
            fig_heatmap.update_layout(
                xaxis=dict(title="Hour of Day", tickmode="linear", dtick=1),  # Increment by 1
                yaxis=dict(title="Appliance"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=800  # Make the heatmap larger
            )

            # Display the heatmap
            st.plotly_chart(fig_heatmap, use_container_width=True, key="heatmap_chart") 
        
        
        with tabs[2]:

            st.header("Outliers Analysis")

            # Section 1: Outlier Count by Appliance
            st.markdown("### **_Outlier Count by Appliance_**")

            # Group by appliance to calculate total outliers
            outlier_count_by_appliance = (
                filtered_df.groupby("appliance")["outlier"].sum().reset_index()
            )
            outlier_count_by_appliance.rename(columns={"outlier": "outlier_count"}, inplace=True)

            # Bar chart for outlier count by appliance
            fig_outlier_count = px.bar(
                outlier_count_by_appliance,
                x="appliance",
                y="outlier_count",
                title="Total Outliers by Appliance",
                labels={"appliance": "Appliance", "outlier_count": "Outlier Count"},
                color="outlier_count",
                color_continuous_scale="Viridis"
            )
            fig_outlier_count.update_layout(
                xaxis=dict(title="Appliance"),
                yaxis=dict(title="Outlier Count"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_outlier_count, use_container_width=True, key="outlier_count_chart")

            # Section 2: Heatmap of Outliers by Appliance and Hour
            st.markdown("### **_Outliers Heatmap by Appliance and Hour_**")

            # Group by appliance and hour to calculate total outliers
            outlier_heatmap_data = (
                filtered_df.groupby(["appliance", "hour"])["outlier"]
                .sum()
                .reset_index()
                .rename(columns={"outlier": "outlier_count"})
            )

            # Heatmap for outliers by appliance and hour
            fig_outlier_heatmap = px.density_heatmap(
                outlier_heatmap_data,
                x="hour",
                y="appliance",
                z="outlier_count",
                color_continuous_scale="Viridis",
                title="Heatmap of Outliers by Appliance and Hour",
                labels={"hour": "Hour of Day", "appliance": "Appliance", "outlier_count": "Outlier Count"},
            )
            fig_outlier_heatmap.update_layout(
                xaxis=dict(title="Hour of Day", tickmode="linear", dtick=1),
                yaxis=dict(title="Appliance"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=600
            )
            st.plotly_chart(fig_outlier_heatmap, use_container_width=True, key="outlier_heatmap_chart")

            # Section 3: Outlier Trends by Day and Month
            st.markdown("### **_Outlier Trends by Day and Month_**")

            # Outliers by day of the week
            outliers_by_day = (
                filtered_df.groupby(filtered_df["date"].dt.day_name())["outlier"]
                .mean() * 100
            ).reset_index().rename(columns={"date": "day_name", "outlier": "outlier_percentage"})

            # Outliers by month
            outliers_by_month = (
                filtered_df.groupby(filtered_df["date"].dt.month_name())["outlier"]
                .mean() * 100
            ).reset_index().rename(columns={"date": "month", "outlier": "outlier_percentage"})

            # Bar charts for trends
            col1, col2 = st.columns(2)
            with col1:
                fig_day = px.bar(
                    outliers_by_day,
                    x="day_name",
                    y="outlier_percentage",
                    title="Outlier Percentage by Day of the Week",
                    labels={"day_name": "Day of the Week", "outlier_percentage": "Outlier Percentage (%)"}
                )
                fig_day.update_layout(
                    xaxis=dict(title="Day of the Week"),
                    yaxis=dict(title="Outlier Percentage (%)"),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_day, use_container_width=True, key="outlier_by_day_chart")

            with col2:
                fig_month = px.bar(
                    outliers_by_month,
                    x="month",
                    y="outlier_percentage",
                    title="Outlier Percentage by Month",
                    labels={"month": "Month", "outlier_percentage": "Outlier Percentage (%)"}
                )
                fig_month.update_layout(
                    xaxis=dict(title="Month"),
                    yaxis=dict(title="Outlier Percentage (%)"),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_month, use_container_width=True, key="outlier_by_month_chart")
            
        with tabs[3]:
            st.header("AI Insights")
            question = st.text_input(
                "Ask about your energy usage:",
                "What are the main patterns and saving opportunities?"
            )
            if st.button("Generate Response"):
                if question:
                    with st.spinner("Analyzing..."):
                        try:
                            analysis = insight_gen.get_analysis(
                                question,
                                filtered_df,
                                appliances,
                                cities,
                                date_range
                            )
                            st.write(analysis)
                        except Exception as e:
                            st.error(f"Error generating analysis: {str(e)}")
                            st.error(str(e))
                else:
                    st.warning("Please enter a question before generating a response.")
                    
if __name__ == "__main__":
    main()