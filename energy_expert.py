import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
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
        
        vector_store = FAISS.from_texts(texts, embeddings)
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
            # Initialize chain
            qa_chain = self.initialize_qa_chain()
            
            # Create prompt
            context_prompt = f"""
            For cities: {', '.join(cities)}
            Date range: {date_range[0]} to {date_range[1]}
            Analyzing appliances: {', '.join(appliances)}
            
            Question: {question}
            """
            
            # Get response
            response = qa_chain.run(context_prompt)
            
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
            options=sorted(df['city'].unique()),
            default=sorted(df['city'].unique())
        )

        # Get the dynamic date range based on selected cities
        min_date, max_date = DataFilter.get_city_date_range(df, cities)
        
        # Date range picker
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            key="date_range"
        )
        
        # Appliance filter: Show all appliances as default
        appliances = st.multiselect(
            "Select Appliances",
            options=sorted(df['appliance'].unique()),
            default=sorted(df['appliance'].unique())
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

            # KPI Cards
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

            # Monthly and Appliance Insights
            st.markdown("### **_Monthly and Appliance Insights_**")
            row2_col1, row2_col2, row2_col3 = st.columns([1, 1, 1])

            # Grid Usage Chart
            with row2_col1:
                unique_months = filtered_df["date"].dt.to_period("M").unique()
                if len(unique_months) >= 2:
                    sorted_months = sorted(unique_months)
                    last_month = sorted_months[-1]
                    prev_month = sorted_months[-2]
                    
                    last_month_data = filtered_df[filtered_df["date"].dt.to_period("M") == last_month]
                    prev_month_data = filtered_df[filtered_df["date"].dt.to_period("M") == prev_month]
                    
                    last_month_data["day_of_month"] = last_month_data["date"].dt.day
                    prev_month_data["day_of_month"] = prev_month_data["date"].dt.day

                    last_month_data["label"] = f"{last_month.start_time.strftime('%B %Y')}"
                    prev_month_data["label"] = f"{prev_month.start_time.strftime('%B %Y')}"

                    combined_data = pd.concat([last_month_data, prev_month_data])
                    combined_data = combined_data.groupby(["day_of_month", "label"])["usage"].sum().reset_index()

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

            # Rest of your visualization code remains the same...

        # AI Insights Tab
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