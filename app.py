import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timezone, timedelta

KST = timezone(timedelta(hours=9))

# Keep your existing data loading and processing functions
def parse_datetime_with_timezone(date_string):
    if pd.isna(date_string) or date_string == '0':
        return pd.NaT
    try:
        return pd.to_datetime(date_string, format='%Y-%m-%d %H:%M:%S%z')
    except ValueError:
        try:
            return pd.to_datetime(date_string).tz_localize(KST)
        except ValueError:
            return pd.NaT

def load_data(glucose_file, meal_file, activity_file, features_file):
    # Your existing load_data function stays the same
    glucose_df = pd.read_csv(glucose_file, parse_dates=['DateTime'], date_parser=parse_datetime_with_timezone)
    glucose_df = glucose_df[glucose_df['IsInterpolated'] == False]
    glucose_df = glucose_df[['DateTime', 'GlucoseValue']]
    glucose_df = glucose_df.dropna(subset=['DateTime'])

    meal_df = pd.read_csv(meal_file, parse_dates=['meal_time'], date_parser=parse_datetime_with_timezone)
    meal_df = meal_df[['meal_time', 'food_name', 'carbohydrates', 'sugars', 'protein', 'fat', 'meal_type']]
    meal_df = meal_df.dropna(subset=['meal_time'])

    activity_df = pd.read_csv(activity_file, parse_dates=['start_time', 'end_time'], date_parser=parse_datetime_with_timezone)
    activity_df = activity_df[['start_time', 'end_time', 'steps', 'distance']]
    activity_df = activity_df.dropna(subset=['start_time', 'end_time'])

    features_df = pd.read_csv(features_file, parse_dates=['meal_time'], date_parser=parse_datetime_with_timezone)
    features_df = features_df.dropna(subset=['meal_time'])
    
    return glucose_df, meal_df, activity_df, features_df

def create_glucose_meal_activity_chart(glucose_df, meal_df, activity_df, features_df, selected_meal):
    # Your existing chart creation function stays the same
    # (Keep all the existing code for this function)
    meal_time = meal_df.loc[selected_meal, 'meal_time'].tz_convert(KST)
    next_meal_time = meal_df[meal_df['meal_time'] > meal_time].iloc[0]['meal_time'].tz_convert(KST) if len(meal_df[meal_df['meal_time'] > meal_time]) > 0 else None
    end_time = min(meal_time + pd.Timedelta(hours=2), next_meal_time) if next_meal_time else meal_time + pd.Timedelta(hours=2)
    
    # Rest of your existing function code...
    return fig

# New Streamlit app code
def run_streamlit_app():
    st.title('Blood Glucose Analysis Dashboard')
    
    # File uploaders
    st.sidebar.header('Upload Data Files')
    glucose_file = st.sidebar.file_uploader("Upload Glucose Data", type=['csv'])
    meal_file = st.sidebar.file_uploader("Upload Meal Data", type=['csv'])
    activity_file = st.sidebar.file_uploader("Upload Activity Data", type=['csv'])
    features_file = st.sidebar.file_uploader("Upload Features Data", type=['csv'])
    
    if all([glucose_file, meal_file, activity_file, features_file]):
        # Load data
        glucose_df, meal_df, activity_df, features_df = load_data(
            glucose_file, meal_file, activity_file, features_file
        )
        
        # Create meal selection dropdown
        meal_options = {
            f"{meal['meal_time'].tz_convert(KST).strftime('%Y-%m-%d %H:%M')} - {meal['food_name']}": i 
            for i, meal in meal_df.iterrows()
        }
        
        selected_meal_label = st.selectbox(
            'Select a meal to view:',
            options=list(meal_options.keys())
        )
        
        selected_meal_idx = meal_options[selected_meal_label]
        
        # Create and display the plot
        fig = create_glucose_meal_activity_chart(
            glucose_df, meal_df, activity_df, features_df, selected_meal_idx
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display additional meal information
        st.subheader('Meal Details')
        meal_data = meal_df.loc[selected_meal_idx]
        st.write(f"Food: {meal_data['food_name']}")
        st.write(f"Carbohydrates: {meal_data['carbohydrates']}g")
        st.write(f"Sugars: {meal_data['sugars']}g")
        st.write(f"Protein: {meal_data['protein']}g")
        st.write(f"Fat: {meal_data['fat']}g")

    else:
        st.info('Please upload all data files to begin analysis.')

if __name__ == '__main__':
    run_streamlit_app()