import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timezone, timedelta

# Set timezone
KST = timezone(timedelta(hours=9))

@st.cache_data  # Add caching for better performance
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

@st.cache_data  # Add caching for better performance
def load_dashboard_data():
    """
    Load and preprocess data files from the repository
    """
    # Load glucose data
    glucose_df = pd.read_csv('data/glucose_data.csv', parse_dates=['DateTime'], 
                            date_parser=parse_datetime_with_timezone)
    glucose_df = glucose_df[glucose_df['IsInterpolated'] == False]
    glucose_df = glucose_df[['MeasurementNumber', 'DateTime', 'GlucoseValue']]
    glucose_df = glucose_df.dropna(subset=['DateTime'])

    # Load meal data
    meal_df = pd.read_csv('data/meal_data.csv', parse_dates=['meal_time'], 
                         date_parser=parse_datetime_with_timezone)
    meal_df = meal_df[[
        'measurement_number', 'meal_time', 'food_name', 'calories',
        'carbohydrates', 'sugars', 'protein', 'fat',
        'saturated_fat', 'trans_fat', 'cholesterol', 'sodium', 'meal_type'
    ]]
    meal_df = meal_df[meal_df['meal_type'] != 'Snack']
    meal_df = meal_df.dropna(subset=['meal_time'])
    meal_df = meal_df.reset_index(drop=True)

    # Load activity data
    activity_df = pd.read_csv('data/activity_data.csv', parse_dates=['start_time', 'end_time'], 
                             date_parser=parse_datetime_with_timezone)
    activity_df = activity_df[[
        'start_time', 'end_time', 'steps', 'distance', 
        'flights', 'activity_level'
    ]]
    activity_df = activity_df.dropna(subset=['start_time', 'end_time'])
    
    return glucose_df, meal_df, activity_df

def get_activity_color(activity_level):
    """Returns a color based on predefined activity levels"""
    # Your existing get_activity_color function code here
    color_map = {
        'Inactive': '#ACABB0',    
        'Light': '#C298A0',       
        'Moderate': '#D68591',    
        'Active': '#E95F73',      
        'Very Active': '#EC3F54', 
        'Intense': '#E01C34'      
    }
    
    base_color = color_map.get(activity_level, '#ACABB0')
    rgb = tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]})"

def create_glucose_meal_activity_chart(glucose_df, meal_df, activity_df, selected_meal):
    """Your existing chart creation function"""
    # Your existing create_glucose_meal_activity_chart function code here
    # No changes needed to this function

def run_streamlit_app():
    st.title('Blood Glucose Analysis Dashboard')
    
    try:
        # Load data directly from repository
        glucose_df, meal_df, activity_df = load_dashboard_data()
        
        # Add data summary in sidebar
        st.sidebar.markdown("### Dataset Information")
        st.sidebar.markdown(f"Total Records:\n"
                          f"- Glucose: {len(glucose_df):,} measurements\n"
                          f"- Meals: {len(meal_df):,} records\n"
                          f"- Activities: {len(activity_df):,} records")
        
        # Add date filter
        min_date = glucose_df['DateTime'].min().date()
        max_date = glucose_df['DateTime'].max().date()
        
        date_range = st.sidebar.date_input(
            "Filter by Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Filter data by selected date range
        start_date, end_date = date_range
        meal_df_filtered = meal_df[
            (meal_df['meal_time'].dt.date >= start_date) &
            (meal_df['meal_time'].dt.date <= end_date)
        ]
        
        # Create meal selection dropdown with additional information
        meal_options = {
            f"{meal['meal_time'].tz_convert(KST).strftime('%Y-%m-%d %H:%M')} - "
            f"{meal['food_name']} (Carbs: {meal['carbohydrates']:.0f}g)": i 
            for i, meal in meal_df_filtered.iterrows()
        }
        
        if meal_options:
            selected_meal_label = st.selectbox(
                'Select a meal to view:',
                options=list(meal_options.keys())
            )
            
            selected_meal_idx = meal_options[selected_meal_label]
            
            # Create and display the plot
            fig = create_glucose_meal_activity_chart(
                glucose_df, meal_df, activity_df, selected_meal_idx
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed meal information
            st.subheader('Detailed Meal Information')
            meal_data = meal_df.loc[selected_meal_idx]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Basic Information:")
                st.write(f"Food: {meal_data['food_name']}")
                st.write(f"Meal Type: {meal_data['meal_type']}")
                st.write(f"Calories: {meal_data['calories']:.1f} kcal")
                st.write(f"Measurement Number: {meal_data['measurement_number']}")
                
            with col2:
                st.write("Nutritional Information:")
                st.write(f"Carbohydrates: {meal_data['carbohydrates']:.1f}g")
                st.write(f"Sugars: {meal_data['sugars']:.1f}g")
                st.write(f"Protein: {meal_data['protein']:.1f}g")
                st.write(f"Total Fat: {meal_data['fat']:.1f}g")
                st.write(f"- Saturated Fat: {meal_data['saturated_fat']:.1f}g")
                st.write(f"- Trans Fat: {meal_data['trans_fat']:.1f}g")
                st.write(f"Cholesterol: {meal_data['cholesterol']:.1f}mg")
                st.write(f"Sodium: {meal_data['sodium']:.1f}mg")
        else:
            st.info('No meals found in the selected date range.')
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure data files are present in the 'data' directory.")

if __name__ == '__main__':
    run_streamlit_app()