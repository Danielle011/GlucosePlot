import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timezone, timedelta

# Set timezone
KST = timezone(timedelta(hours=9))

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

def load_data(glucose_file, meal_file, activity_file):
    """
    Load and preprocess data files with updated column structures
    """
    # Load glucose data
    glucose_df = pd.read_csv(glucose_file, parse_dates=['DateTime'], date_parser=parse_datetime_with_timezone)
    glucose_df = glucose_df[glucose_df['IsInterpolated'] == False]
    glucose_df = glucose_df[['MeasurementNumber', 'DateTime', 'GlucoseValue']]
    glucose_df = glucose_df.dropna(subset=['DateTime'])

    # Load meal data
    meal_df = pd.read_csv(meal_file, parse_dates=['meal_time'], date_parser=parse_datetime_with_timezone)
    meal_df = meal_df[[
        'measurement_number', 'meal_time', 'food_name', 'calories',
        'carbohydrates', 'sugars', 'protein', 'fat',
        'saturated_fat', 'trans_fat', 'cholesterol', 'sodium', 'meal_type'
    ]]
    # Filter out snacks
    meal_df = meal_df[meal_df['meal_type'] != 'Snack']
    meal_df = meal_df.dropna(subset=['meal_time'])
    # Reset index after filtering
    meal_df = meal_df.reset_index(drop=True)

    # Load activity data with updated structure
    activity_df = pd.read_csv(activity_file, parse_dates=['start_time', 'end_time'], date_parser=parse_datetime_with_timezone)
    activity_df = activity_df[[
        'start_time', 'end_time', 'steps', 'distance', 
        'flights', 'activity_level'
    ]]
    activity_df = activity_df.dropna(subset=['start_time', 'end_time'])
    
    return glucose_df, meal_df, activity_df

# Remove these functions as they're no longer needed
# def get_activity_score(row):  # Remove this
# def get_activity_color(score): # Replace with new version

def get_activity_color(activity_level):
    """Returns a color based on predefined activity levels"""
    # Define color mapping for each activity level
    color_map = {
        'Inactive': '#ACABB0',    # Light gray
        'Light': '#C298A0',       # Light pink-gray
        'Moderate': '#D68591',    # Pink
        'Active': '#E95F73',      # Light red
        'Very Active': '#EC3F54', # Bright red
        'Intense': '#E01C34'      # Deep red
    }
    
    # Get base color and add opacity
    base_color = color_map.get(activity_level, '#ACABB0')  # Default to gray if level not found
    rgb = tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.3)"

def create_glucose_meal_activity_chart(glucose_df, meal_df, activity_df, selected_meal):
    """Creates an interactive plotly figure with enhanced styling and readability"""
    meal_time = meal_df.loc[selected_meal, 'meal_time']
    next_meal_time = meal_df[meal_df['meal_time'] > meal_time].iloc[0]['meal_time'] if len(meal_df[meal_df['meal_time'] > meal_time]) > 0 else None
    end_time = min(meal_time + pd.Timedelta(hours=2), next_meal_time) if next_meal_time else meal_time + pd.Timedelta(hours=2)
    
    # Filter data for the specific time window
    glucose_window = glucose_df[(glucose_df['DateTime'] >= meal_time) & 
                               (glucose_df['DateTime'] <= end_time)]
    activity_window = activity_df[(activity_df['start_time'] >= meal_time) & 
                                (activity_df['end_time'] <= end_time)].copy()
    
    # Calculate activity scores
    activity_window['activity_score'] = activity_window.apply(get_activity_score, axis=1)
    
    # Add relative time in minutes to glucose data
    glucose_window = glucose_window.copy()
    glucose_window['minutes_from_meal'] = ((glucose_window['DateTime'] - meal_time).dt.total_seconds() / 60).round().astype(int)
    
    # Format meal information for subtitle with enhanced styling
    meal_data = meal_df.loc[selected_meal]
    meal_subtitle = (
        f"{meal_data['food_name']} | "
        f"Calories: {meal_data['calories']:.0f} | "
        f"Carbs: {meal_data['carbohydrates']:.1f}g | "
        f"Protein: {meal_data['protein']:.1f}g | "
        f"Fat: {meal_data['fat']:.1f}g"
    )
    
    fig = go.Figure()
    
    # Add activity data as background shading
    for idx, activity in activity_window[activity_window['steps'] > 100].iterrows():
        score = activity['activity_score']
        color = get_activity_color(score)
        
        # Calculate minutes from meal for activity times
        start_minutes = int(((activity['start_time'] - meal_time).total_seconds() / 60))
        end_minutes = int(((activity['end_time'] - meal_time).total_seconds() / 60))
        
        fig.add_trace(
            go.Scatter(
                x=[activity['start_time'], activity['start_time'], 
                   activity['end_time'], activity['end_time']],
                y=[0, 200, 200, 0],
                fill='toself',
                mode='none',
                name='Activity',
                fillcolor=color,
                customdata=[[
                    f"+{start_minutes}",
                    f"+{end_minutes}",
                    int(activity["steps"]),
                    activity["distance"],
                    int(activity["flights"])
                ]],
                hovertemplate=(
                    '<b>Activity Data</b><br>' +
                    'Time: %{customdata[0]} min to %{customdata[1]} min<br>' +
                    'Steps: %{customdata[2]}<br>' +
                    'Distance: %{customdata[3]:.2f} km<br>' +
                    'Flights: %{customdata[4]}<extra></extra>'
                ),
                hoveron='fills',
                showlegend=False,
            )
        )
    
    # Add glucose data with enhanced hover
    fig.add_trace(
        go.Scatter(
            x=glucose_window['DateTime'],
            y=glucose_window['GlucoseValue'],
            mode='lines+markers',
            name='Glucose',
            line=dict(color='#000035', width=1.5),
            marker=dict(size=5),
            customdata=glucose_window['minutes_from_meal'],
            hovertemplate=(
                '<b>Time:</b> +%{customdata} min<br>' +
                '<b>Glucose:</b> %{y:.0f} mg/dL<br>' +
                '<extra></extra>'
            )
        )
    )
    
    # Add reference lines
    fig.add_hline(
        y=180,
        line_dash="dot",
        line_color="rgba(200, 200, 200, 0.6)",
        line_width=1,
    )
    
    fig.add_hline(
        y=70,
        line_dash="dot",
        line_color="rgba(200, 200, 200, 0.6)",
        line_width=1,
    )
    
    # Create custom tick values every 15 minutes
    time_range = pd.date_range(start=meal_time, end=end_time, freq='15min')
    tick_values = time_range.tolist()
    tick_texts = [f"+{int((t - meal_time).total_seconds() / 60)}" for t in time_range]
    
    # Update layout with improved styling
    fig.update_layout(
        title=dict(
            text=(
                f'Blood Glucose Pattern after Meal on {meal_time.strftime("%Y-%m-%d %H:%M")}<br>'
                f'<span style="font-size: 14px; color: #000035; background-color: #f8f9fa; padding: 5px; '
                f'border-radius: 4px; margin-top: 8px; display: inline-block">{meal_subtitle}</span>'
            ),
            font=dict(size=16),
            y=0.95,
            x=0,
            xanchor='left'
        ),
        xaxis=dict(
            title='Time (minutes from meal)',
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=False,
            ticktext=tick_texts,  # Custom tick labels
            tickvals=tick_values,  # Custom tick positions
            title_font=dict(size=12),
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title='Blood Glucose (mg/dL)',
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=False,
            title_font=dict(size=12),
            tickfont=dict(size=10),
            range=[0, max(200, glucose_window['GlucoseValue'].max() * 1.1)]
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        showlegend=False,
        margin=dict(t=100, l=60, r=20, b=60),
    )
    
    # Set x-axis range
    fig.update_xaxes(range=[meal_time, end_time])
    
    return fig

def run_streamlit_app():
    st.title('Blood Glucose Analysis Dashboard')
    
    # File uploaders
    st.sidebar.header('Upload Data Files')
    glucose_file = st.sidebar.file_uploader("Upload Glucose Data", type=['csv'])
    meal_file = st.sidebar.file_uploader("Upload Meal Data", type=['csv'])
    activity_file = st.sidebar.file_uploader("Upload Activity Data", type=['csv'])
    
    if all([glucose_file, meal_file, activity_file]):
        # Load data
        glucose_df, meal_df, activity_df = load_data(
            glucose_file, meal_file, activity_file
        )
        
        # Create meal selection dropdown grouped by measurement number
        meal_options = {
            f"#{meal['measurement_number']} - {meal['meal_time'].tz_convert(KST).strftime('%Y-%m-%d %H:%M')} - {meal['food_name']}": i 
            for i, meal in meal_df.iterrows()
        }
        
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
        
        # Create two columns for better layout
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
        st.info('Please upload all data files to begin analysis.')

if __name__ == '__main__':
    run_streamlit_app()