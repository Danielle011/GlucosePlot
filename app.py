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

    # Load activity data
    activity_df = pd.read_csv(activity_file, parse_dates=['start_time', 'end_time'], date_parser=parse_datetime_with_timezone)
    activity_df = activity_df[['start_time', 'end_time', 'steps', 'distance', 'flights']]
    activity_df = activity_df.dropna(subset=['start_time', 'end_time'])
    
    return glucose_df, meal_df, activity_df

def get_activity_score(row):
    """Calculate activity score based on distance and flights"""
    # Thresholds for distance (km per 10min)
    distance_thresholds = {
        'light': 0.5,
        'moderate': 1.0,
        'vigorous': 1.5
    }
    
    # Thresholds for flights (per 10min)
    flights_thresholds = {
        'light': 2,
        'moderate': 5,
        'vigorous': 10
    }
    
    # Calculate individual scores
    distance_score = (
        3 if row['distance'] >= distance_thresholds['vigorous'] else
        2 if row['distance'] >= distance_thresholds['moderate'] else
        1 if row['distance'] >= distance_thresholds['light'] else 0
    )
    
    flights_score = (
        3 if row['flights'] >= flights_thresholds['vigorous'] else
        2 if row['flights'] >= flights_thresholds['moderate'] else
        1 if row['flights'] >= flights_thresholds['light'] else 0
    )
    
    # Weighted combination (60% distance, 40% flights)
    return (distance_score * 0.6) + (flights_score * 0.4)

def get_activity_color(score):
    """Returns an RGBA color based on activity score"""
    if score == 0:
        return "rgba(200, 200, 200, 0.2)"  # Very light gray for inactive
    elif score <= 1:
        return f"rgba(255, 165, 0, {0.3 + score * 0.2})"  # Light orange
    elif score <= 2:
        return f"rgba(255, 69, 0, {0.4 + (score-1) * 0.2})"  # Darker orange
    else:
        return f"rgba(255, 0, 0, {0.5 + (min(score-2, 1) * 0.3)})"  # Red

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
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add glucose data with enhanced hover
    fig.add_trace(
        go.Scatter(
            x=glucose_window['DateTime'],
            y=glucose_window['GlucoseValue'],
            mode='lines+markers',
            name='Glucose',
            line=dict(color='#2C5282', width=2),  # Darker blue for better visibility
            marker=dict(size=8),
            hovertemplate=(
                '<b>Glucose Level</b><br>' +
                '%{y:.0f} mg/dL<br>' +
                '<i>Time: %{x|%H:%M}</i><extra></extra>'
            )
        ),
        secondary_y=False,
    )
    
    # Add activity data with improved visuals
    annotations = []  # Store annotations to manage positioning
    
    for idx, activity in activity_window[activity_window['steps'] > 100].iterrows():
        score = activity['activity_score']
        color = get_activity_color(score)
        
        # Add activity rectangle with hover info
        fig.add_trace(
            go.Scatter(
                x=[activity['start_time'], activity['start_time'], 
                   activity['end_time'], activity['end_time']],
                y=[0, 200, 200, 0],  # Full height rectangle
                fill='toself',
                mode='none',
                name=f'Activity',
                fillcolor=color,
                hoverinfo='text',
                hovertext=(
                    f'<b>Activity Period</b><br>' +
                    f'Time: {activity["start_time"].strftime("%H:%M")} - {activity["end_time"].strftime("%H:%M")}<br>' +
                    f'Steps: {int(activity["steps"])}<br>' +
                    f'Distance: {activity["distance"]:.2f} km<br>' +
                    f'Flights: {activity["flights"]:.1f}'
                ),
                showlegend=False,
            ),
            secondary_y=False,
        )
        
        # Calculate y-position for annotation to prevent overlap
        y_pos = glucose_window['GlucoseValue'].max() + (10 * (idx % 3))  # Stagger vertically
        
        # Create annotation with minimal text
        annotations.append(dict(
            x=activity['start_time'],
            y=y_pos,
            text=f"Steps: {int(activity['steps'])}",
            showarrow=False,
            font=dict(size=10, color="#1A365D"),  # Darker color for better readability
            bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent white background
            bordercolor=color,
            borderwidth=1,
            borderpad=4,
            yshift=0
        ))
    
    # Add meal details annotation
    meal_details = (
        f"<b>{meal_df.loc[selected_meal, 'food_name']}</b><br>" +
        f"Calories: {meal_df.loc[selected_meal, 'calories']:.0f} kcal<br>" +
        f"Carbs: {meal_df.loc[selected_meal, 'carbohydrates']:.1f}g<br>" +
        f"Sugar: {meal_df.loc[selected_meal, 'sugars']:.1f}g<br>" +
        f"Protein: {meal_df.loc[selected_meal, 'protein']:.1f}g<br>" +
        f"Fat: {meal_df.loc[selected_meal, 'fat']:.1f}g"
    )
    
    annotations.append(dict(
        x=meal_time,
        y=glucose_window['GlucoseValue'].max(),
        text=meal_details,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#4A5568",
        ax=0,
        ay=-60,
        font=dict(size=11),
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="#4A5568",
        borderwidth=1,
        borderpad=8,
        align='left'
    ))
    
    # Add reference lines with improved styling
    fig.add_hline(
        y=140,
        line_dash="dash",
        line_color="rgba(245, 101, 101, 0.5)",  # Lighter red
        annotation=dict(
            text="High",
            font=dict(color="#C53030"),
            xref="paper",
            x=1.02,
            showarrow=False
        )
    )
    
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="rgba(245, 101, 101, 0.5)",  # Lighter red
        annotation=dict(
            text="Low",
            font=dict(color="#C53030"),
            xref="paper",
            x=1.02,
            showarrow=False
        )
    )
    
    # Update layout with improved styling
    fig.update_layout(
        title=dict(
            text=f'Blood Glucose Pattern after Meal<br><sup>{meal_time.strftime("%Y-%m-%d %H:%M")}</sup>',
            font=dict(size=20),
            y=0.95
        ),
        xaxis=dict(
            title='Time',
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=False,
            dtick='M30',  # 30-minute intervals
            tickformat='%H:%M'
        ),
        yaxis=dict(
            title='Blood Glucose (mg/dL)',
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=False,
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        annotations=annotations,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        margin=dict(t=100),  # Increased top margin for title
    )
    
    fig.update_yaxes(range=[0, max(200, glucose_window['GlucoseValue'].max() * 1.1)], secondary_y=False)
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