import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timezone, timedelta

# Set timezone
KST = timezone(timedelta(hours=9))

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_glucose_data():
    """Load pre-filtered glucose data"""
    return pd.read_csv(
        'data/original_glucose_data.csv',
        usecols=['MeasurementNumber', 'DateTime', 'GlucoseValue'],
        parse_dates=['DateTime']
    )

@st.cache_data(ttl=3600)
def load_full_meal_data():
    """Load meal data including snacks"""
    return pd.read_csv(
        'data/processed_meal_data.csv',
        parse_dates=['meal_time']
    )

@st.cache_data(ttl=3600)
def load_meal_data():
    """Load and preprocess meal data, excluding snacks"""
    meal_df = load_full_meal_data()
    return meal_df[meal_df['meal_type'] != 'Snack'].reset_index(drop=True)

@st.cache_data(ttl=3600)
def load_activity_data():
    """Load and preprocess activity data"""
    return pd.read_csv(
        'data/activity_data_with_levels.csv',
        parse_dates=['start_time', 'end_time']
    )

@st.cache_data
def get_data_for_meal(glucose_df, activity_df, meal_time, meal_number, full_meal_df):
    """Efficiently get relevant glucose and activity data for a specific meal"""
    # Find the next meal time (including snacks)
    next_meal = full_meal_df[full_meal_df['meal_time'] > meal_time].iloc[0] if not full_meal_df[full_meal_df['meal_time'] > meal_time].empty else None
    
    # Set end time based on next meal or default 2-hour window
    if next_meal is not None and (next_meal['meal_time'] - meal_time) <= pd.Timedelta(hours=2):
        end_time = next_meal['meal_time']
    else:
        end_time = meal_time + pd.Timedelta(hours=2)
    
    # Get glucose data for this meal
    glucose_window = glucose_df[
        (glucose_df['MeasurementNumber'] == meal_number) &
        (glucose_df['DateTime'] >= meal_time) & 
        (glucose_df['DateTime'] <= end_time)
    ].copy()
    
    # Get activity data for this meal
    activity_window = activity_df[
        (activity_df['start_time'] >= meal_time) & 
        (activity_df['end_time'] <= end_time)
    ].copy()
    
    return glucose_window, activity_window, end_time

def get_activity_color_gradient(activity_level):
    """Returns a color based on activity level using a gradient scale"""
    level_map = {
        'Inactive': 0.1,
        'Light': 0.2,
        'Moderate': 0.3,
        'Active': 0.7,
        'Very Active': 0.85,
        'Intense': 1.0
    }
    opacity = level_map.get(activity_level, 0.1)
    return f"rgba(255, 0, 0, {opacity})"

def format_time_12hr(dt):
    """Convert datetime to 12-hour format string"""
    return dt.strftime("%I:%M %p").lstrip("0")

def create_glucose_meal_activity_chart_gradient(glucose_window, meal_data, activity_window, end_time, selected_idx=0):
    """Creates chart with gradient colors for activities"""
    meal_time = meal_data.iloc[selected_idx]['meal_time']
    
    # Add relative time in minutes to glucose data
    glucose_window['minutes_from_meal'] = (
        (glucose_window['DateTime'] - meal_time).dt.total_seconds() / 60
    ).round().astype(int)
    
    # Format meal information for subtitle
    meal = meal_data.iloc[selected_idx]
    meal_subtitle = (
        f"{meal['food_name']} | "
        f"Calories: {meal['calories']:.0f} | "
        f"Carbs: {meal['carbohydrates']:.1f}g | "
        f"Protein: {meal['protein']:.1f}g | "
        f"Fat: {meal['fat']:.1f}g"
    )
    
    fig = go.Figure()
    
    # Add activity data as background shading with gradient colors
    for _, activity in activity_window[activity_window['steps'] > 100].iterrows():
        color = get_activity_color_gradient(activity['activity_level'])
        
        # Format start and end times
        start_time_str = format_time_12hr(activity['start_time'])
        end_time_str = format_time_12hr(activity['end_time'])
        
        # Format steps with thousand separator
        steps_str = f"{int(activity['steps']):,}"
        
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
                    start_time_str,
                    end_time_str,
                    steps_str,
                    activity["distance"],
                    int(activity["flights"]),
                    activity["activity_level"],
                    color  # Pass color to use in hover template
                ]],
                hovertemplate=(
                    '<br>'.join([
                        '<b>%{customdata[0]} - %{customdata[1]}</b>',
                        '<b>Steps:</b> %{customdata[2]} steps',
                        '<b>Distance:</b> %{customdata[3]:.1f} km',
                        '<b>Flights:</b> %{customdata[4]} flights',
                        '<span style="display: inline-block; margin-top: 5px; padding: 2px 6px; ' +
                        'border-radius: 3px; background-color: %{customdata[6]}; color: white;">' +
                        '%{customdata[5]}</span>'
                    ]) + '<extra></extra>'
                ),
                hoveron='fills',
                showlegend=False,
            )
        )
    
    # Add glucose data
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
    fig.add_hline(y=180, line_dash="dot", line_color="rgba(200, 200, 200)", line_width=1)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(200, 200, 200)", line_width=1)
    
    # Create custom tick values every 15 minutes
    time_range = pd.date_range(start=meal_time, end=end_time, freq='15min')
    tick_values = time_range.tolist()
    tick_texts = [f"+{int((t - meal_time).total_seconds() / 60)}" for t in time_range]
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=(
                f'Blood Glucose Pattern after Meal on {meal_time.strftime("%Y-%m-%d %H:%M")}<br>'
                f'<span style="font-size: 14px; color: #000035; background-color: #f8f9fa; '
                f'padding: 5px; border-radius: 4px; margin-top: 8px; display: inline-block">'
                f'{meal_subtitle}</span>'
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
            ticktext=tick_texts,
            tickvals=tick_values,
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
    
    fig.update_xaxes(range=[meal_time, end_time])
    
    return fig

def run_streamlit_app():
    st.set_page_config(page_title="Glucose Analysis", layout="wide")
    st.title('Blood Glucose Analysis Dashboard')
    
    try:
        # Load data with progress indicators
        with st.spinner('Loading data...'):
            glucose_df = load_glucose_data()
            meal_df = load_meal_data()
            activity_df = load_activity_data()
            full_meal_df = load_full_meal_data()
        
        st.sidebar.markdown("### Dataset Information")
        date_min = meal_df['meal_time'].dt.date.min()
        date_max = meal_df['meal_time'].dt.date.max()
        
        st.sidebar.markdown(f"Available Date Range:\n"
                          f"{date_min} to {date_max}\n\n"
                          f"Total Meals: {len(meal_df):,}")
        
        # Default to showing most recent week
        default_end_date = date_max
        default_start_date = max(date_min, default_end_date - pd.Timedelta(days=7))
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(default_start_date, default_end_date),
            min_value=date_min,
            max_value=date_max
        )
        
        # Filter meals by date
        start_date, end_date = date_range
        meal_mask = (meal_df['meal_time'].dt.date >= start_date) & \
                   (meal_df['meal_time'].dt.date <= end_date)
        meals_filtered = meal_df[meal_mask]
        
        if not meals_filtered.empty:
            # Create meal selection dropdown with useful information
            meal_options = {
                f"{meal['meal_time'].strftime('%Y-%m-%d %H:%M')} - "
                f"{meal['food_name']} "
                f"(Carbs: {meal['carbohydrates']:.0f}g)": idx
                for idx, meal in meals_filtered.iterrows()
            }
            
            selected_meal_label = st.selectbox(
                'Select a meal to view:',
                options=list(meal_options.keys())
            )
            
            selected_meal_idx = meal_options[selected_meal_label]
            selected_meal = meals_filtered.loc[selected_meal_idx]
            
            # Get data for selected meal
            glucose_window, activity_window, end_time = get_data_for_meal(
                glucose_df, 
                activity_df,
                selected_meal['meal_time'],
                selected_meal['measurement_number'],
                full_meal_df
            )

            # Calculate window duration
            window_duration = (end_time - selected_meal['meal_time']).total_seconds() / 60
            
            # Create container for activity level legend
            st.markdown("### Activity Level Color Guide")
            for level, opacity in {
                'Inactive': 0.1,
                'Light': 0.2,
                'Moderate': 0.3,
                'Active': 0.7,
                'Very Active': 0.85,
                'Intense': 1.0
            }.items():
                st.markdown(
                    f'<div style="background-color: rgba(255,0,0,{opacity}); padding: 5px; '
                    f'margin: 2px; border-radius: 3px;">{level}</div>',
                    unsafe_allow_html=True
                )
            
            # Display chart
            fig = create_glucose_meal_activity_chart_gradient(
                glucose_window, 
                pd.DataFrame([selected_meal]), 
                activity_window,
                end_time,
                0
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics in sidebar
            st.sidebar.markdown("### Window Information")
            st.sidebar.markdown(f"- Duration: {window_duration:.0f} minutes")
            
            # Calculate glucose metrics
            initial_glucose = glucose_window.iloc[0]['GlucoseValue']
            peak_glucose = glucose_window['GlucoseValue'].max()
            peak_time = (glucose_window.loc[glucose_window['GlucoseValue'].idxmax(), 'DateTime'] - 
                        selected_meal['meal_time']).total_seconds() / 60
            
            st.sidebar.markdown("### Glucose Response")
            st.sidebar.markdown(f"- Initial: {initial_glucose:.0f} mg/dL")
            st.sidebar.markdown(f"- Peak: {peak_glucose:.0f} mg/dL")
            st.sidebar.markdown(f"- Time to Peak: {peak_time:.0f} min")
            
            st.sidebar.markdown("### Meal Content")
            st.sidebar.markdown(f"- Food: {selected_meal['food_name']}")
            st.sidebar.markdown(f"- Calories: {selected_meal['calories']:.0f} kcal")
            st.sidebar.markdown(f"- Carbs: {selected_meal['carbohydrates']:.0f}g")
            st.sidebar.markdown(f"- Protein: {selected_meal['protein']:.0f}g")
            st.sidebar.markdown(f"- Fat: {selected_meal['fat']:.0f}g")
            
            if not activity_window.empty:
                st.sidebar.markdown("### Activity")
                for _, activity in activity_window.iterrows():
                    minutes_from_meal = (activity['start_time'] - 
                                       selected_meal['meal_time']).total_seconds() / 60
                    st.sidebar.markdown(
                        f"- At +{minutes_from_meal:.0f} min:\n"
                        f"  - Steps: {activity['steps']:,}\n"
                        f"  - Level: {activity['activity_level']}"
                    )
        else:
            st.info('No meals found in the selected date range.')
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please check if all required data files are present in the data directory.")

if __name__ == '__main__':
    run_streamlit_app()