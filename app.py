import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timezone, timedelta
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

@st.cache_data(ttl=3600)
def load_workout_data():
    """Load workout data"""
    return pd.read_csv(
        'combined_workouts.csv',
        parse_dates=['start_time', 'end_time']
    )

@st.cache_data(ttl=3600)
def load_workout_glucose_data():
    """Load glucose data for workouts"""
    return pd.read_csv(
        'processed_glucose_data.csv',
        parse_dates=['DateTime']
    )

@st.cache_data(ttl=3600)
def load_workout_heart_rate_data():
    """Load heart rate data for workouts"""
    return pd.read_csv(
        'heart_rate_data.csv',
        parse_dates=['start_date']
    )

@st.cache_data(ttl=3600)
def load_workout_meal_data():
    """Load meal data for workouts"""
    return pd.read_csv(
        'processed_meal_data.csv',
        parse_dates=['meal_time']
    )

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
    opacity = level_map.get(activity_level, 0.2)
    return f"rgba(255, 0, 0, {opacity})"

def format_time_12hr(dt):
    """Convert datetime to 12-hour format string"""
    return dt.strftime("%I:%M %p").lstrip("0")

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

def preprocess_workout_data(workouts_df, heart_rate_df, glucose_df, meal_df, KST):
    """Preprocess all workout-related data with consistent column names and timezones"""
    # Process workout data
    workouts_df = workouts_df.copy()
    for col in ['start_time', 'end_time']:
        if workouts_df[col].dt.tz is None:
            workouts_df[col] = workouts_df[col].dt.tz_localize(KST)
        else:
            workouts_df[col] = workouts_df[col].dt.tz_convert(KST)
    
    # Process heart rate data
    heart_rate_df = heart_rate_df.copy()
    heart_rate_df['timestamp'] = pd.to_datetime(heart_rate_df['start_date'])
    if heart_rate_df['timestamp'].dt.tz is None:
        heart_rate_df['timestamp'] = heart_rate_df['timestamp'].dt.tz_localize(KST)
    else:
        heart_rate_df['timestamp'] = heart_rate_df['timestamp'].dt.tz_convert(KST)
    heart_rate_df['heart_rate'] = heart_rate_df['value']
    heart_rate_df = heart_rate_df[['timestamp', 'heart_rate']]
    
    # Process glucose data
    glucose_df = glucose_df.copy()
    glucose_df['timestamp'] = pd.to_datetime(glucose_df['DateTime'])
    if glucose_df['timestamp'].dt.tz is None:
        glucose_df['timestamp'] = glucose_df['timestamp'].dt.tz_localize(KST)
    else:
        glucose_df['timestamp'] = glucose_df['timestamp'].dt.tz_convert(KST)
    glucose_df['glucose'] = glucose_df['GlucoseValue']
    glucose_df = glucose_df[['timestamp', 'glucose']]
    
    # Process meal data
    meal_df = meal_df.copy()
    meal_df['timestamp'] = pd.to_datetime(meal_df['meal_time'])
    if meal_df['timestamp'].dt.tz is None:
        meal_df['timestamp'] = meal_df['timestamp'].dt.tz_localize(KST)
    else:
        meal_df['timestamp'] = meal_df['timestamp'].dt.tz_convert(KST)
    meal_df = meal_df[['timestamp', 'food_name']]
    
    return workouts_df, heart_rate_df, glucose_df, meal_df

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
        f"{meal_time.strftime('%I:%M %p')} | "
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
        
        # Create background shade
        fig.add_trace(
            go.Scatter(
                x=[activity['start_time'], activity['start_time'], 
                   activity['end_time'], activity['end_time']],
                y=[0, 200, 200, 0],
                fill='toself',
                mode='none',
                showlegend=False,
                fillcolor=color,
                hoverinfo='skip'
            )
        )
        
        # Create hover points
        hover_times = pd.date_range(
            start=activity['start_time'],
            end=activity['end_time'],
            periods=5
        )
        
        hover_text = (
            f"{format_time_12hr(activity['start_time'])} - {format_time_12hr(activity['end_time'])}<br>" +
            f"Steps: {int(activity['steps']):,} steps<br>" +
            f"Distance: {activity['distance']:.1f} km<br>" +
            f"Flights: {int(activity['flights'])} flights<br>" +
            f"<span style='background-color: {color}; color: white; padding: 2px 6px; border-radius: 3px;'>" +
            f"{activity['activity_level']}</span>"
        )
        
        fig.add_trace(
            go.Scatter(
                x=hover_times,
                y=[100] * len(hover_times),
                mode='markers',
                marker=dict(
                    size=20,
                    color='rgba(0,0,0,0)',
                    symbol='square',
                ),
                hoverinfo='text',
                hovertext=hover_text,
                showlegend=False,
                name='',
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
    
    # Create custom tick values every 15 minutes
    time_range = pd.date_range(start=meal_time, end=end_time, freq='15min')
    tick_values = time_range.tolist()
    tick_texts = [f"+{int((t - meal_time).total_seconds() / 60)}" for t in time_range]
    
    # Update layout with new styling
    fig.update_layout(
        title=dict(
            text=(
                f'Blood Glucose Pattern after Meal on {meal_time.strftime("%Y-%m-%d")}<br>'
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
            showgrid=False,
            zeroline=False,
            ticktext=tick_texts,
            tickvals=tick_values,
            title_font=dict(size=12, color='black'),
            tickfont=dict(size=10, color='black'),
            linecolor='black',
            mirror=False
        ),
        yaxis=dict(
            title='Blood Glucose (mg/dL)',
            showgrid=False,
            zeroline=False,
            title_font=dict(size=12, color='black'),
            tickfont=dict(size=10, color='black'),
            linecolor='black',
            mirror=False,
            range=[0, max(200, glucose_window['GlucoseValue'].max() * 1.1)]
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        showlegend=False,
        margin=dict(t=100, l=60, r=20, b=60)
    )
    
    # Add only 140 reference line
    fig.add_hline(y=140, line_dash="dot", line_color="rgba(0, 0, 0, 0.3)", line_width=1)
    
    fig.update_xaxes(range=[meal_time, end_time])
    
    return fig

# Add new functions for EDA visualizations
def create_glucose_distribution_plot(glucose_df):
    """Create glucose value distribution plot"""
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=glucose_df['GlucoseValue'],
        nbinsx=50,
        name='Distribution'
    ))
    
    # Add vertical line at 140
    fig.add_vline(x=140, line_dash="dash", line_color="red", annotation_text="Target Limit (140)")
    
    # Add stats annotation
    stats_text = (
        f"Mean: {glucose_df['GlucoseValue'].mean():.1f}<br>"
        f"Median: {glucose_df['GlucoseValue'].median():.1f}<br>"
        f"Std: {glucose_df['GlucoseValue'].std():.1f}<br>"
        f"95th percentile: {glucose_df['GlucoseValue'].quantile(0.95):.1f}"
    )
    
    fig.add_annotation(
        x=0.95,
        y=0.95,
        xref="paper",
        yref="paper",
        text=stats_text,
        showarrow=False,
        align="right",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    fig.update_layout(
        title="Distribution of Glucose Values",
        xaxis_title="Glucose Value (mg/dL)",
        yaxis_title="Count",
        showlegend=False
    )
    
    return fig

def create_daily_glucose_pattern(glucose_df):
    """Create average glucose by hour plot"""
    # Calculate hourly statistics
    glucose_df['Hour'] = glucose_df['DateTime'].dt.hour
    hourly_stats = glucose_df.groupby('Hour')['GlucoseValue'].agg(['mean', 'std']).reset_index()
    
    fig = go.Figure()
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=hourly_stats['Hour'],
        y=hourly_stats['mean'],
        mode='lines',
        name='Mean',
        line=dict(color='blue'),
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=hourly_stats['Hour'].tolist() + hourly_stats['Hour'].tolist()[::-1],
        y=(hourly_stats['mean'] + hourly_stats['std']).tolist() + 
          (hourly_stats['mean'] - hourly_stats['std']).tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,0,255,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='±1 SD'
    ))
    
    fig.update_layout(
        title="Average Glucose Pattern by Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title="Glucose Value (mg/dL)",
        showlegend=True
    )
    
    return fig

def create_meal_macronutrient_plot(meal_df):
    """Create macronutrient distribution by meal type plot"""
    fig = go.Figure()
    
    meal_types = meal_df['meal_type'].unique()
    
    for nutrient in ['carbohydrates', 'protein', 'fat']:
        fig.add_trace(go.Box(
            x=meal_df['meal_type'],
            y=meal_df[nutrient],
            name=nutrient.capitalize(),
            boxpoints='outliers'
        ))
    
    fig.update_layout(
        title="Macronutrient Distribution by Meal Type",
        xaxis_title="Meal Type",
        yaxis_title="Grams",
        boxmode='group'
    )
    
    return fig

def create_activity_pattern_plot(activity_df):
    """Create activity pattern plot"""
    # Calculate hourly averages
    activity_df['Hour'] = activity_df['start_time'].dt.hour
    hourly_stats = activity_df.groupby('Hour').agg({
        'steps': 'mean',
        'flights': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    # Add steps line
    fig.add_trace(go.Scatter(
        x=hourly_stats['Hour'],
        y=hourly_stats['steps'],
        name='Steps',
        line=dict(color='blue')
    ))
    
    # Add flights line on secondary y-axis
    fig.add_trace(go.Scatter(
        x=hourly_stats['Hour'],
        y=hourly_stats['flights'],
        name='Flights',
        line=dict(color='red'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Activity Patterns by Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title="Average Steps",
        yaxis2=dict(
            title="Average Flights",
            overlaying='y',
            side='right'
        )
    )
    
    return fig

# Add these imports at the top of your file
import plotly.express as px
import plotly.figure_factory as ff

# Add new functions for EDA visualizations
def create_glucose_distribution_plot(glucose_df):
    """Create glucose value distribution plot"""
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=glucose_df['GlucoseValue'],
        nbinsx=50,
        name='Distribution'
    ))
    
    # Add vertical line at 140
    fig.add_vline(x=140, line_dash="dash", line_color="red", annotation_text="Target Limit (140)")
    
    # Add stats annotation
    stats_text = (
        f"Mean: {glucose_df['GlucoseValue'].mean():.1f}<br>"
        f"Median: {glucose_df['GlucoseValue'].median():.1f}<br>"
        f"Std: {glucose_df['GlucoseValue'].std():.1f}<br>"
        f"95th percentile: {glucose_df['GlucoseValue'].quantile(0.95):.1f}"
    )
    
    fig.add_annotation(
        x=0.95,
        y=0.95,
        xref="paper",
        yref="paper",
        text=stats_text,
        showarrow=False,
        align="right",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    fig.update_layout(
        title="Distribution of Glucose Values",
        xaxis_title="Glucose Value (mg/dL)",
        yaxis_title="Count",
        showlegend=False
    )
    
    return fig

def create_daily_glucose_pattern(glucose_df):
    """Create average glucose by hour plot"""
    # Calculate hourly statistics
    glucose_df['Hour'] = glucose_df['DateTime'].dt.hour
    hourly_stats = glucose_df.groupby('Hour')['GlucoseValue'].agg(['mean', 'std']).reset_index()
    
    fig = go.Figure()
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=hourly_stats['Hour'],
        y=hourly_stats['mean'],
        mode='lines',
        name='Mean',
        line=dict(color='blue'),
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=hourly_stats['Hour'].tolist() + hourly_stats['Hour'].tolist()[::-1],
        y=(hourly_stats['mean'] + hourly_stats['std']).tolist() + 
          (hourly_stats['mean'] - hourly_stats['std']).tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,0,255,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='±1 SD'
    ))
    
    fig.update_layout(
        title="Average Glucose Pattern by Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title="Glucose Value (mg/dL)",
        showlegend=True
    )
    
    return fig

def create_meal_macronutrient_plot(meal_df):
    """Create macronutrient distribution by meal type plot"""
    fig = go.Figure()
    
    meal_types = meal_df['meal_type'].unique()
    
    for nutrient in ['carbohydrates', 'protein', 'fat']:
        fig.add_trace(go.Box(
            x=meal_df['meal_type'],
            y=meal_df[nutrient],
            name=nutrient.capitalize(),
            boxpoints='outliers'
        ))
    
    fig.update_layout(
        title="Macronutrient Distribution by Meal Type",
        xaxis_title="Meal Type",
        yaxis_title="Grams",
        boxmode='group'
    )
    
    return fig

def create_activity_pattern_plot(activity_df):
    """Create activity pattern plot"""
    # Calculate hourly averages
    activity_df['Hour'] = activity_df['start_time'].dt.hour
    hourly_stats = activity_df.groupby('Hour').agg({
        'steps': 'mean',
        'flights': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    # Add steps line
    fig.add_trace(go.Scatter(
        x=hourly_stats['Hour'],
        y=hourly_stats['steps'],
        name='Steps',
        line=dict(color='blue')
    ))
    
    # Add flights line on secondary y-axis
    fig.add_trace(go.Scatter(
        x=hourly_stats['Hour'],
        y=hourly_stats['flights'],
        name='Flights',
        line=dict(color='red'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Activity Patterns by Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title="Average Steps",
        yaxis2=dict(
            title="Average Flights",
            overlaying='y',
            side='right'
        )
    )
    
    return fig

def create_time_in_range_plot(glucose_df):
    """Create time in range pie chart"""
    time_in_range = (glucose_df['GlucoseValue'] <= 140).mean() * 100
    above_range = 100 - time_in_range
    
    fig = go.Figure(data=[go.Pie(
        labels=['≤140 mg/dL', '>140 mg/dL'],
        values=[time_in_range, above_range],
        hole=.3,
        marker_colors=['#2ecc71', '#e74c3c'],
        textinfo='label+percent',
        textposition='inside',
        insidetextorientation='radial'
    )])
    
    fig.update_layout(
        title="Time in Range Analysis",
        annotations=[dict(text=f"{time_in_range:.1f}%", x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    
    return fig

def create_meal_response_analysis(meal_df, glucose_df):
    """Create meal response analysis plots"""
    meal_responses = []
    
    for _, meal in meal_df.iterrows():
        # Get glucose data for 2 hours after meal
        post_meal = glucose_df[
            (glucose_df['DateTime'] >= meal['meal_time']) &
            (glucose_df['DateTime'] <= meal['meal_time'] + pd.Timedelta(hours=2))
        ]
        
        if not post_meal.empty:
            baseline = post_meal.iloc[0]['GlucoseValue']
            peak = post_meal['GlucoseValue'].max()
            # Fix the time calculation
            peak_time = (post_meal.loc[post_meal['GlucoseValue'].idxmax(), 'DateTime'] - 
                        meal['meal_time']).total_seconds() / 60
            
            meal_responses.append({
                'meal_type': meal['meal_type'],
                'carbs': meal['carbohydrates'],
                'glucose_rise': peak - baseline,
                'time_to_peak': peak_time,
                'baseline': baseline,
                'peak': peak
            })
    
    response_df = pd.DataFrame(meal_responses)
    
    # Create scatter plot
    fig = go.Figure()
    
    for meal_type in response_df['meal_type'].unique():
        mask = response_df['meal_type'] == meal_type
        fig.add_trace(go.Scatter(
            x=response_df[mask]['carbs'],
            y=response_df[mask]['glucose_rise'],
            mode='markers',
            name=meal_type,
            text=[f"Rise: {rise:.1f} mg/dL<br>Time to Peak: {time:.0f} min<br>Carbs: {carbs:.1f}g"
                  for rise, time, carbs in zip(
                      response_df[mask]['glucose_rise'],
                      response_df[mask]['time_to_peak'],
                      response_df[mask]['carbs'])],
            hoverinfo="text+name"
        ))
    
    fig.update_layout(
        title="Glucose Rise vs Carbohydrate Load by Meal Type",
        xaxis_title="Carbohydrates (g)",
        yaxis_title="Glucose Rise (mg/dL)",
        showlegend=True,
        height=600,
        template="plotly_white"
    )
    
    # Add reference lines
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(255,0,0,0.3)", 
                  annotation_text="30 mg/dL rise")
    fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,0,0,0.5)", 
                  annotation_text="50 mg/dL rise")
    
    return fig, response_df

def create_activity_impact_plot(meal_df, glucose_df, activity_df):
    """Analyze activity impact on glucose response"""
    meal_activity = []
    
    for _, meal in meal_df.iterrows():
        # Get 2-hour windows
        end_time = meal['meal_time'] + pd.Timedelta(hours=2)
        
        # Get glucose data
        glucose_window = glucose_df[
            (glucose_df['DateTime'] >= meal['meal_time']) &
            (glucose_df['DateTime'] <= end_time)
        ]
        
        # Get activity data
        activity_window = activity_df[
            (activity_df['start_time'] >= meal['meal_time']) &
            (activity_df['start_time'] <= end_time)
        ]
        
        if not glucose_window.empty:
            total_steps = activity_window['steps'].sum()
            glucose_rise = glucose_window['GlucoseValue'].max() - glucose_window.iloc[0]['GlucoseValue']
            
            meal_activity.append({
                'meal_type': meal['meal_type'],
                'carbs': meal['carbohydrates'],
                'total_steps': total_steps,
                'glucose_rise': glucose_rise
            })
    
    activity_df = pd.DataFrame(meal_activity)
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add scatter plot with color based on carbs
    fig.add_trace(go.Scatter(
        x=activity_df['total_steps'],
        y=activity_df['glucose_rise'],
        mode='markers',
        marker=dict(
            size=10,
            color=activity_df['carbs'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Carbs (g)")
        ),
        text=[f"Meal Type: {mt}<br>Steps: {steps:,.0f}<br>Carbs: {carbs:.1f}g<br>Rise: {rise:.1f} mg/dL"
              for mt, steps, carbs, rise in zip(
                  activity_df['meal_type'],
                  activity_df['total_steps'],
                  activity_df['carbs'],
                  activity_df['glucose_rise'])],
        hoverinfo="text"
    ))
    
    fig.update_layout(
        title="Impact of Post-Meal Activity on Glucose Rise",
        xaxis_title="Total Steps in 2h Window",
        yaxis_title="Glucose Rise (mg/dL)"
    )
    
    return fig, activity_df

# Add these functions after your existing functions but before run_streamlit_app()

def analyze_main_meal_responses(glucose_df, meal_df, activity_df):
    """Analyze glucose responses for main meals only"""
    windows = list(range(15, 121, 15))
    meal_results = []
    
    # Filter for main meals only
    main_meals = meal_df[meal_df['meal_type'].isin(['Breakfast', 'Lunch', 'Dinner'])]
    
    for _, meal in main_meals.iterrows():
        meal_time = meal['meal_time']
        meal_end = meal_time + pd.Timedelta(hours=2)
        
        glucose_window = glucose_df[
            (glucose_df['DateTime'] >= meal_time) &
            (glucose_df['DateTime'] <= meal_end)
        ]
        
        activity_window = activity_df[
            (activity_df['start_time'] >= meal_time) &
            (activity_df['start_time'] <= meal_end)
        ]
        
        if not glucose_window.empty:
            baseline = glucose_window.iloc[0]['GlucoseValue']
            peak = glucose_window['GlucoseValue'].max()
            peak_time = (glucose_window.loc[glucose_window['GlucoseValue'].idxmax(), 'DateTime'] - 
                        meal_time).total_seconds() / 60
            
            # Calculate area under curve (AUC) for glucose response
            glucose_values = glucose_window['GlucoseValue'].values
            time_points = np.arange(len(glucose_values))
            auc = np.trapz(glucose_values - baseline, time_points)
            
            # Calculate time spent above 140 mg/dL
            time_above_140 = (glucose_window['GlucoseValue'] > 140).mean() * 100
            
            # Activity metrics
            total_steps = activity_window['steps'].sum() if not activity_window.empty else 0
            total_flights = activity_window['flights'].sum() if not activity_window.empty else 0
            
            # Calculate early activity (first 30 minutes) vs later activity
            early_activity = activity_window[
                activity_window['start_time'] <= meal_time + pd.Timedelta(minutes=30)
            ]['steps'].sum()
            
            later_activity = total_steps - early_activity
            
            meal_results.append({
                'meal_time': meal_time,
                'meal_type': meal['meal_type'],
                'carbs': meal['carbohydrates'],
                'protein': meal['protein'],
                'fat': meal['fat'],
                'baseline_glucose': baseline,
                'peak_glucose': peak,
                'glucose_rise': peak - baseline,
                'time_to_peak': peak_time,
                'auc': auc,
                'time_above_140': time_above_140,
                'total_steps': total_steps,
                'total_flights': total_flights,
                'early_activity': early_activity,
                'later_activity': later_activity,
                'carb_protein_ratio': meal['carbohydrates'] / meal['protein'] if meal['protein'] > 0 else np.nan
            })
    
    return pd.DataFrame(meal_results)

def identify_optimal_responses(meal_responses):
    """Identify optimal meal responses with multiple criteria"""
    
    # Calculate composite score based on multiple metrics
    meal_responses['composite_score'] = (
        # Normalize and weight each metric
        (meal_responses['glucose_rise'] / meal_responses['carbs'] * 0.3) +
        (meal_responses['time_above_140'] * 0.3) +
        (meal_responses['auc'] / meal_responses['carbs'] * 0.4)
    )
    
    # Get top 10 best responses (lowest composite scores)
    best_responses = meal_responses.nsmallest(10, 'composite_score')
    
    return best_responses[['meal_type', 'carbs', 'protein', 'fat', 
                          'glucose_rise', 'time_above_140', 'auc',
                          'total_steps', 'early_activity', 'later_activity',
                          'composite_score']]

def create_meal_deep_analysis_plots(meal_responses):
    """Create detailed visualizations using plotly"""
    # 1. AUC vs Carb-to-Protein Ratio
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=meal_responses['carb_protein_ratio'],
        y=meal_responses['auc'],
        mode='markers',
        marker=dict(
            size=10,
            color=meal_responses['total_steps'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Total Steps")
        ),
        text=[f"Meal Type: {mt}<br>Carb/Protein: {cr:.1f}<br>Steps: {st:,.0f}<br>AUC: {auc:.0f}"
              for mt, cr, st, auc in zip(
                  meal_responses['meal_type'],
                  meal_responses['carb_protein_ratio'],
                  meal_responses['total_steps'],
                  meal_responses['auc'])],
        hoverinfo="text"
    ))
    fig1.update_layout(
        title="Glucose Response vs Carb-to-Protein Ratio",
        xaxis_title="Carb-to-Protein Ratio",
        yaxis_title="Glucose AUC",
        height=500,
        template="plotly_white"
    )

    # 2. Early Activity Impact
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=meal_responses['early_activity'],
        y=meal_responses['glucose_rise'],
        mode='markers',
        marker=dict(
            size=10,
            color=meal_responses['carbs'],
            colorscale='YlOrRd',
            showscale=True,
            colorbar=dict(title="Carbs (g)")
        ),
        text=[f"Meal Type: {mt}<br>Early Steps: {ea:,.0f}<br>Carbs: {c:.1f}g<br>Rise: {gr:.1f}"
              for mt, ea, c, gr in zip(
                  meal_responses['meal_type'],
                  meal_responses['early_activity'],
                  meal_responses['carbs'],
                  meal_responses['glucose_rise'])],
        hoverinfo="text"
    ))
    fig2.update_layout(
        title="Impact of Early Activity on Glucose Rise",
        xaxis_title="Early Activity (steps in first 30min)",
        yaxis_title="Glucose Rise (mg/dL)",
        height=500,
        template="plotly_white"
    )

    return fig1, fig2

def create_meal_deep_analysis_plots(meal_responses):
    """Create detailed visualizations using plotly"""
    # 1. AUC vs Carb-to-Protein Ratio
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=meal_responses['carb_protein_ratio'],
        y=meal_responses['auc'],
        mode='markers',
        marker=dict(
            size=10,
            color=meal_responses['total_steps'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Total Steps")
        ),
        text=[f"Meal Type: {mt}<br>Carb/Protein: {cr:.1f}<br>Steps: {st:,.0f}<br>AUC: {auc:.0f}"
              for mt, cr, st, auc in zip(
                  meal_responses['meal_type'],
                  meal_responses['carb_protein_ratio'],
                  meal_responses['total_steps'],
                  meal_responses['auc'])],
        hoverinfo="text"
    ))
    fig1.update_layout(
        title="Glucose Response vs Carb-to-Protein Ratio",
        xaxis_title="Carb-to-Protein Ratio",
        yaxis_title="Glucose AUC",
        height=500
    )

    # 2. Early Activity Impact
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=meal_responses['early_activity'],
        y=meal_responses['glucose_rise'],
        mode='markers',
        marker=dict(
            size=10,
            color=meal_responses['carbs'],
            colorscale='YlOrRd',
            showscale=True,
            colorbar=dict(title="Carbs (g)")
        ),
        text=[f"Meal Type: {mt}<br>Early Steps: {ea:,.0f}<br>Carbs: {c:.1f}g<br>Rise: {gr:.1f}"
              for mt, ea, c, gr in zip(
                  meal_responses['meal_type'],
                  meal_responses['early_activity'],
                  meal_responses['carbs'],
                  meal_responses['glucose_rise'])],
        hoverinfo="text"
    ))
    fig2.update_layout(
        title="Impact of Early Activity on Glucose Rise",
        xaxis_title="Early Activity (steps in first 30min)",
        yaxis_title="Glucose Rise (mg/dL)",
        height=500
    )

    return fig1, fig2

def analyze_carb_categories(meal_df, glucose_df):
    """Analyze glucose response patterns by carb categories"""
    
    # Create carb categories using percentiles
    meal_df['carb_category'] = pd.qcut(
        meal_df['carbohydrates'], 
        q=3, 
        labels=['Low Carb', 'Medium Carb', 'High Carb']
    )
    
    # Get the actual thresholds for reference
    carb_thresh = pd.qcut(meal_df['carbohydrates'], q=3).unique()
    thresh_values = [f"({int(cat.left)}-{int(cat.right)}g)" for cat in carb_thresh]
    
    # Analyze post-meal responses
    meal_responses = []
    
    for _, meal in meal_df.iterrows():
        # Get 2-hour glucose window
        post_meal = glucose_df[
            (glucose_df['DateTime'] >= meal['meal_time']) &
            (glucose_df['DateTime'] <= meal['meal_time'] + pd.Timedelta(hours=2))
        ]
        
        if not post_meal.empty:
            baseline = post_meal.iloc[0]['GlucoseValue']
            peak = post_meal['GlucoseValue'].max()
            
            meal_responses.append({
                'meal_type': meal['meal_type'],
                'carb_category': meal['carb_category'],
                'actual_carbs': meal['carbohydrates'],
                'peak_glucose': peak,
                'baseline': baseline,
                'glucose_rise': peak - baseline
            })
    
    response_df = pd.DataFrame(meal_responses)
    
    # Create visualization
    fig = go.Figure()
    
    # Create box and scatter plots for each meal type
    colors = {'Breakfast': 'rgb(31, 119, 180)', 
             'Lunch': 'rgb(44, 160, 44)', 
             'Dinner': 'rgb(214, 39, 40)'}
    
    for meal_type in response_df['meal_type'].unique():
        meal_data = response_df[response_df['meal_type'] == meal_type]
        
        # Add box plot
        fig.add_trace(go.Box(
            x=meal_data['carb_category'],
            y=meal_data['glucose_rise'],
            name=meal_type,
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(
                color=colors[meal_type],
                size=8,
                opacity=0.6
            ),
            hovertemplate=(
                "Meal Type: %{fullData.name}<br>" +
                "Carb Category: %{x}<br>" +
                "Glucose Rise: %{y:.1f} mg/dL<br>" +
                "<extra></extra>"
            ),
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=(
                "Glucose Rise by Carb Category and Meal Type<br>" +
                f"<span style='font-size:12px'>Categories: Low {thresh_values[0]}, " +
                f"Medium {thresh_values[1]}, High {thresh_values[2]}</span>"
            )
        ),
        xaxis_title="Carb Category",
        yaxis_title="Glucose Rise (mg/dL)",
        boxmode='group',
        height=600,
        template="plotly_white",
        showlegend=True
    )
    
    # Add reference lines
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(255,0,0,0.3)", 
                  annotation_text="30 mg/dL rise")
    fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,0,0,0.5)", 
                  annotation_text="50 mg/dL rise")
    
    # Calculate summary statistics
    summary_stats = pd.DataFrame({
        'Avg Rise': response_df.groupby(['meal_type', 'carb_category'])['glucose_rise'].mean(),
        'Median Rise': response_df.groupby(['meal_type', 'carb_category'])['glucose_rise'].median(),
        'Std Dev': response_df.groupby(['meal_type', 'carb_category'])['glucose_rise'].std(),
        'Count': response_df.groupby(['meal_type', 'carb_category'])['glucose_rise'].count(),
        'Avg Carbs': response_df.groupby(['meal_type', 'carb_category'])['actual_carbs'].mean(),
        'Min Carbs': response_df.groupby(['meal_type', 'carb_category'])['actual_carbs'].min(),
        'Max Carbs': response_df.groupby(['meal_type', 'carb_category'])['actual_carbs'].max(),
        '% Over 30': response_df.groupby(['meal_type', 'carb_category']).apply(
            lambda x: (x['glucose_rise'] > 30).mean() * 100
        ),
        '% Over 50': response_df.groupby(['meal_type', 'carb_category']).apply(
            lambda x: (x['glucose_rise'] > 50).mean() * 100
        )
    }).round(1)
    
    return fig, summary_stats, response_df

def analyze_carb_categories_with_activity(meal_df, glucose_df, activity_df):
    """Analyze glucose response patterns by carb categories and activity level"""
    
    # Create carb categories using percentiles
    meal_df['carb_category'] = pd.qcut(
        meal_df['carbohydrates'], 
        q=3, 
        labels=['Low Carb', 'Medium Carb', 'High Carb']
    )
    
    # Get the actual thresholds for reference
    carb_thresh = pd.qcut(meal_df['carbohydrates'], q=3).unique()
    thresh_values = [f"({int(cat.left)}-{int(cat.right)}g)" for cat in carb_thresh]
    
    # Analyze post-meal responses with activity
    meal_responses = []
    
    for _, meal in meal_df.iterrows():
        # Get 2-hour windows
        meal_time = meal['meal_time']
        meal_end = meal_time + pd.Timedelta(hours=2)
        
        # Get glucose data
        glucose_window = glucose_df[
            (glucose_df['DateTime'] >= meal_time) &
            (glucose_df['DateTime'] <= meal_end)
        ]
        
        # Get activity data
        activity_window = activity_df[
            (activity_df['start_time'] >= meal_time) &
            (activity_df['start_time'] <= meal_end)
        ]
        
        if not glucose_window.empty:
            baseline = glucose_window.iloc[0]['GlucoseValue']
            peak = glucose_window['GlucoseValue'].max()
            
            # Determine if period was active (any period >= 'Active')
            was_active = any(activity_window['activity_level'].isin(['Active', 'Very Active', 'Intense']))
            activity_status = 'Active' if was_active else 'Inactive'
            
            # Calculate activity metrics
            total_steps = activity_window['steps'].sum()
            max_activity_level = (activity_window['activity_level']
                                .map({'Inactive': 0, 'Light': 1, 'Moderate': 2,
                                     'Active': 3, 'Very Active': 4, 'Intense': 5})
                                .max())
            
            meal_responses.append({
                'meal_type': meal['meal_type'],
                'carb_category': meal['carb_category'],
                'activity_status': activity_status,
                'actual_carbs': meal['carbohydrates'],
                'peak_glucose': peak,
                'baseline': baseline,
                'glucose_rise': peak - baseline,
                'total_steps': total_steps,
                'max_activity_level': max_activity_level
            })
    
    response_df = pd.DataFrame(meal_responses)
    
    # Create visualization
    fig = go.Figure()
    
    # Colors for activity status
    colors = {'Active': 'rgb(44, 160, 44)', 'Inactive': 'rgb(214, 39, 40)'}
    
    # Create box plots for each activity status and meal type
    for activity_status in ['Active', 'Inactive']:
        for meal_type in response_df['meal_type'].unique():
            mask = (response_df['activity_status'] == activity_status) & \
                  (response_df['meal_type'] == meal_type)
            
            name = f"{meal_type} ({activity_status})"
            
            fig.add_trace(go.Box(
                x=response_df[mask]['carb_category'],
                y=response_df[mask]['glucose_rise'],
                name=name,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8,
                marker=dict(
                    color=colors[activity_status],
                    opacity=0.6 if activity_status == 'Inactive' else 0.8
                ),
                hovertemplate=(
                    f"{meal_type} ({activity_status})<br>" +
                    "Carb Category: %{x}<br>" +
                    "Glucose Rise: %{y:.1f} mg/dL<br>" +
                    "<extra></extra>"
                )
            ))
    
    fig.update_layout(
        title=dict(
            text=(
                "Glucose Rise by Carb Category, Meal Type, and Activity<br>" +
                f"<span style='font-size:12px'>Categories: Low {thresh_values[0]}, " +
                f"Medium {thresh_values[1]}, High {thresh_values[2]}</span>"
            )
        ),
        xaxis_title="Carb Category",
        yaxis_title="Glucose Rise (mg/dL)",
        boxmode='group',
        height=700,
        template="plotly_white",
        showlegend=True
    )
    
    # Add reference lines
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(255,0,0,0.3)", 
                  annotation_text="30 mg/dL rise")
    fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,0,0,0.5)", 
                  annotation_text="50 mg/dL rise")
    
    # Calculate summary statistics with activity status
    summary_stats = pd.DataFrame({
        'Avg Rise': response_df.groupby(['meal_type', 'activity_status', 'carb_category'])['glucose_rise'].mean(),
        'Median Rise': response_df.groupby(['meal_type', 'activity_status', 'carb_category'])['glucose_rise'].median(),
        'Std Dev': response_df.groupby(['meal_type', 'activity_status', 'carb_category'])['glucose_rise'].std(),
        'Count': response_df.groupby(['meal_type', 'activity_status', 'carb_category'])['glucose_rise'].count(),
        'Avg Steps': response_df.groupby(['meal_type', 'activity_status', 'carb_category'])['total_steps'].mean(),
        '% Over 30': response_df.groupby(['meal_type', 'activity_status', 'carb_category']).apply(
            lambda x: (x['glucose_rise'] > 30).mean() * 100
        ),
        '% Over 50': response_df.groupby(['meal_type', 'activity_status', 'carb_category']).apply(
            lambda x: (x['glucose_rise'] > 50).mean() * 100
        )
    }).round(1)
    
    return fig, summary_stats, response_df

# Add these functions before run_streamlit_app()

def find_recent_meal(workout_start, meal_df):
    """
    Find the most recent meal within 2 hours before workout
    """
    two_hours_before = workout_start - pd.Timedelta(hours=2)
    mask = (meal_df['timestamp'] >= two_hours_before) & (meal_df['timestamp'] <= workout_start)
    recent_meals = meal_df[mask]
    
    if not recent_meals.empty:
        most_recent = recent_meals.iloc[recent_meals['timestamp'].argmax()]
        time_diff = int((workout_start - most_recent['timestamp']).total_seconds() / 60)
        return f"Last meal ({time_diff} min before): {most_recent['timestamp'].strftime('%Y-%m-%d %H:%M')} {most_recent['food_name']}"
    return None

def create_workout_plot(workouts_df, heart_rate_df, glucose_df, meal_df, selected_workout_idx=0):
    """
    Create interactive plot for selected workout
    """
    # Get selected workout data
    workout = workouts_df.iloc[selected_workout_idx]
    start_time = workout['start_time']
    end_time = workout['end_time']
    window_end = end_time + pd.Timedelta(hours=2)
    
    # Filter data for the time window
    mask_hr = (heart_rate_df['timestamp'] >= start_time) & (heart_rate_df['timestamp'] <= window_end)
    mask_glucose = (glucose_df['timestamp'] >= start_time) & (glucose_df['timestamp'] <= window_end)
    
    hr_data = heart_rate_df[mask_hr]
    glucose_data = glucose_df[mask_glucose]
    
    # Create activity metrics string
    metrics = []
    if pd.notna(workout.get('total_distance')):
        metrics.append(f"Distance: {workout['total_distance']:.2f}")
    if pd.notna(workout.get('total_energy_burned')):
        metrics.append(f"Energy: {workout['total_energy_burned']:.0f}")
    if pd.notna(workout.get('avg_mets')):
        metrics.append(f"Avg METs: {workout['avg_mets']:.1f}")
    activity_metrics = ", ".join(metrics)
    
    # Find recent meal
    meal_info = find_recent_meal(start_time, meal_df)
    
    # Find meals after workout
    mask_meals = (meal_df['timestamp'] >= end_time) & (meal_df['timestamp'] <= window_end)
    after_meals = meal_df[mask_meals]
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add workout period highlight
    fig.add_shape(
        type="rect",
        x0=start_time,
        x1=end_time,
        y0=0,
        y1=1,
        yref="paper",
        fillcolor="rgba(128, 128, 128, 0.1)",
        line=dict(width=0),
        layer="below"
    )
    
    # Add glucose line
    if not glucose_data.empty:
        fig.add_trace(
            go.Scatter(
                x=glucose_data['timestamp'],
                y=glucose_data['glucose'],
                name='Glucose',
                line=dict(color='#2E86C1', width=2),
                hovertemplate='Time: %{x|%H:%M}<br>Glucose: %{y:.1f} mg/dL<extra></extra>'
            ),
            secondary_y=False
        )
    
    # Add heart rate line
    if not hr_data.empty:
        fig.add_trace(
            go.Scatter(
                x=hr_data['timestamp'],
                y=hr_data['heart_rate'],
                name='Heart Rate',
                line=dict(color='#E74C3C', width=2, dash='3.2'),
                hovertemplate='Time: %{x|%H:%M}<br>Heart Rate: %{y:.0f} bpm<extra></extra>'
            ),
            secondary_y=True
        )
    
    # Add workout end line
    fig.add_shape(
        type="line",
        x0=end_time,
        x1=end_time,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", width=1, dash="dot")
    )
    
    fig.add_annotation(
        x=end_time,
        y=1,
        text="Workout End",
        showarrow=False,
        yref="paper",
        yshift=10
    )
    
    # Add meal lines and annotations
    for _, meal in after_meals.iterrows():
        fig.add_shape(
            type="line",
            x0=meal['timestamp'],
            x1=meal['timestamp'],
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="green", width=1, dash="dot")
        )
        
        fig.add_annotation(
            x=meal['timestamp'],
            y=0.9,
            text=f"Meal: {meal['timestamp'].strftime('%H:%M')}<br>{meal['food_name']}",
            showarrow=False,
            yref="paper",
            yshift=10,
            align='left'
        )
    
    # Update layout
    workout_time = f"{start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%H:%M')}"
    title = f"Workout: {workout['type']}<br><sup>{workout_time}</sup>"
    if activity_metrics:
        title += f"<br><sup>{activity_metrics}</sup>"
    if pd.notna(workout.get('avg_heart_rate')):
        hr_stats = f"Heart Rate (avg/min/max): {workout['avg_heart_rate']:.0f}/{workout['min_heart_rate']:.0f}/{workout['max_heart_rate']:.0f}"
        title += f"<br><sup>{hr_stats}</sup>"
    if meal_info:
        title += f"<br><sup>{meal_info}</sup>"
    
    fig.update_layout(
        title=dict(
            text=title,
            y=0.95,
            yanchor='top'
        ),
        margin=dict(t=120),
        xaxis_title="Time",
        plot_bgcolor='#F8FBFE',
        paper_bgcolor='white',
        hovermode='x unified',
        showlegend=True,
        height=600,
        xaxis=dict(
            tickformat='%H:%M',
            title_text="Time",
            gridcolor='rgba(220,220,220,0.4)',
            showgrid=True
        )
    )
    
    # Update y-axes with fixed ranges
    fig.update_yaxes(
        title_text="Glucose (mg/dL)",
        secondary_y=False,
        range=[60, 190],
        gridcolor='rgba(220,220,220,0.4)',
        showgrid=True
    )
    fig.update_yaxes(
        title_text="Heart Rate (bpm)",
        secondary_y=True,
        range=[50, 190],
        gridcolor='rgba(220,220,220,0.4)',
        showgrid=True
    )
    
    return fig

def run_streamlit_app():
    st.set_page_config(page_title="Glucose Analysis", layout="wide")
    
    # Add page navigation in sidebar
    page = st.sidebar.radio(
        "Navigate", 
        ["Meal Analysis Dashboard", "Workout Analysis Dashboard", "EDA Dashboard"]
    )

    try:
        # Load data with progress indicators
        with st.spinner('Loading data...'):
            # Load common data
            glucose_df = load_glucose_data()
            meal_df = load_meal_data()
            activity_df = load_activity_data()
            full_meal_df = load_full_meal_data()
            
            # Load workout-specific data if needed
            if page == "Workout Analysis Dashboard":
                workouts_df = pd.read_csv(
                    'data/combined_workouts.csv',
                    parse_dates=['start_time', 'end_time']
                )
                workout_glucose_df = glucose_df.copy()  # Use the same glucose data
                heart_rate_df = pd.read_csv(
                    'data/heart_rate_data.csv',
                    parse_dates=['start_date']
                )
                workout_meal_df = meal_df.copy()  # Use the same meal data
                
                # Ensure timezone consistency
                if workouts_df['start_time'].dt.tz is None:
                    workouts_df['start_time'] = workouts_df['start_time'].dt.tz_localize(KST)
                    workouts_df['end_time'] = workouts_df['end_time'].dt.tz_localize(KST)

        if page == "Meal Analysis Dashboard":
            st.title('Blood Glucose Analysis Dashboard')
            
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
                
                # Create and display the plot
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

        elif page == "Workout Analysis Dashboard":
            st.title('Workout Analysis Dashboard')
            
            try:
                # Preprocess data for workout analysis
                workouts_df, heart_rate_df, workout_glucose_df, workout_meal_df = preprocess_workout_data(
                    workouts_df,
                    heart_rate_df,
                    workout_glucose_df,
                    workout_meal_df,
                    KST
                )
                
                # Create workout selection dropdown
                workout_options = {
                    f"{row['start_time'].strftime('%Y-%m-%d %H:%M')} - "
                    f"{row['end_time'].strftime('%H:%M')}, {row['type']}": idx
                    for idx, row in workouts_df.iterrows()
                }
                
                selected_workout_label = st.selectbox(
                    'Select a workout to view:',
                    options=list(workout_options.keys())
                )
                
                selected_idx = workout_options[selected_workout_label]
                
                # Create the workout plot
                fig = create_workout_plot(
                    workouts_df, 
                    heart_rate_df, 
                    workout_glucose_df, 
                    workout_meal_df, 
                    selected_idx
                )
                
                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Add workout metrics in sidebar
                workout = workouts_df.iloc[selected_idx]
                
                st.sidebar.markdown("### Workout Information")
                metrics = []
                if pd.notna(workout.get('total_distance')):
                    metrics.append(f"Distance: {workout['total_distance']:.2f}")
                if pd.notna(workout.get('total_energy_burned')):
                    metrics.append(f"Energy: {workout['total_energy_burned']:.0f}")
                if pd.notna(workout.get('avg_mets')):
                    metrics.append(f"Avg METs: {workout['avg_mets']:.1f}")
                if metrics:
                    st.sidebar.markdown("#### Activity Metrics")
                    for metric in metrics:
                        st.sidebar.markdown(f"- {metric}")
                        
                if pd.notna(workout.get('avg_heart_rate')):
                    st.sidebar.markdown("#### Heart Rate Stats")
                    st.sidebar.markdown(
                        f"- Avg: {workout['avg_heart_rate']:.0f} bpm\n"
                        f"- Min: {workout['min_heart_rate']:.0f} bpm\n"
                        f"- Max: {workout['max_heart_rate']:.0f} bpm"
                    )
                
                # Add date range information
                st.sidebar.markdown("### Dataset Information")
                workout_date_min = workouts_df['start_time'].dt.date.min()
                workout_date_max = workouts_df['start_time'].dt.date.max()
                st.sidebar.markdown(f"Available Date Range:\n"
                                f"{workout_date_min} to {workout_date_max}\n\n"
                                f"Total Workouts: {len(workouts_df):,}")
            
            except Exception as e:
                st.error(f"Error processing workout data: {str(e)}")
                st.info("Please check the format of your workout data files.")

        # Modify the EDA section in your run_streamlit_app function
        # Replace the EDA dashboard section with this:
        else:  # EDA Dashboard
            st.title('Exploratory Data Analysis Dashboard')
            
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Glucose Patterns", 
                "Meal Response Analysis",
                "Activity Impact",
                "Summary Statistics",
                "Meal Response Deep Dive",
                "Carb Category Analysis"
            ])
            
            with tab1:
                st.subheader("Glucose Patterns and Time in Range")
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_dist = create_glucose_distribution_plot(glucose_df)
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
                    fig_time_range = create_time_in_range_plot(glucose_df)
                    st.plotly_chart(fig_time_range, use_container_width=True)
                
                st.subheader("Daily Glucose Pattern")
                fig_daily = create_daily_glucose_pattern(glucose_df)
                st.plotly_chart(fig_daily, use_container_width=True)
            
            with tab2:
                st.subheader("Meal Response Analysis")
                
                fig_response, response_df = create_meal_response_analysis(meal_df, glucose_df)
                st.plotly_chart(fig_response, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Average Glucose Rise by Meal Type")
                    st.dataframe(
                        response_df.groupby('meal_type').agg({
                            'glucose_rise': ['mean', 'std'],
                            'time_to_peak': ['mean', 'std']
                        }).round(1)
                    )
                
                with col2:
                    fig_macro = create_meal_macronutrient_plot(meal_df)
                    st.plotly_chart(fig_macro, use_container_width=True)
            
            with tab3:
                st.subheader("Activity Impact Analysis")
                
                fig_activity_impact, activity_impact_df = create_activity_impact_plot(
                    meal_df, glucose_df, activity_df
                )
                st.plotly_chart(fig_activity_impact, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_activity = create_activity_pattern_plot(activity_df)
                    st.plotly_chart(fig_activity, use_container_width=True)
                
                with col2:
                    st.markdown("### Activity Impact Statistics")
                    # Calculate correlations
                    correlations = activity_impact_df[['total_steps', 'glucose_rise', 'carbs']].corr()
                    st.dataframe(correlations.round(3))
            
            with tab4:
                st.subheader("Summary Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### Glucose Metrics")
                    metrics = {
                        "Mean Glucose": f"{glucose_df['GlucoseValue'].mean():.1f} mg/dL",
                        "Time in Range": f"{(glucose_df['GlucoseValue'] <= 140).mean()*100:.1f}%",
                        "Standard Deviation": f"{glucose_df['GlucoseValue'].std():.1f}",
                        "95th Percentile": f"{glucose_df['GlucoseValue'].quantile(0.95):.1f} mg/dL"
                    }
                    for metric, value in metrics.items():
                        st.metric(metric, value)
                
                with col2:
                    st.markdown("### Meal Metrics")
                    meal_stats = meal_df.groupby('meal_type').agg({
                        'carbohydrates': ['mean', 'std'],
                        'calories': ['mean', 'std']
                    }).round(1)
                    st.dataframe(meal_stats)
                
                with col3:
                    st.markdown("### Activity Metrics")
                    daily_activity = activity_df.groupby(
                        activity_df['start_time'].dt.date
                    ).agg({
                        'steps': 'sum',
                        'flights': 'sum'
                    }).describe().round(1)
                    st.dataframe(daily_activity)

            with tab5:
                st.subheader("Deep Analysis of Meal Responses")
                
                # Calculate meal responses using the analyze_main_meal_responses function
                meal_responses = analyze_main_meal_responses(glucose_df, meal_df, activity_df)
                
                # Show optimal responses
                st.markdown("### Optimal Meal Responses")
                optimal_responses = identify_optimal_responses(meal_responses)
                st.dataframe(optimal_responses)
                
                # Show correlation analysis
                st.markdown("### Key Metric Correlations")
                correlations = meal_responses[['carbs', 'protein', 'fat', 'glucose_rise', 
                                            'auc', 'total_steps', 'early_activity',
                                            'carb_protein_ratio']].corr()
                
                # Create a heatmap for correlations
                fig_corr = go.Figure(data=go.Heatmap(
                    z=correlations.values,
                    x=correlations.columns,
                    y=correlations.columns,
                    text=correlations.values.round(3),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False,
                    colorscale='RdBu'
                ))
                fig_corr.update_layout(
                    title="Correlation Matrix of Key Metrics",
                    height=600
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Show detailed analysis plots
                col1, col2 = st.columns(2)
                fig1, fig2 = create_meal_deep_analysis_plots(meal_responses)
                
                with col1:
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Show summary statistics
                st.markdown("### Summary Statistics by Meal Type")
                summary_stats = meal_responses.groupby('meal_type').agg({
                    'glucose_rise': ['mean', 'std'],
                    'auc': ['mean', 'std'],
                    'time_above_140': 'mean',
                    'carb_protein_ratio': 'mean',
                    'total_steps': 'mean'
                }).round(2)
                
                # Format the summary stats for better display
                st.dataframe(summary_stats.style.format("{:.2f}"))
                
                # Add insights
                st.markdown("### Key Insights")
                st.markdown("""
                1. **Meal Composition Impact:**
                - Higher carb-to-protein ratios generally lead to larger glucose responses
                - Protein content shows a moderating effect on glucose rise
                
                2. **Activity Effects:**
                - Early activity (within 30 minutes) shows stronger correlation with reduced glucose rise
                - Total steps correlate with better glucose responses
                
                3. **Optimal Responses:**
                - Best responses typically combine moderate carbs with adequate protein
                - Activity plays a crucial role in managing post-meal glucose
                """)

                # Update tab6 content
                with tab6:
                    st.subheader("Carbohydrate and Activity Analysis")
                    
                    # Create the analysis
                    carb_fig, summary_stats, response_df = analyze_carb_categories_with_activity(
                        meal_df, glucose_df, activity_df
                    )
                    
                    # Show the visualization
                    st.plotly_chart(carb_fig, use_container_width=True)
                    
                    # Show activity distribution
                    st.markdown("### Activity Distribution")
                    activity_dist = pd.DataFrame({
                        'Total Meals': response_df.groupby(['meal_type', 'activity_status'])['glucose_rise'].count()
                    }).unstack()
                    st.dataframe(activity_dist)
                    
                    # Show detailed statistics
                    st.markdown("### Detailed Statistics by Meal Type, Activity, and Carb Category")
                    st.dataframe(summary_stats)
                    
                    # Calculate and show activity impact
                    st.markdown("### Activity Impact Analysis")
                    impact_analysis = response_df.groupby('carb_category').apply(
                        lambda x: pd.Series({
                            'Active vs Inactive Difference': 
                                x[x['activity_status']=='Active']['glucose_rise'].mean() - 
                                x[x['activity_status']=='Inactive']['glucose_rise'].mean(),
                            'Active Count': (x['activity_status']=='Active').sum(),
                            'Inactive Count': (x['activity_status']=='Inactive').sum()
                        })
                    ).round(1)
                    st.dataframe(impact_analysis)

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please check if all required data files are present in the data directory.")

if __name__ == '__main__':
    run_streamlit_app()