import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timezone, timedelta
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

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

def run_streamlit_app():
    st.set_page_config(page_title="Glucose Analysis", layout="wide")
    
    # Add page navigation in sidebar
    page = st.sidebar.radio("Navigate", ["Meal Analysis Dashboard", "EDA Dashboard"])
    
    try:
        # Load data with progress indicators
        with st.spinner('Loading data...'):
            glucose_df = load_glucose_data()
            meal_df = load_meal_data()
            activity_df = load_activity_data()
            full_meal_df = load_full_meal_data()
        
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
                
                # Create legend-style activity guide
                st.markdown(
                    """
                    <div style="
                        display: inline-flex;
                        align-items: center;
                        padding: 8px 16px;
                        background-color: white;
                        border: 1px solid #333;
                    ">
                        <div style="display: flex; gap: 20px; align-items: center;">
                    """,
                    unsafe_allow_html=True
                )

                level_map = {
                    'Inactive': 0.1,
                    'Light': 0.2,
                    'Moderate': 0.3,
                    'Active': 0.7,
                    'Very Active': 0.85,
                    'Intense': 1.0
                }

                for level, opacity in level_map.items():
                    st.markdown(
                        f'<div style="display: flex; align-items: center; gap: 5px;">'
                        f'<div style="'
                        f'width: 15px;'
                        f'height: 15px;'
                        f'background-color: rgba(255,0,0,{opacity});'
                        f'"></div>'
                        f'<span style="font-size: 0.9em;">{level}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("</div></div>", unsafe_allow_html=True)

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


        # Modify the EDA section in your run_streamlit_app function
        # Replace the EDA dashboard section with this:
        else:  # EDA Dashboard
            st.title('Exploratory Data Analysis Dashboard')
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "Glucose Patterns", 
                "Meal Response Analysis",
                "Activity Impact",
                "Summary Statistics"
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
                
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please check if all required data files are present in the data directory.")

if __name__ == '__main__':
    run_streamlit_app()