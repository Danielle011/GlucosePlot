import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timezone, timedelta
import numpy as np 

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

def create_glucose_activity_bars_numbers(glucose_window, meal_data, activity_window, end_time, selected_idx=0):
    """Creates chart with bars and flight numbers on top"""

    meal_time = meal_data.iloc[selected_idx]['meal_time']
    
    # Add relative time in minutes to glucose data
    glucose_window = glucose_window.copy()
    glucose_window['minutes_from_meal'] = (
        (glucose_window['DateTime'] - meal_time).dt.total_seconds() / 60
    ).round().astype(int)
    
 # Update meal subtitle to include time
    meal = meal_data.iloc[selected_idx]
    meal_subtitle = (
        f"{meal_time.strftime('%I:%M %p')} | "  # Added meal time
        f"{meal['food_name']} | "
        f"Calories: {meal['calories']:.0f} | "
        f"Carbs: {meal['carbohydrates']:.1f}g | "
        f"Protein: {meal['protein']:.1f}g | "
        f"Fat: {meal['fat']:.1f}g"
    )
    
    fig = go.Figure()
    
    # Add activity bars
    for _, activity in activity_window[activity_window['steps'] > 100].iterrows():
        color = get_activity_color_gradient(activity['activity_level'])
        
        # Add bar for steps
        fig.add_trace(
            go.Bar(
                x=[activity['start_time'] + (activity['end_time'] - activity['start_time'])/2],
                y=[activity['steps']],
                width=600000,  # 10 minutes in milliseconds
                marker_color=color,
                name='Activity',
                hovertemplate=(
                    f"Time: {format_time_12hr(activity['start_time'])} - {format_time_12hr(activity['end_time'])}<br>" +
                    f"Steps: {int(activity['steps']):,} steps<br>" +
                    f"Distance: {activity['distance']:.1f} km<br>" +
                    f"Flights: {int(activity['flights'])} flights<br>" +
                    f"Level: {activity['activity_level']}"
                ),
                yaxis='y2',
                showlegend=False
            )
        )
        
        # Add flight number as text if flights exist
        if activity['flights'] > 0:
            fig.add_trace(
                go.Scatter(
                    x=[activity['start_time'] + (activity['end_time'] - activity['start_time'])/2],
                    y=[activity['steps']],
                    text=f"↑{int(activity['flights'])}",
                    mode='text',
                    textposition='top center',
                    showlegend=False,
                    yaxis='y2'
                )
            )
    
    # Add glucose line
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
    
    # Update layout with secondary y-axis
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
        yaxis2=dict(
            title='Steps',
            overlaying='y',
            side='right',
            range=[0, max(3000, activity_window['steps'].max() * 1.1)],
            showgrid=False
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        showlegend=False,
        margin=dict(t=100, l=60, r=60, b=60),
        bargap=0
    )
    
    fig.update_xaxes(range=[meal_time, end_time])
    
    return fig

def create_glucose_activity_bars_markers(glucose_window, meal_data, activity_window, end_time, selected_idx=0):
    """Creates chart with bars and flight markers"""
    meal_time = meal_data.iloc[selected_idx]['meal_time']
    
    # Add relative time in minutes to glucose data
    glucose_window = glucose_window.copy()
    glucose_window['minutes_from_meal'] = (
        (glucose_window['DateTime'] - meal_time).dt.total_seconds() / 60
    ).round().astype(int)
    
    # Format meal information for subtitle
    meal = meal_data.iloc[selected_idx]
    meal_subtitle = (
        f"{meal_time.strftime('%I:%M %p')} | "  # Added meal time
        f"{meal['food_name']} | "
        f"Calories: {meal['calories']:.0f} | "
        f"Carbs: {meal['carbohydrates']:.1f}g | "
        f"Protein: {meal['protein']:.1f}g | "
        f"Fat: {meal['fat']:.1f}g"
    )
    
    fig = go.Figure()
    
    # Add activity bars
    for _, activity in activity_window[activity_window['steps'] > 100].iterrows():
        color = get_activity_color_gradient(activity['activity_level'])
        
        # Add bar for steps
        fig.add_trace(
            go.Bar(
                x=[activity['start_time'] + (activity['end_time'] - activity['start_time'])/2],
                y=[activity['steps']],
                width=600000,  # 10 minutes in milliseconds
                marker_color=color,
                name='Activity',
                hovertemplate=(
                    f"Time: {format_time_12hr(activity['start_time'])} - {format_time_12hr(activity['end_time'])}<br>" +
                    f"Steps: {int(activity['steps']):,} steps<br>" +
                    f"Distance: {activity['distance']:.1f} km<br>" +
                    f"Flights: {int(activity['flights'])} flights<br>" +
                    f"Level: {activity['activity_level']}"
                ),
                yaxis='y2',
                showlegend=False
            )
        )
        
        # Add flights as diamond markers
        if activity['flights'] > 0:
            fig.add_trace(
                go.Scatter(
                    x=[activity['start_time'] + (activity['end_time'] - activity['start_time'])/2],
                    y=[activity['steps'] * 0.8],  # Position marker at 80% of bar height
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=activity['flights'] * 8,  # Size based on number of flights
                        color='black',
                        line=dict(color='white', width=1)
                    ),
                    name='Flights',
                    hovertemplate=f"Flights: {int(activity['flights'])}<br>",
                    showlegend=False,
                    yaxis='y2'
                )
            )
    
    # Add glucose line
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
    
    # Update layout with secondary y-axis
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
            showgrid=False,  # Removed grid
            zeroline=False,
            ticktext=tick_texts,
            tickvals=tick_values,
            title_font=dict(size=12, color='black'),  # Made black
            tickfont=dict(size=10, color='black'),    # Made black
            linecolor='black',  # Made axis line black
            mirror=True        # Show axis line on both sides
        ),
        yaxis=dict(
            title='Blood Glucose (mg/dL)',
            showgrid=False,  # Removed grid
            zeroline=False,
            title_font=dict(size=12, color='black'),  # Made black
            tickfont=dict(size=10, color='black'),    # Made black
            linecolor='black',  # Made axis line black
            mirror=True,       # Show axis line on both sides
            range=[0, max(200, glucose_window['GlucoseValue'].max() * 1.1)]
        ),
        yaxis2=dict(
            title='Steps',
            overlaying='y',
            side='right',
            range=[0, max(3000, activity_window['steps'].max() * 1.1)],
            showgrid=False,
            title_font=dict(size=12, color='black'),  # Made black
            tickfont=dict(size=10, color='black'),    # Made black
            linecolor='black'  # Made axis line black
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        showlegend=False,
        margin=dict(t=100, l=60, r=60, b=60),
        bargap=0
    )
    
    # Add only 140 reference line
    fig.add_hline(y=140, line_dash="dot", line_color="rgba(0, 0, 0, 0.3)", line_width=1)
    
    fig.update_xaxes(range=[meal_time, end_time])
    
    return fig

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
    
    # Format meal information for subtitle - Updated to include time
    meal = meal_data.iloc[selected_idx]
    meal_subtitle = (
        f"{meal_time.strftime('%I:%M %p')} | "  # Added meal time
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
        
        # Create background shade with legend
        fig.add_trace(
            go.Scatter(
                x=[activity['start_time'], activity['start_time'], 
                   activity['end_time'], activity['end_time']],
                y=[0, 200, 200, 0],
                fill='toself',
                mode='none',
                name=activity['activity_level'],  # Add activity level as name
                fillcolor=color,
                hoverinfo='skip',  # Disable hover for the shade
                showlegend=True,   # Show in legend
                legendgroup=activity['activity_level'],  # Group same activity levels
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
            showgrid=False,  # Removed grid
            zeroline=False,
            ticktext=tick_texts,
            tickvals=tick_values,
            title_font=dict(size=12, color='black'),  # Made black
            tickfont=dict(size=10, color='black'),    # Made black
            linecolor='black',  # Made axis line black
            mirror=False       # Show axis line on both sides
        ),
        yaxis=dict(
            title='Blood Glucose (mg/dL)',
            showgrid=False,  # Removed grid
            zeroline=False,
            title_font=dict(size=12, color='black'),  # Made black
            tickfont=dict(size=10, color='black'),    # Made black
            linecolor='black',  # Made axis line black
            mirror=False,       # Show axis line on both sides
            range=[0, max(200, glucose_window['GlucoseValue'].max() * 1.1)]
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'  # Semi-transparent white background
        ) 
        margin=dict(t=100, l=60, r=20, b=60),
    )
    
    # Add only 140 reference line
    fig.add_hline(y=140, line_dash="dot", line_color="rgba(0, 0, 0, 0.3)", line_width=1)
    
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
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs([
                "Original (Shaded)",
                "Bar Plot with Numbers",
                "Bar Plot with Markers"
            ])
            
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
            
            with tab1:
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
                
                # Original shaded plot
                fig1 = create_glucose_meal_activity_chart_gradient(
                    glucose_window, 
                    pd.DataFrame([selected_meal]), 
                    activity_window,
                    end_time,
                    0
                )
                st.plotly_chart(fig1, use_container_width=True, key="plot1")
            
            with tab2:
                # Activity level legend for bars
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
                st.markdown("↑n : n flights climbed")
                
                # Bar plot with numbers
                fig2 = create_glucose_activity_bars_numbers(
                    glucose_window, 
                    pd.DataFrame([selected_meal]), 
                    activity_window,
                    end_time,
                    0
                )
                st.plotly_chart(fig2, use_container_width=True, key="plot2" )
            
            with tab3:
                # Activity level legend for bars with markers
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
                st.markdown("♦ : Diamond size indicates number of flights")
                
                # Bar plot with markers
                fig3 = create_glucose_activity_bars_markers(
                    glucose_window, 
                    pd.DataFrame([selected_meal]), 
                    activity_window,
                    end_time,
                    0
                )
                st.plotly_chart(fig3, use_container_width=True, key="plot3" )
            
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