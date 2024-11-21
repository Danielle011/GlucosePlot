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

# Modify your run_streamlit_app function
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
            # Your existing dashboard code here
            # (keep all your existing dashboard code)
            
        else:  # EDA Dashboard
            st.title('Exploratory Data Analysis Dashboard')
            
            # Add tabs for different aspects of EDA
            tab1, tab2, tab3 = st.tabs(["Glucose Patterns", "Meal Patterns", "Activity Patterns"])
            
            with tab1:
                st.subheader("Glucose Value Distribution and Patterns")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_dist = create_glucose_distribution_plot(glucose_df)
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
                    fig_daily = create_daily_glucose_pattern(glucose_df)
                    st.plotly_chart(fig_daily, use_container_width=True)
                
                # Add key statistics
                st.markdown("### Key Glucose Statistics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean Glucose", f"{glucose_df['GlucoseValue'].mean():.1f}")
                col2.metric("Time in Range (<140)", f"{(glucose_df['GlucoseValue'] <= 140).mean()*100:.1f}%")
                col3.metric("Standard Deviation", f"{glucose_df['GlucoseValue'].std():.1f}")
                col4.metric("95th Percentile", f"{glucose_df['GlucoseValue'].quantile(0.95):.1f}")
            
            with tab2:
                st.subheader("Meal Composition Analysis")
                fig_macro = create_meal_macronutrient_plot(meal_df)
                st.plotly_chart(fig_macro, use_container_width=True)
                
                # Add meal statistics
                st.markdown("### Average Macronutrients by Meal Type")
                st.dataframe(
                    meal_df.groupby('meal_type')[['carbohydrates', 'protein', 'fat']].mean().round(1)
                )
            
            with tab3:
                st.subheader("Activity Patterns")
                fig_activity = create_activity_pattern_plot(activity_df)
                st.plotly_chart(fig_activity, use_container_width=True)
                
                # Add activity statistics
                st.markdown("### Activity Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Daily Steps", 
                             f"{int(activity_df.groupby(activity_df['start_time'].dt.date)['steps'].sum().mean()):,}")
                with col2:
                    st.metric("Average Daily Flights", 
                             f"{activity_df.groupby(activity_df['start_time'].dt.date)['flights'].sum().mean():.1f}")
                
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please check if all required data files are present in the data directory.")

if __name__ == '__main__':
    run_streamlit_app()