import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time
import io
import warnings
warnings.filterwarnings('ignore')

# ---------- PAGE CONFIG AND THEME ----------
st.set_page_config(
    page_title="COVID-19 Analytics Hub",
    layout="wide",
    page_icon="🦠",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS FOR ENHANCED STYLING ----------
# This CSS works with both Streamlit's default light and dark modes
st.markdown("""
<style>
    /* Typography styles */
    h1 {font-size: 2.5rem !important; font-weight: 800 !important; letter-spacing: -0.02em; margin-bottom: 0.2em !important;}
    h2 {font-weight: 700 !important; letter-spacing: -0.01em; border: none; padding-bottom: 0.3em;}
    h3 {font-weight: 600 !important;}
    .subtitle {font-size: 1.2rem; margin-top: -1em; margin-bottom: 1em; opacity: 0.8;}
    
    /* Cards with shadow and rounded corners */
    .card {
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric cards with hover effect */
    .metrics-container {
        display: flex; 
        flex-wrap: wrap; 
        gap: 16px; 
        margin: 1rem 0 2rem 0;
    }
    
    .metric-card {
        flex: 1; 
        min-width: 180px;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card.cases { border-top: 5px solid #3b82f6; }
    .metric-card.deaths { border-top: 5px solid #ef4444; }
    .metric-card.recovered { border-top: 5px solid #10b981; }
    .metric-card.active { border-top: 5px solid #f59e0b; }
    
    .metric-value {
        font-size: 2.2rem; 
        font-weight: 700; 
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 1rem; 
        margin-top: 0.5rem; 
        font-weight: 500;
        opacity: 0.8;
    }
    
    .metric-change {
        display: inline-block; 
        padding: 3px 8px; 
        border-radius: 12px; 
        font-size: 0.8rem; 
        margin-top: 0.5rem;
    }
    
    /* Status indicators */
    .kpi-positive {background: rgba(16, 185, 129, 0.15); color: #10b981;}
    .kpi-negative {background: rgba(239, 68, 68, 0.15); color: #ef4444;}
    
    /* Enhanced tab styling */
    .stTabs {
        padding: 0px 4px 0px 4px !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        padding: 8px 12px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
        border: none !important;
        transition: all 0.3s ease;
    }
    
    /* Data badge */
    .data-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 20px;
        background-color: rgba(59, 130, 246, 0.1);
        color: #3b82f6;
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 18px;
        height: 18px;
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top-color: currentColor;
        animation: spin 1s ease-in-out infinite;
        margin-right: 10px;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Table styling that works with both light and dark modes */
    .dataframe {
        width: 100%;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    
    /* Improved buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px 0;
        margin-top: 40px;
        font-size: 0.9rem;
        opacity: 0.8;
        border-top: 1px solid rgba(127, 127, 127, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ---------- DATA LOADING WITH ERROR HANDLING ----------
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(dataset_type="weekly"):
    """
    Optimized data loading function with robust error handling
    """
    try:
        start_time = time.time()
        if dataset_type == "daily":
            file_path = "WHO-COVID-19-global-daily-data.csv"
            df = pd.read_csv(file_path, parse_dates=['Date_reported'])
        else:  # weekly data
            file_path = "WHO-COVID-19-global-data.csv"
            df = pd.read_csv(file_path, parse_dates=['Date_reported'])
            
        # Basic data cleaning
        df['Year'] = df['Date_reported'].dt.year
        df['Month'] = df['Date_reported'].dt.strftime('%b %Y')
        df['Week'] = df['Date_reported'].dt.isocalendar().week
        
        # Handle NaN values in WHO_region to fix tree map errors
        df['WHO_region'] = df['WHO_region'].fillna('OTHER')
        
        # Make sure Country has no NaN values
        df['Country'] = df['Country'].fillna('Unknown')
        
        # Calculate metrics based on data type
        df = df.sort_values(['Country', 'Date_reported'])
        
        if dataset_type == "daily":
            # Daily metrics
            df['New_daily_cases'] = df.groupby('Country')['Cumulative_cases'].diff().fillna(0)
            df['New_daily_deaths'] = df.groupby('Country')['Cumulative_deaths'].diff().fillna(0)
            
            # Calculate weekly metrics from daily data
            df['week_id'] = df['Date_reported'].dt.strftime('%Y-%U')
            weekly_aggs = df.groupby(['Country', 'WHO_region', 'week_id']).agg({
                'Date_reported': 'last',
                'Cumulative_cases': 'last',
                'Cumulative_deaths': 'last',
                'New_daily_cases': 'sum',
                'New_daily_deaths': 'sum'
            }).reset_index()
            
            weekly_aggs = weekly_aggs.rename(columns={
                'New_daily_cases': 'New_weekly_cases',
                'New_daily_deaths': 'New_weekly_deaths'
            })
            
            # Merge weekly metrics back into daily data
            df = pd.merge(
                df, 
                weekly_aggs[['Country', 'Date_reported', 'New_weekly_cases', 'New_weekly_deaths']], 
                how='left', 
                on=['Country', 'Date_reported']
            )
        else:
            # Weekly metrics
            df['New_weekly_cases'] = df.groupby('Country')['Cumulative_cases'].diff().fillna(0)
            df['New_weekly_deaths'] = df.groupby('Country')['Cumulative_deaths'].diff().fillna(0)
            
            # Add placeholder columns for UI consistency
            df['New_daily_cases'] = np.nan
            df['New_daily_deaths'] = np.nan
        
        # Fix negative values
        for col in ['New_daily_cases', 'New_daily_deaths', 'New_weekly_cases', 'New_weekly_deaths']:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
        
        # Calculate mortality rate
        df['Mortality_rate'] = (df['Cumulative_deaths'] / df['Cumulative_cases'] * 100).round(2)
        df['Mortality_rate'] = df['Mortality_rate'].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Calculate load time for performance monitoring
        load_time = time.time() - start_time
        return df, load_time
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), 0

# ---------- SIDEBAR ----------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2913/2913465.png", width=80)
    st.title("COVID-19 Analytics")
    
    st.markdown("### Dataset Selection")
    dataset_type = st.radio(
        "Select Dataset Type",
        ["Daily", "Weekly"],
        horizontal=True,
        help="Daily data provides more granular analysis, weekly data offers summarized trends."
    )
    
    # Display a loading spinner while data loads
    with st.spinner('Loading and processing data...'):
        covid_data, load_time = load_data(dataset_type.lower())
    
    # Show performance indicator
    st.caption(f"Data loaded in {load_time:.2f} seconds")
    
    # Continue only if data is loaded successfully
    if not covid_data.empty:
        # Calculate date ranges from data
        min_date = covid_data['Date_reported'].min().date()
        max_date = covid_data['Date_reported'].max().date()
        
        # Date filter
        st.markdown("### Date Range")
        date_range = st.date_input(
            "Select period:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Handle single date selection case
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range
        
        # Convert back to Timestamp for filtering
        start_date_ts = pd.Timestamp(start_date)
        end_date_ts = pd.Timestamp(end_date)
        
        # Region filter
        st.markdown("### Geographic Filters")
        all_regions = sorted(covid_data['WHO_region'].unique())
        region_filter = st.multiselect(
            "WHO Region",
            options=all_regions,
            default=all_regions,
            help="Filter by WHO region(s)"
        )
        
        # Country filter
        if region_filter:
            country_options = sorted(covid_data[covid_data['WHO_region'].isin(region_filter)]['Country'].unique())
        else:
            country_options = sorted(covid_data['Country'].unique())
            
        country_filter = st.multiselect(
            "Country",
            options=country_options,
            default=[],
            help="Filter by specific countries (optional)"
        )
        
        # View options based on dataset type
        st.markdown("### Data View")
        if dataset_type == "Daily":
            view_options = st.radio(
                "Metric Type",
                options=["Cumulative", "Daily New", "Weekly New"],
                horizontal=True
            )
        else:
            view_options = st.radio(
                "Metric Type",
                options=["Cumulative", "Weekly New"],
                horizontal=True
            )
        
        # Visual options
        st.markdown("### Visualization Options")
        log_scale = st.checkbox("Use Log Scale", value=True, help="Better for comparing values of different magnitudes")
        show_trends = st.checkbox("Show Trend Lines", value=True, help="Display moving averages")
        
        # Advanced options in an expander
        with st.expander("Advanced Options", expanded=False):
            animation_speed = st.slider("Animation Speed", 100, 1000, 300, step=100, 
                                       help="Speed of map animations in milliseconds")
            map_style = st.selectbox(
                "Map Style", 
                ["auto", "light", "dark", "satellite"], 
                index=0,
                help="Visual theme for map displays"
            )
            color_theme = st.selectbox(
                "Color Theme", 
                ["Viridis", "Plasma", "Inferno", "Turbo"], 
                index=1,
                help="Color scale for visualizations"
            )
        
        # Apply filters
        with st.spinner('Applying filters...'):
            filtered = covid_data[
                (covid_data['Date_reported'] >= start_date_ts) &
                (covid_data['Date_reported'] <= end_date_ts)
            ]
            
            if region_filter:
                filtered = filtered[filtered['WHO_region'].isin(region_filter)]
                
            if country_filter:
                filtered = filtered[filtered['Country'].isin(country_filter)]
    else:
        st.error("Failed to load data. Please check the dataset files and try again.")
        st.stop()

# ---------- MAIN CONTENT ----------
# Header section with dynamic title based on selected dataset
st.markdown(f"""
# COVID-19 Global Dashboard
<div class='subtitle'>Analyzing {dataset_type} data from {start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')}</div>
""", unsafe_allow_html=True)

# Progress indicator while page elements load
progress_bar = st.progress(0)
    
# Latest date for metrics
latest_date = filtered['Date_reported'].max()

# Check if we have data after filtering
if filtered.empty:
    st.warning("No data available with the current filter settings. Please adjust your filters.")
    st.stop()

# Update progress
progress_bar.progress(20)

# ---------- KEY PERFORMANCE INDICATORS ----------
# Extract latest metrics
latest = filtered[filtered['Date_reported'] == latest_date]
global_cases = int(latest['Cumulative_cases'].sum())
global_deaths = int(latest['Cumulative_deaths'].sum())
affected_countries = filtered['Country'].nunique()

# Calculate changes from previous period
previous_date = filtered[filtered['Date_reported'] < latest_date]['Date_reported'].max()
if pd.notna(previous_date):
    previous = filtered[filtered['Date_reported'] == previous_date]
    case_change = global_cases - int(previous['Cumulative_cases'].sum())
    death_change = global_deaths - int(previous['Cumulative_deaths'].sum())
    case_percent = (case_change / int(previous['Cumulative_cases'].sum()) * 100) if int(previous['Cumulative_cases'].sum()) > 0 else 0
    death_percent = (death_change / int(previous['Cumulative_deaths'].sum()) * 100) if int(previous['Cumulative_deaths'].sum()) > 0 else 0
else:
    case_change = death_change = case_percent = death_percent = 0

# Calculate new cases based on view option
if view_options == "Daily New" and dataset_type == "Daily":
    new_metric = 'New_daily_cases'
    new_deaths = 'New_daily_deaths'
    period = "Daily"
else:
    new_metric = 'New_weekly_cases'
    new_deaths = 'New_weekly_deaths'
    period = "Weekly"
    
new_cases = int(latest[new_metric].sum())
new_deaths = int(latest[new_deaths].sum())

# Calculate mortality rate
avg_mortality = (global_deaths / global_cases * 100) if global_cases > 0 else 0

# Update progress
progress_bar.progress(30)

# Display KPI cards with enhanced styling
st.markdown('<div class="metrics-container">', unsafe_allow_html=True)

# Countries card
st.markdown(f'''
<div class="metric-card">
    <div class="metric-value">{affected_countries:,}</div>
    <div class="metric-label">Affected Countries</div>
</div>
''', unsafe_allow_html=True)

# Total cases card
case_class = "kpi-positive" if case_change >= 0 else "kpi-negative"
case_symbol = "+" if case_change >= 0 else ""
st.markdown(f'''
<div class="metric-card cases">
    <div class="metric-value">{global_cases:,}</div>
    <div class="metric-label">Total Cases</div>
    <span class="metric-change {case_class}">{case_symbol}{case_change:,} ({case_percent:.1f}%)</span>
</div>
''', unsafe_allow_html=True)

# Total deaths card
death_class = "kpi-negative" if death_change >= 0 else "kpi-positive"
death_symbol = "+" if death_change >= 0 else ""
st.markdown(f'''
<div class="metric-card deaths">
    <div class="metric-value">{global_deaths:,}</div>
    <div class="metric-label">Total Deaths</div>
    <span class="metric-change {death_class}">{death_symbol}{death_change:,} ({death_percent:.1f}%)</span>
</div>
''', unsafe_allow_html=True)

# New cases card
st.markdown(f'''
<div class="metric-card active">
    <div class="metric-value">{new_cases:,}</div>
    <div class="metric-label">New {period} Cases</div>
</div>
''', unsafe_allow_html=True)

# Mortality card
st.markdown(f'''
<div class="metric-card recovered">
    <div class="metric-value">{avg_mortality:.2f}%</div>
    <div class="metric-label">Mortality Rate</div>
</div>
''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Display last updated badge
st.markdown(f'''
<div class="data-badge">
    <div class="loading"></div> Last updated: {latest_date.strftime("%B %d, %Y")}
</div>
''', unsafe_allow_html=True)

# Update progress
progress_bar.progress(40)

# ---------- ENHANCED TAB NAVIGATION ----------
tab_icons = {
    "map": "🗺️",
    "countries": "📊",
    "trends": "📈",
    "regional": "🌐",
    "explorer": "🔍",
    "data": "📋"
}

# Create modern tab navigation
tabs = st.tabs([
    f"{tab_icons['map']} Animated Map",
    f"{tab_icons['countries']} Top Countries",
    f"{tab_icons['trends']} Trends",
    f"{tab_icons['regional']} Regional Analysis",
    f"{tab_icons['explorer']} Explorer",
    f"{tab_icons['data']} Data Table"
])

# ---------- ANIMATED MAP TAB ----------
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Global COVID-19 Spread")
    
    # Create map data
    map_data = filtered.copy()
    map_data['date'] = map_data['Date_reported'].dt.strftime('%m/%d/%Y')
    
    # Determine metrics based on view option
    if view_options == "Cumulative":
        map_metric = "Cumulative_cases"
        map_deaths = "Cumulative_deaths"
        title_metric = "Cumulative Cases"
    elif view_options == "Daily New" and dataset_type == "Daily":
        map_metric = "New_daily_cases"
        map_deaths = "New_daily_deaths"
        title_metric = "New Daily Cases"
    else:  # Weekly New
        map_metric = "New_weekly_cases"
        map_deaths = "New_weekly_deaths"
        title_metric = "New Weekly Cases"
    
    # Make bubble sizes more visually appealing
    map_data['size'] = map_data[map_metric].clip(lower=1).pow(0.3)
    
    # Get ordered dates for animation
    ordered_dates = sorted(map_data['date'].unique(), key=lambda x: pd.to_datetime(x, format='%m/%d/%Y'))
    
    # Sample dates to reduce frame count for smoother animations
    if len(ordered_dates) > 30:
        sample_step = len(ordered_dates) // 30
        sampled_dates = ordered_dates[::sample_step]
        if ordered_dates[-1] not in sampled_dates:
            sampled_dates.append(ordered_dates[-1])
    else:
        sampled_dates = ordered_dates
    
    # Create map with selected color theme
    fig_map = px.scatter_geo(
        map_data,
        locations="Country",
        locationmode='country names',
        color=map_metric,
        size='size',
        hover_name="Country",
        hover_data={
            map_metric: True,
            map_deaths: True,
            "Mortality_rate": True,
            "size": False
        },
        projection="natural earth",
        animation_frame="date",
        title=f'COVID-19: {title_metric} Over Time',
        color_continuous_scale=color_theme.lower(),
        range_color=[0, map_data[map_metric].quantile(0.95)]  # Use 95th percentile for better contrast
    )
    
    # Determine if we should use dark mode for map
    # Use Streamlit's config to determine if we're in dark mode
    is_dark_theme = st.get_option("theme.base") == "dark"
    
    # Set map styling based on user selection or theme
    if map_style == "dark" or (map_style == "auto" and is_dark_theme):
        bgcolor = "rgba(30,30,40,0.95)"
        landcolor = "#252c36"
        oceancolor = "#121418"
        textcolor = "white"
        template = "plotly_dark"
    elif map_style == "light" or (map_style == "auto" and not is_dark_theme):
        bgcolor = "rgba(240,242,246,0.95)"
        landcolor = "#ebedf0"
        oceancolor = "#f7f8fa"
        textcolor = "black"
        template = "plotly_white"
    else:  # satellite
        bgcolor = "rgba(30,30,40,0.95)"
        landcolor = "#3b3b3b"
        oceancolor = "#111111"
        textcolor = "white"
        template = "plotly_dark"
    
    # Enhance map appearance
    fig_map.update_layout(
        template=template,
        paper_bgcolor=bgcolor,
        geo=dict(
            showland=True,
            landcolor=landcolor,
            showocean=True,
            oceancolor=oceancolor,
            showcountries=True,
            countrycolor="#666666",
            showcoastlines=False,
            projection_type="natural earth",
            showframe=False
        ),
        height=620,
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": animation_speed, "redraw": True}, "fromcurrent": True}],
                    "label": "▶",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    "label": "■",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 10},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "y": 0,
            "bgcolor": "rgba(100,100,100,0.5)",
            "font": {"color": textcolor}
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16, "color": textcolor},
                "prefix": "Date: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": animation_speed},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [date],
                        {"frame": {"duration": animation_speed, "redraw": True}, "mode": "immediate"}
                    ],
                    "label": date,
                    "method": "animate"
                } for date in sampled_dates  # Use sampled dates for better performance
            ]
        }]
    )
    
    # Only create frames for sampled dates to improve performance
    map_data_sampled = map_data[map_data['date'].isin(sampled_dates)]
    fig_map.frames = [
        go.Frame(
            data=[go.Scattergeo(
                locations=map_data_sampled[map_data_sampled['date'] == date]['Country'],
                locationmode='country names',
                marker=dict(
                    size=map_data_sampled[map_data_sampled['date'] == date]['size'],
                    color=map_data_sampled[map_data_sampled['date'] == date][map_metric],
                    colorscale=color_theme.lower(),
                    colorbar=dict(title=map_metric),
                    cmin=0,
                    cmax=map_data[map_metric].quantile(0.95)
                )
            )],
            name=date
        )
        for date in sampled_dates
    ]
    
    # Render map with loading indicator
    with st.spinner("Rendering map..."):
        st.plotly_chart(fig_map, use_container_width=True)
    
    st.info("💡 **Pro Tip:** Use the play button to animate the map through time, or click on specific dates in the slider to jump to that point.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Update progress
progress_bar.progress(60)
    
# ---------- TOP COUNTRIES TAB ----------
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Top Countries Analysis")
    
    # Get metrics based on selected view
    if view_options == "Cumulative":
        top_metrics = ['Cumulative_cases', 'Cumulative_deaths']
        title_prefix = "Cumulative"
    elif view_options == "Daily New" and dataset_type == "Daily":
        top_metrics = ['New_daily_cases', 'New_daily_deaths']
        title_prefix = "New Daily"
    else:  # Weekly New
        top_metrics = ['New_weekly_cases', 'New_weekly_deaths']
        title_prefix = "New Weekly"
        
    # Get data for the latest date for each country
    latest_by_country = filtered.sort_values('Date_reported').groupby('Country').last().reset_index()
    
    # Find top 10 countries by cases
    top_by_cases = latest_by_country.sort_values(top_metrics[0], ascending=False).head(10)
    top_by_cases['Mortality_rate'] = (top_by_cases['Cumulative_deaths'] / top_by_cases['Cumulative_cases'] * 100).round(2)
    
    col_top1, col_top2 = st.columns(2)
    
    with col_top1:
        # Enhanced bar chart for cases
        fig_topcases = px.bar(
            top_by_cases,
            x='Country',
            y=top_metrics[0],
            color=top_metrics[0],
            color_continuous_scale='Blues',
            title=f"Top 10 Countries by {title_prefix} Cases",
            log_y=log_scale,
            text=top_metrics[0],
            height=450
        )
        
        fig_topcases.update_layout(
            xaxis_title="",
            yaxis_title=f"{title_prefix} Case Count",
            coloraxis_showscale=False,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        
        # Format the value labels
        fig_topcases.update_traces(
            texttemplate='%{y:,.0f}',
            textposition='outside',
            textfont_size=10,
            hovertemplate='<b>%{x}</b><br>Cases: %{y:,.0f}<extra></extra>'
        )
        
        st.plotly_chart(fig_topcases, use_container_width=True)
        
    with col_top2:
        # Enhanced bar chart for deaths
        fig_topdeaths = px.bar(
            top_by_cases,
            x='Country',
            y=top_metrics[1],
            color=top_metrics[1],
            color_continuous_scale='Reds',
            title=f"{title_prefix} Deaths in Top 10 Case Countries",
            log_y=log_scale,
            text=top_metrics[1],
            height=450
        )
        
        fig_topdeaths.update_layout(
            xaxis_title="",
            yaxis_title=f"{title_prefix} Death Count",
            coloraxis_showscale=False,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        
        # Format the value labels
        fig_topdeaths.update_traces(
            texttemplate='%{y:,.0f}',
            textposition='outside',
            textfont_size=10,
            hovertemplate='<b>%{x}</b><br>Deaths: %{y:,.0f}<extra></extra>'
        )
        
        st.plotly_chart(fig_topdeaths, use_container_width=True)
    
    # Create mortality rate visualization
    st.subheader("Mortality Rate Analysis")
    
    # Create scatter plot comparing cases, deaths and mortality rate
    fig_scatter = px.scatter(
        top_by_cases,
        x='Cumulative_cases',
        y='Cumulative_deaths',
        color='Mortality_rate',
        size='Mortality_rate',
        hover_name='Country',
        text='Country',
        log_x=log_scale,
        log_y=log_scale,
        title="Case-Death Relationship & Mortality Rate",
        color_continuous_scale=color_theme.lower(),
        height=500
    )
    
    fig_scatter.update_layout(
        xaxis_title="Total Cases",
        yaxis_title="Total Deaths",
        coloraxis_colorbar=dict(title="Mortality Rate (%)"),
        hovermode='closest'
    )
    
    fig_scatter.update_traces(
        textposition='top center',
        marker=dict(sizemin=5, sizeref=0.1),
        hovertemplate='<b>%{hovertext}</b><br>Cases: %{x:,.0f}<br>Deaths: %{y:,.0f}<br>Mortality: %{marker.color:.2f}%<extra></extra>'
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Mortality rate bar chart
    fig_mortality = px.bar(
        top_by_cases.sort_values('Mortality_rate', ascending=False),
        x='Country',
        y='Mortality_rate',
        color='Mortality_rate',
        color_continuous_scale='Viridis',
        title="Mortality Rate (%) in Top 10 Countries",
        text='Mortality_rate',
        height=450
    )
    
    fig_mortality.update_layout(
        xaxis_title="",
        yaxis_title="Mortality Rate (%)",
        coloraxis_showscale=False
    )
    
    fig_mortality.update_traces(
        texttemplate='%{y:.2f}%',
        textposition='outside',
        textfont_size=10,
        hovertemplate='<b>%{x}</b><br>Mortality Rate: %{y:.2f}%<extra></extra>'
    )
    
    st.plotly_chart(fig_mortality, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Update progress
progress_bar.progress(70)

# ---------- TRENDS TAB ----------
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Global Trends Over Time")
    
    # Create timeline data - optimize by grouping first
    timeline_metrics = {
        'Cumulative_cases': 'sum',
        'Cumulative_deaths': 'sum',
    }
    
    # Add appropriate metrics based on dataset type
    if dataset_type == "Daily":
        timeline_metrics.update({
            'New_daily_cases': 'sum',
            'New_daily_deaths': 'sum',
            'New_weekly_cases': 'sum',
            'New_weekly_deaths': 'sum'
        })
    else:
        timeline_metrics.update({
            'New_weekly_cases': 'sum',
            'New_weekly_deaths': 'sum'
        })
    
    # Calculate aggregates more efficiently
    with st.spinner("Calculating trends..."):
        timeline = filtered.groupby('Date_reported').agg(timeline_metrics).reset_index()
        timeline = timeline.sort_values('Date_reported')
    
    # Calculate moving averages for trend lines if enabled
    if show_trends:
        window_size = 7 if dataset_type == "Daily" else 4
        if len(timeline) >= window_size:
            if dataset_type == "Daily" and 'New_daily_cases' in timeline.columns:
                timeline['Cases_MA'] = timeline['New_daily_cases'].rolling(window=window_size, min_periods=1).mean()
                timeline['Deaths_MA'] = timeline['New_daily_deaths'].rolling(window=window_size, min_periods=1).mean()
            
            if 'New_weekly_cases' in timeline.columns:
                timeline['Weekly_Cases_MA'] = timeline['New_weekly_cases'].rolling(window=min(window_size, len(timeline)//2 or 1), min_periods=1).mean()
                timeline['Weekly_Deaths_MA'] = timeline['New_weekly_deaths'].rolling(window=min(window_size, len(timeline)//2 or 1), min_periods=1).mean()
    
    # Create interactive time series charts based on view options
    if view_options == "Cumulative":
        # Cumulative cases and deaths chart
        fig_cumulative = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add traces with attractive styling
        fig_cumulative.add_trace(
            go.Scatter(
                x=timeline['Date_reported'], 
                y=timeline['Cumulative_cases'],
                name="Total Cases",
                line=dict(color="#3b82f6", width=3, shape='spline'),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.2)'
            )
        )
        
        fig_cumulative.add_trace(
            go.Scatter(
                x=timeline['Date_reported'], 
                y=timeline['Cumulative_deaths'],
                name="Total Deaths",
                line=dict(color="#ef4444", width=3, shape='spline'),
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.2)'
            ),
            secondary_y=True
        )
        
        # Update layout with modern styling
        fig_cumulative.update_layout(
            title=f"Cumulative Cases and Deaths ({dataset_type} Data)",
            height=500,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        # Update axes
        fig_cumulative.update_yaxes(
            title_text="Cumulative Cases", 
            secondary_y=False,
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)',
            type='linear' if not log_scale else 'log'
        )
        fig_cumulative.update_yaxes(
            title_text="Cumulative Deaths", 
            secondary_y=True,
            showgrid=False,
            type='linear' if not log_scale else 'log'
        )
        fig_cumulative.update_xaxes(
            title_text="Date",
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)',
        )
        
        # Add hover template for better interaction
        fig_cumulative.update_traces(
            hovertemplate='<b>%{x|%B %d, %Y}</b><br>%{y:,.0f}<extra>%{fullData.name}</extra>'
        )
        
        st.plotly_chart(fig_cumulative, use_container_width=True)
        
    elif view_options == "Daily New" and dataset_type == "Daily":
        # Daily new cases and deaths chart
        fig_daily = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bar traces for new cases and deaths
        fig_daily.add_trace(
            go.Bar(
                x=timeline['Date_reported'], 
                y=timeline['New_daily_cases'],
                name="New Daily Cases",
                marker_color='rgba(59, 130, 246, 0.7)'
            )
        )
        
        fig_daily.add_trace(
            go.Bar(
                x=timeline['Date_reported'], 
                y=timeline['New_daily_deaths'],
                name="New Daily Deaths",
                marker_color='rgba(239, 68, 68, 0.7)'
            ),
            secondary_y=True
        )
        
        # Add trend lines if enabled
        if show_trends and 'Cases_MA' in timeline.columns:
            fig_daily.add_trace(
                go.Scatter(
                    x=timeline['Date_reported'], 
                    y=timeline['Cases_MA'],
                    name="Cases Trend (7-day MA)",
                    line=dict(color="#1e40af", width=3, dash='solid')
                )
            )
            
            fig_daily.add_trace(
                go.Scatter(
                    x=timeline['Date_reported'], 
                    y=timeline['Deaths_MA'],
                    name="Deaths Trend (7-day MA)",
                    line=dict(color="#b91c1c", width=3, dash='solid')
                ),
                secondary_y=True
            )
        
        # Update layout
        fig_daily.update_layout(
            title="Daily New Cases and Deaths",
            height=500,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        # Update axes
        fig_daily.update_yaxes(
            title_text="New Daily Cases", 
            secondary_y=False,
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)',
            type='linear' if not log_scale else 'log'
        )
        fig_daily.update_yaxes(
            title_text="New Daily Deaths", 
            secondary_y=True,
            showgrid=False,
            type='linear' if not log_scale else 'log'
        )
        fig_daily.update_xaxes(
            title_text="Date",
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)',
        )
        
        # Add hover template
        fig_daily.update_traces(
            hovertemplate='<b>%{x|%B %d, %Y}</b><br>%{y:,.0f}<extra>%{fullData.name}</extra>'
        )
        
        st.plotly_chart(fig_daily, use_container_width=True)
    
    else:  # Weekly New (available in both dataset types)
        # Weekly new cases and deaths chart
        fig_weekly = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bar traces for new cases and deaths
        fig_weekly.add_trace(
            go.Bar(
                x=timeline['Date_reported'], 
                y=timeline['New_weekly_cases'],
                name="New Weekly Cases",
                marker_color='rgba(59, 130, 246, 0.7)'
            )
        )
        
        fig_weekly.add_trace(
            go.Bar(
                x=timeline['Date_reported'], 
                y=timeline['New_weekly_deaths'],
                name="New Weekly Deaths",
                marker_color='rgba(239, 68, 68, 0.7)'
            ),
            secondary_y=True
        )
        
        # Add trend lines if enabled
        if show_trends and 'Weekly_Cases_MA' in timeline.columns:
            fig_weekly.add_trace(
                go.Scatter(
                    x=timeline['Date_reported'], 
                    y=timeline['Weekly_Cases_MA'],
                    name="Cases Trend (4-wk MA)",
                    line=dict(color="#1e40af", width=3, dash='solid')
                )
            )
            
            fig_weekly.add_trace(
                go.Scatter(
                    x=timeline['Date_reported'], 
                    y=timeline['Weekly_Deaths_MA'],
                    name="Deaths Trend (4-wk MA)",
                    line=dict(color="#b91c1c", width=3, dash='solid')
                ),
                secondary_y=True
            )
        
        # Update layout
        fig_weekly.update_layout(
            title=f"Weekly New Cases and Deaths ({dataset_type} Data)",
            height=500,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        # Update axes
        fig_weekly.update_yaxes(
            title_text="New Weekly Cases", 
            secondary_y=False,
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)',
            type='linear' if not log_scale else 'log'
        )
        fig_weekly.update_yaxes(
            title_text="New Weekly Deaths", 
            secondary_y=True,
            showgrid=False,
            type='linear' if not log_scale else 'log'
        )
        fig_weekly.update_xaxes(
            title_text="Date",
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)',
        )
        
        # Add hover template
        fig_weekly.update_traces(
            hovertemplate='<b>%{x|%B %d, %Y}</b><br>%{y:,.0f}<extra>%{fullData.name}</extra>'
        )
        
        st.plotly_chart(fig_weekly, use_container_width=True)
    
    # Create a stacked area chart for cases by WHO region
    st.subheader("Regional Breakdown Over Time")
    
    # More efficient data aggregation for regions
    with st.spinner("Calculating regional breakdown..."):
        region_metrics = {
            'Cumulative_cases': 'sum',
            'Cumulative_deaths': 'sum',
            'New_weekly_cases': 'sum',
            'New_weekly_deaths': 'sum'
        }
        
        if dataset_type == "Daily":
            region_metrics.update({
                'New_daily_cases': 'sum',
                'New_daily_deaths': 'sum'
            })
            
        region_timeline = filtered.groupby(['Date_reported', 'WHO_region']).agg(region_metrics).reset_index()
    
    # Select metric based on view options
    if view_options == "Cumulative":
        region_metric = "Cumulative_cases"
        region_title = "Cumulative Cases by WHO Region"
    elif view_options == "Daily New" and dataset_type == "Daily":
        region_metric = "New_daily_cases"
        region_title = "New Daily Cases by WHO Region"
    else:  # Weekly New
        region_metric = "New_weekly_cases"
        region_title = "New Weekly Cases by WHO Region"
    
    # Create enhanced area chart
    fig_region_area = px.area(
        region_timeline,
        x='Date_reported',
        y=region_metric,
        color='WHO_region',
        title=region_title,
        color_discrete_sequence=px.colors.qualitative.Bold,
        log_y=log_scale,
        height=500
    )
    
    fig_region_area.update_layout(
        xaxis_title="Date",
        yaxis_title=region_title.replace(" by WHO Region", ""),
        hovermode="x unified",
        legend_title="WHO Region",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0),
    )
    
    # Add hover template
    fig_region_area.update_traces(
        hovertemplate='<b>%{x|%B %d, %Y}</b><br>%{y:,.0f}<extra>%{fullData.name}</extra>'
    )
    
    st.plotly_chart(fig_region_area, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Update progress
progress_bar.progress(80)

# ---------- REGIONAL ANALYSIS TAB ----------
with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("WHO Regional Analysis")
    
    # Calculate region summary - more efficiently
    with st.spinner("Analyzing regional data..."):
        region_summary_metrics = {
            'Country': 'nunique',
            'Cumulative_cases': 'sum',
            'Cumulative_deaths': 'sum',
            'New_weekly_cases': 'sum',
            'New_weekly_deaths': 'sum'
        }
        
        if dataset_type == "Daily":
            region_summary_metrics.update({
                'New_daily_cases': 'sum',
                'New_daily_deaths': 'sum'
            })
            
        region_summary = filtered[filtered['Date_reported']==latest_date].groupby('WHO_region').agg(
            region_summary_metrics
        ).reset_index().rename(columns={'Country': 'Countries'})
        
        region_summary['Mortality_rate'] = (region_summary['Cumulative_deaths'] / region_summary['Cumulative_cases'] * 100).round(2)
        region_summary = region_summary.sort_values('Cumulative_cases', ascending=False)
    
    # Create continent mapping
    continent_mapping = {
        'AMRO': 'Americas',
        'EURO': 'Europe',
        'AFRO': 'Africa',
        'EMRO': 'Eastern Mediterranean',
        'WPRO': 'Western Pacific',
        'SEARO': 'South-East Asia',
        'OTHER': 'Other'
    }
    
    # Regional pie charts
    col_reg1, col_reg2 = st.columns(2)
    
    with col_reg1:
        # Determine metric for cases pie chart
        if view_options == "Cumulative":
            pie_metric = "Cumulative_cases"
            pie_title = "Distribution of Cases by WHO Region"
        elif view_options == "Daily New" and dataset_type == "Daily":
            pie_metric = "New_daily_cases"
            pie_title = "Distribution of New Daily Cases by WHO Region"
        else:  # Weekly New
            pie_metric = "New_weekly_cases"
            pie_title = "Distribution of New Weekly Cases by WHO Region"
        
        # Enhanced pie chart for cases
        fig_reg_cases = px.pie(
            region_summary,
            values=pie_metric,
            names='WHO_region',
            title=pie_title,
            color='WHO_region',
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.4
        )
        
        fig_reg_cases.update_layout(
            height=450,
            legend_title="WHO Region"
        )
        
        fig_reg_cases.update_traces(
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Cases: %{value:,.0f}<br>Share: %{percent}<extra></extra>'
        )
        
        st.plotly_chart(fig_reg_cases, use_container_width=True)
        
    with col_reg2:
        # Determine metric for deaths pie chart
        if view_options == "Cumulative":
            death_metric = "Cumulative_deaths"
            death_title = "Distribution of Deaths by WHO Region"
        elif view_options == "Daily New" and dataset_type == "Daily":
            death_metric = "New_daily_deaths"
            death_title = "Distribution of New Daily Deaths by WHO Region"
        else:  # Weekly New
            death_metric = "New_weekly_deaths"
            death_title = "Distribution of New Weekly Deaths by WHO Region"
        
        # Enhanced pie chart for deaths
        fig_reg_deaths = px.pie(
            region_summary,
            values=death_metric,
            names='WHO_region',
            title=death_title,
            color='WHO_region',
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.4
        )
        
        fig_reg_deaths.update_layout(
            height=450,
            legend_title="WHO Region"
        )
        
        fig_reg_deaths.update_traces(
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Deaths: %{value:,.0f}<br>Share: %{percent}<extra></extra>'
        )
        
        st.plotly_chart(fig_reg_deaths, use_container_width=True)
    
    # Create treemap data with proper error handling
    st.subheader("Hierarchical View of COVID-19 Impact")
    
    # Prepare treemap data with careful handling of nulls
    with st.spinner("Preparing hierarchical visualization..."):
        try:
            # Get the latest data
            treemap_base = filtered[filtered['Date_reported'] == latest_date].copy()
            
            # Handle nulls and fix common treemap errors
            treemap_base['WHO_region'] = treemap_base['WHO_region'].fillna('OTHER')
            treemap_base['Country'] = treemap_base['Country'].fillna('Unknown')
            
            # Add continent information
            treemap_base['Continent'] = treemap_base['WHO_region'].map(continent_mapping)
            
            # Rename columns for visualization clarity
            treemap_data = treemap_base.rename(columns={
                'Country': 'Country_Name',
                'Cumulative_cases': 'Total_Cases',
                'Cumulative_deaths': 'Total_Deaths'
            })
            
            # Choose metrics based on view options
            if view_options == "Cumulative":
                treemap_metric = "Total_Cases"
                treemap_title = "Cumulative COVID-19 Cases"
            elif view_options == "Daily New" and dataset_type == "Daily":
                treemap_data['Daily_Cases'] = treemap_data['New_daily_cases']
                treemap_metric = "Daily_Cases"
                treemap_title = "New Daily COVID-19 Cases"
            else:  # Weekly New
                treemap_data['Weekly_Cases'] = treemap_data['New_weekly_cases']
                treemap_metric = "Weekly_Cases"
                treemap_title = "New Weekly COVID-19 Cases"
            
            # Create treemap with error handling
            fig_treemap = px.treemap(
                treemap_data,
                path=['Continent', 'WHO_region', 'Country_Name'],
                values=treemap_metric,
                color='WHO_region',
                color_discrete_sequence=px.colors.qualitative.Bold,
                title=f"{treemap_title} by Geographic Hierarchy ({dataset_type} Data)",
                height=600
            )
            
            fig_treemap.update_layout(
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            fig_treemap.update_traces(
                textinfo='label+value',
                hovertemplate='<b>%{label}</b><br>Cases: %{value:,.0f}<extra></extra>'
            )
            
            st.plotly_chart(fig_treemap, use_container_width=True)
            
        except Exception as e:
            st.error(f"Unable to create treemap visualization: {str(e)}")
            st.info("This is likely due to missing or inconsistent categorical data in your dataset.")
            
            # Fallback: Show a simpler visualization that doesn't rely on hierarchical paths
            st.subheader("Alternative Regional View")
            
            # Create a horizontal bar chart instead
            region_data = filtered[filtered['Date_reported'] == latest_date].groupby('WHO_region').agg({
                'Cumulative_cases': 'sum',
                'Cumulative_deaths': 'sum'
            }).reset_index().sort_values('Cumulative_cases')
            
            fig_bar = px.bar(
                region_data,
                y='WHO_region',
                x='Cumulative_cases',
                color='WHO_region',
                orientation='h',
                title="COVID-19 Cases by WHO Region",
                color_discrete_sequence=px.colors.qualitative.Bold,
                height=500
            )
            
            fig_bar.update_layout(
                xaxis_title="Cumulative Cases",
                yaxis_title="WHO Region",
                showlegend=False
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Regional summary table with improved styling for both light and dark mode
    st.subheader("WHO Regional Summary")
    
    # Format the table
    formatted_summary = region_summary.copy()
    for col in formatted_summary.columns:
        if col in ['Cumulative_cases', 'Cumulative_deaths', 'New_weekly_cases', 'New_weekly_deaths']:
            formatted_summary[col] = formatted_summary[col].apply(lambda x: f"{int(x):,}")
        
        if dataset_type == "Daily" and col in ['New_daily_cases', 'New_daily_deaths']:
            formatted_summary[col] = formatted_summary[col].apply(lambda x: f"{int(x):,}")
            
    formatted_summary['Mortality_rate'] = formatted_summary['Mortality_rate'].apply(lambda x: f"{x:.2f}%")
    
    # Rename columns for better display
    rename_dict = {
        'Countries': 'Countries Affected',
        'Cumulative_cases': 'Total Cases',
        'Cumulative_deaths': 'Total Deaths',
        'New_weekly_cases': 'Weekly New Cases',
        'New_weekly_deaths': 'Weekly New Deaths',
        'Mortality_rate': 'Mortality Rate',
        'WHO_region': 'WHO Region'
    }
    
    if dataset_type == "Daily":
        rename_dict.update({
            'New_daily_cases': 'Daily New Cases',
            'New_daily_deaths': 'Daily New Deaths'
        })
    
    # Display the enhanced table
    st.dataframe(
        formatted_summary.rename(columns=rename_dict),
        use_container_width=True,
        height=300
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Update progress
progress_bar.progress(90)

# ---------- INTERACTIVE EXPLORER TAB ----------
with tabs[4]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Interactive Country Explorer")
    
    # Allow selecting specific countries to compare
    compare_countries = st.multiselect(
        "Select Countries to Compare",
        options=sorted(filtered['Country'].unique()),
        default=sorted(filtered['Country'].unique())[:5] if len(filtered['Country'].unique()) > 5 else sorted(filtered['Country'].unique()),
        help="Choose countries to compare (limit to 5-10 for better visualization)"
    )
    
    if not compare_countries:
        st.warning("Please select at least one country to explore.")
    else:
        # Get data for selected countries
        country_data = filtered[filtered['Country'].isin(compare_countries)].copy()
        
        # Determine metrics based on view options
        if view_options == "Cumulative":
            country_cases = 'Cumulative_cases'
            country_deaths = 'Cumulative_deaths'
            title_prefix = "Cumulative"
        elif view_options == "Daily New" and dataset_type == "Daily":
            country_cases = 'New_daily_cases'
            country_deaths = 'New_daily_deaths'
            title_prefix = "New Daily"
        else:  # Weekly New
            country_cases = 'New_weekly_cases'
            country_deaths = 'New_weekly_deaths'
            title_prefix = "New Weekly"
        
        # Line chart for selected countries with improved styling
        st.subheader(f"{title_prefix} Cases Comparison")
        
        fig_country_line = px.line(
            country_data,
            x='Date_reported',
            y=country_cases,
            color='Country',
            title=f"{title_prefix} Cases by Country",
            log_y=log_scale,
            color_discrete_sequence=px.colors.qualitative.Bold,
            height=500
        )
        
        fig_country_line.update_layout(
            xaxis_title="Date",
            yaxis_title=f"{title_prefix} Cases",
            hovermode="x unified",
            legend_title="Country",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # Add hover template
        fig_country_line.update_traces(
            hovertemplate='<b>%{x|%B %d, %Y}</b><br>%{y:,.0f}<extra>%{fullData.name}</extra>'
        )
        
        st.plotly_chart(fig_country_line, use_container_width=True)
        
        # Deaths comparison chart
        st.subheader(f"{title_prefix} Deaths Comparison")
        
        fig_country_deaths = px.line(
            country_data,
            x='Date_reported',
            y=country_deaths,
            color='Country',
            title=f"{title_prefix} Deaths by Country",
            log_y=log_scale,
            color_discrete_sequence=px.colors.qualitative.Bold,
            height=500
        )
        
        fig_country_deaths.update_layout(
            xaxis_title="Date",
            yaxis_title=f"{title_prefix} Deaths",
            hovermode="x unified",
            legend_title="Country",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # Add hover template
        fig_country_deaths.update_traces(
            hovertemplate='<b>%{x|%B %d, %Y}</b><br>%{y:,.0f}<extra>%{fullData.name}</extra>'
        )
        
        st.plotly_chart(fig_country_deaths, use_container_width=True)
        
        # Mortality rate over time
        st.subheader("Mortality Rate Over Time")
        
        # Calculate mortality rate over time
        mortality_data = country_data.copy()
        mortality_data['Mortality_rate'] = (mortality_data['Cumulative_deaths'] / mortality_data['Cumulative_cases'] * 100).round(2)
        mortality_data['Mortality_rate'] = mortality_data['Mortality_rate'].fillna(0).replace([np.inf, -np.inf], 0)
        
        fig_mortality_time = px.line(
            mortality_data,
            x='Date_reported',
            y='Mortality_rate',
            color='Country',
            title="Mortality Rate (%) Over Time",
            color_discrete_sequence=px.colors.qualitative.Bold,
            height=500
        )
        
        fig_mortality_time.update_layout(
            xaxis_title="Date",
            yaxis_title="Mortality Rate (%)",
            hovermode="x unified",
            legend_title="Country",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # Add hover template
        fig_mortality_time.update_traces(
            hovertemplate='<b>%{x|%B %d, %Y}</b><br>%{y:.2f}%<extra>%{fullData.name}</extra>'
        )
        
        st.plotly_chart(fig_mortality_time, use_container_width=True)
        
        # Create radar chart for multi-dimensional comparison
        st.subheader("Multi-dimensional Country Comparison")
        
        # Get the latest data for each selected country
        latest_country_data = country_data.groupby('Country').apply(lambda x: x[x['Date_reported'] == x['Date_reported'].max()]).reset_index(drop=True)
        
        # Normalize data for radar chart
        radar_data = latest_country_data.copy()
        normalized_cols = []
        
        # Choose columns to include based on dataset type
        if dataset_type == "Daily":
            radar_metrics = ['Cumulative_cases', 'Cumulative_deaths', 'New_daily_cases', 'New_daily_deaths', 'Mortality_rate']
            radar_labels = ['Total Cases', 'Total Deaths', 'New Daily Cases', 'New Daily Deaths', 'Mortality Rate']
        else:
            radar_metrics = ['Cumulative_cases', 'Cumulative_deaths', 'New_weekly_cases', 'New_weekly_deaths', 'Mortality_rate']
            radar_labels = ['Total Cases', 'Total Deaths', 'New Weekly Cases', 'New Weekly Deaths', 'Mortality Rate']
        
        # Perform normalization for each metric
        for col in radar_metrics:
            max_val = radar_data[col].max()
            if max_val > 0:  # Avoid division by zero
                radar_data[f'{col}_norm'] = (radar_data[col] / max_val) * 100
            else:
                radar_data[f'{col}_norm'] = 0
            normalized_cols.append(f'{col}_norm')
        
        # Create radar chart
        fig_radar = go.Figure()
        
        # Define colors for radar chart - use Streamlit theme compatible colors
        radar_colors = px.colors.qualitative.Bold
        
        for i, country in enumerate(radar_data['Country'].unique()):
            country_row = radar_data[radar_data['Country'] == country].iloc[0]
            fig_radar.add_trace(go.Scatterpolar(
                r=[country_row[f'{col}_norm'] for col in radar_metrics],
                theta=radar_labels,
                fill='toself',
                name=country,
                line_color=radar_colors[i % len(radar_colors)]
            ))
        
        # Use neutral grid colors that work in both light and dark mode
        grid_color = "rgba(128, 128, 128, 0.2)"
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor=grid_color
                ),
                angularaxis=dict(
                    gridcolor=grid_color
                )
            ),
            showlegend=True,
            height=600,
            title=f"Multi-dimensional Country Comparison ({dataset_type} Data, Normalized to 100%)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("Note: All metrics are normalized relative to the maximum value across the selected countries.")
            
    st.markdown('</div>', unsafe_allow_html=True)

# Update progress
progress_bar.progress(95)

# ---------- DATA TABLE TAB ----------
with tabs[5]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Detailed Data Table & Export")
    
    # Add options for data display
    table_options = st.radio(
        "Choose data to display:",
        options=["All Data", "Latest Date Only", "Summary by Country", "Summary by WHO Region"],
        horizontal=True
    )
    
    # Prepare data based on selection
    with st.spinner("Preparing data table..."):
        if table_options == "All Data":
            display_data = filtered.sort_values(['Date_reported', 'Country'])
        elif table_options == "Latest Date Only":
            display_data = filtered[filtered['Date_reported'] == latest_date].sort_values('Country')
        elif table_options == "Summary by Country":
            # Determine which metrics to include based on dataset type
            agg_metrics = {
                'WHO_region': 'first',
                'Cumulative_cases': 'max',
                'Cumulative_deaths': 'max',
                'New_weekly_cases': 'sum',
                'New_weekly_deaths': 'sum',
                'Mortality_rate': 'max'
            }
            
            if dataset_type == "Daily":
                agg_metrics.update({
                    'New_daily_cases': 'sum',
                    'New_daily_deaths': 'sum'
                })
                
            display_data = filtered.groupby('Country').agg(agg_metrics).reset_index().sort_values('Cumulative_cases', ascending=False)
        else:  # Summary by WHO Region
            # Determine which metrics to include based on dataset type
            region_agg_metrics = {
                'Country': 'nunique',
                'Cumulative_cases': 'sum',
                'Cumulative_deaths': 'sum',
                'New_weekly_cases': 'sum',
                'New_weekly_deaths': 'sum'
            }
            
            if dataset_type == "Daily":
                region_agg_metrics.update({
                    'New_daily_cases': 'sum',
                    'New_daily_deaths': 'sum'
                })
                
            display_data = filtered.groupby(['WHO_region', 'Date_reported']).agg(region_agg_metrics).reset_index()
            display_data['Mortality_rate'] = (display_data['Cumulative_deaths'] / display_data['Cumulative_cases'] * 100).round(2)
            display_data = display_data.rename(columns={'Country': 'Countries'}).sort_values(['WHO_region', 'Date_reported'])
    
    # Show the data table with a custom height
    table_height = 450
    st.dataframe(
        display_data,
        use_container_width=True,
        height=table_height
    )
    
    # Show row count
    st.caption(f"Showing {len(display_data):,} records")
    
    # Export options
    st.subheader("Export Data")
    st.markdown("Download the data in your preferred format:")
    
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    
    with col_ex1:
        # CSV export
        csv = display_data.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            data=csv,
            file_name=f"covid_data_{dataset_type.lower()}_{table_options.lower().replace(' ', '_')}_{latest_date.strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Export data as CSV (comma-separated values)"
        )
    
    with col_ex2:
        # Excel export
        try:
            # Use BytesIO to create an Excel file for download
            excel_buffer = io.BytesIO()
            display_data.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            st.download_button(
                "📊 Download Excel",
                data=excel_buffer,
                file_name=f"covid_data_{dataset_type.lower()}_{table_options.lower().replace(' ', '_')}_{latest_date.strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Export data as Excel spreadsheet"
            )
        except Exception as e:
            st.warning(f"Excel export not available: {str(e)}")
    
    with col_ex3:
        # JSON export
        json_data = display_data.to_json(orient='records')
        st.download_button(
            "🔄 Download JSON",
            data=json_data,
            file_name=f"covid_data_{dataset_type.lower()}_{table_options.lower().replace(' ', '_')}_{latest_date.strftime('%Y%m%d')}.json",
            mime="application/json",
            help="Export data as JSON for API integration"
        )
        
    # Data dictionary
    with st.expander("Data Dictionary", expanded=False):
        st.markdown("""
        ### Column Descriptions
        
        * **Date_reported**: Date of the report
        * **Country**: Country, territory, or area name
        * **WHO_region**: WHO regional offices: AFRO (Africa), AMRO (Americas), EMRO (Eastern Mediterranean), EURO (Europe), SEARO (South-East Asia), WPRO (Western Pacific)
        * **Cumulative_cases**: Cumulative confirmed cases reported to WHO to date
        * **Cumulative_deaths**: Cumulative deaths reported to WHO to date
        * **New_daily_cases**: New confirmed cases reported in the last 24 hours (daily data only)
        * **New_daily_deaths**: New deaths reported in the last 24 hours (daily data only)
        * **New_weekly_cases**: New confirmed cases reported in the last 7 days
        * **New_weekly_deaths**: New deaths reported in the last 7 days
        * **Mortality_rate**: Deaths as a percentage of cases (Cumulative_deaths / Cumulative_cases * 100)
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Final progress update and remove progress bar
progress_bar.progress(100)
time.sleep(0.5)  # Small pause to show 100% completion
progress_bar.empty()

# ---------- FOOTER ----------
st.markdown(f"""
<div class="footer">
    <p>COVID-19 Analytics Hub - Data Source: World Health Organization (WHO)</p>
    <p>Dashboard by AdilShamim8 | Last Updated: July 3, 2025</p>
    <p>Created with Streamlit, Plotly & Pandas | Version 5.5</p>
</div>
""", unsafe_allow_html=True)
