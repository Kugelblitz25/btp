import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import httpx
import datetime
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d
import asyncio
import isodate

# Set page configuration
st.set_page_config(page_title="Camera Analytics System", layout="wide")

API_BASE_URL = "https://dbapi-2zb1.onrender.com"  

# Function to fetch data from API endpoints
async def fetch_data(endpoint, params=None):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}{endpoint}", params=params)
        print(f"{API_BASE_URL}{endpoint}")
        if response.status_code == 200:
            return response.json()
        else:
            print(response.json())
            st.error(f"Error fetching data: {response.status_code} - {response.text}")
            return []

# Functions for specific features
async def count_people_in_timeframe(initial_time, window):
    """Count unique people in the given timeframe"""
    # Convert string times to datetime objects
    initial_time_dt = datetime.datetime.fromisoformat(initial_time)
    window_end_dt = initial_time_dt + datetime.timedelta(seconds=window)
    
    # Get all tracks
    tracks = await fetch_data("/tracks/", {"limit": 1000})
    
    unique_people = set()
    for track in tracks:
        # Parse track times
        track_time = datetime.datetime.fromisoformat(track["time"])
        track_end_time = track_time + datetime.timedelta(seconds=track["duration"])
        
        # Check if the track falls within our window
        if ((track_end_time > initial_time_dt and track_time < window_end_dt) or 
            (track_time >= initial_time_dt and track_time <= window_end_dt)):
            unique_people.add(track["person_id"])
    
    return len(unique_people), unique_people

# async def get_heatmap_data(initial_time, window):
#     """Get x,y coordinates for heatmap in the given timeframe"""
#     # Convert string times to datetime objects
#     initial_time_dt = datetime.datetime.fromisoformat(initial_time)
#     window_end_dt = initial_time_dt + datetime.timedelta(seconds=window)
    
#     # Get all tracks
#     tracks = await fetch_data("/tracks/", {"limit": 1000})
    
#     x_coords = []
#     y_coords = []
    
#     for track in tracks:
#         # Parse track times
#         track_time = datetime.datetime.fromisoformat(track["time"])
#         track_end_time = track_time + datetime.timedelta(seconds=track["duration"])
        
#         # Check if the track falls within our window
#         if ((track_end_time > initial_time_dt and track_time < window_end_dt) or 
#             (track_time >= initial_time_dt and track_time <= window_end_dt)):
#             x_coords.append(track["x"])
#             y_coords.append(track["y"])
    
#     return x_coords, y_coords

async def get_heatmap_data(initial_time_dt, window):
    """Get x,y coordinates for heatmap in the given timeframe"""
    # Convert string times to datetime objects
    window_end_dt = initial_time_dt + datetime.timedelta(seconds=window)
    print(initial_time_dt, window_end_dt)
    
    # Get all tracks
    tracks = await fetch_data("/tracks/", {"limit": 1000})
    
    x_coords = []
    y_coords = []
    
    for track in tracks:
        # Parse track times
        track_time = datetime.datetime.fromisoformat(track["time"])
        
        # Convert duration to float or int if it's a string
        try:
            if isinstance(track["duration"], str):
                duration_seconds = isodate.parse_duration(track["duration"]).total_seconds()
            else:
                duration_seconds = track["duration"]
                
            track_end_time = track_time + datetime.timedelta(seconds=duration_seconds)
            
            # Check if the track falls within our window
            if ((track_end_time > initial_time_dt and track_time < window_end_dt) or 
                (track_time >= initial_time_dt and track_time <= window_end_dt)):
                x_coords.append(track["x"])
                y_coords.append(track["y"])
                
        except (ValueError, TypeError) as e:
            print(f"Error processing track {track['id']}: {e}")
            # Skip this track and continue with others
            continue
    
    return x_coords, y_coords

async def track_person(person_id):
    """Get tracking data for a specific person"""
    # Get tracks for the specific person
    tracks = await fetch_data("/tracks/", {"person_id": person_id, "limit": 1000})
    
    # Sort by timestamp
    tracks.sort(key=lambda x: x["time"])
    
    # Extract x, y coordinates and timestamps
    x_coords = [track["x"] for track in tracks]
    y_coords = [track["y"] for track in tracks]
    timestamps = [datetime.datetime.fromisoformat(track["time"]) for track in tracks]
    
    return x_coords, y_coords, timestamps

async def get_person_info(person_id):
    """Get detailed information about a person"""
    # Get person data
    person = await fetch_data(f"/persons/{person_id}")
    print(person)
    
    # Get additional information
    if person:
        # Get gender info
        gender = None
        if person.get("gender_id"):
            gender = await fetch_data(f"/genders/{person['gender_id']}")
        
        # Get age info if available
        age = None
        if person.get("age_id"):
            age = await fetch_data(f"/ages/{person['age_id']}")
        
        # Get race info if available
        race = None
        if person.get("race_id"):
            race = await fetch_data(f"/races/{person['race_id']}")
        
        # Get hairline info if available
        hairline = None
        if person.get("hairline_id"):
            hairline = await fetch_data(f"/hairlines/{7 - person['hairline_id']}")
        
        # Get apparel info
        apparel = await fetch_data("/apparels/", {"person_id": person_id})
        
        return {
            "person": person,
            "gender": gender,
            "age": age,
            "race": race,
            "hairline": hairline,
            "apparel": apparel
        }
    
    return None

# Create the Streamlit UI
st.title("Camera Analytics System")

# Create tabs for different features
tab1, tab2, tab3, tab4 = st.tabs([
    "🧮 People Counter", 
    "🔥 Heat Map", 
    "🔍 Person Tracker", 
    "ℹ Person Info"
])

# Tab 1: People Counter
with tab1:
    st.header("Count People in Time Frame")
    
    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        initial_time = st.text_input(
            "Initial Time (YYYY-MM-DD HH:MM:SS)", 
            value=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    with col2:
        window_seconds = st.slider("Time Window (seconds)", 1, 3600, 300)
    
    # Button to trigger counting
    if st.button("Count People", key="count_btn"):
        with st.spinner("Counting people..."):
            try:
                count, unique_ids = asyncio.run(count_people_in_timeframe(initial_time, window_seconds))
                
                # Display results
                st.success(f"Found {count} unique people in the specified time frame")
                
                # Show the IDs in an expander
                with st.expander("View Person IDs"):
                    st.write(", ".join([str(id) for id in sorted(unique_ids)]))
                    
            except Exception as e:
                st.error(f"Error counting people: {str(e)}")

# Tab 2: Heat Map
with tab2:
    st.header("Person Density Heat Map")
    
    # Input fields
    col1, col2 = st.columns(2)
    if "hm_time" not in st.session_state:
        st.session_state.hm_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with col1:
        hm_initial_time = st.text_input(
            "Initial Time (YYYY-MM-DD HH:MM:SS)", 
            key="hm_time"
        )
    with col2:
        hm_window_seconds = st.slider("Time Window (seconds)", 1, 3600, 300, key="hm_window")
    
    # Button to generate heatmap
    if st.button("Generate Heat Map", key="heatmap_btn"):
        with st.spinner("Generating heat map..."):
            try:
                initial_time_dt = datetime.datetime.fromisoformat(hm_initial_time)
                hm_window_seconds = int(hm_window_seconds)
                x_coords, y_coords = asyncio.run(get_heatmap_data(initial_time_dt, hm_window_seconds))
                
                if len(x_coords) > 0:
                    # Create a 2D histogram
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Generate heatmap
                    heatmap = ax.hist2d(x_coords, y_coords, bins=20, cmap='hot')
                    plt.colorbar(heatmap[3], ax=ax, label='Frequency')
                    
                    ax.set_title(f"Person Density Heat Map ({hm_initial_time} to +{hm_window_seconds}s)")
                    ax.set_xlabel("X Coordinate")
                    ax.set_ylabel("Y Coordinate")
                    
                    # Display the plot
                    st.pyplot(fig)
                    
                    # Show the raw data in an expander
                    with st.expander("View Raw Coordinate Data"):
                        df = pd.DataFrame({
                            'X': x_coords,
                            'Y': y_coords
                        })
                        st.dataframe(df)
                else:
                    st.warning("No data found for the specified time frame")
                    
            except Exception as e:
                st.error(f"Error generating heat map: {str(e)}")

# Tab 3: Person Tracker
with tab3:
    st.header("Track Person Movement")
    
    # Input field for person ID
    person_id_track = st.number_input("Person ID", min_value=1, step=1, key="track_person_id")
    
    # Button to track person
    if st.button("Track Person", key="track_btn"):
        with st.spinner("Tracking person..."):
            try:
                x_coords, y_coords, timestamps = asyncio.run(track_person(person_id_track))
                
                if len(x_coords) > 0:
                    # Create a DataFrame for plotting
                    df = pd.DataFrame({
                        'X': x_coords,
                        'Y': y_coords,
                        'Time': timestamps
                    })
                    
                    # Create a scatter plot with connected lines
                    fig = px.scatter(df, x='X', y='Y', color_discrete_sequence=['blue'],
                                    title=f"Movement Path for Person ID: {person_id_track}")
                    
                    # Add lines connecting the points
                    fig.add_trace(go.Scatter(
                        x=df['X'], 
                        y=df['Y'],
                        mode='lines',
                        line=dict(color='red', width=2),
                        name='Path'
                    ))
                    
                    # Add arrows to show direction
                    for i in range(len(df) - 1):
                        fig.add_annotation(
                            x=df['X'].iloc[i],
                            y=df['Y'].iloc[i],
                            ax=df['X'].iloc[i+1],
                            ay=df['Y'].iloc[i+1],
                            xref="x",
                            yref="y",
                            axref="x",
                            ayref="y",
                            showarrow=True,
                            arrowhead=3,
                            arrowsize=1,
                            arrowwidth=1,
                            arrowcolor="green"
                        )
                    
                    # Improve layout
                    fig.update_layout(
                        xaxis_title="X Coordinate",
                        yaxis_title="Y Coordinate",
                        showlegend=False,
                        height=600
                    )
                    
                    # Display the plot
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show the timestamps in an expander
                    with st.expander("View Tracking Timeline"):
                        time_df = pd.DataFrame({
                            'Point': range(1, len(timestamps) + 1),
                            'Time': timestamps,
                            'X': x_coords,
                            'Y': y_coords
                        })
                        st.dataframe(time_df)
                else:
                    st.warning(f"No tracking data found for Person ID: {person_id_track}")
                    
            except Exception as e:
                st.error(f"Error tracking person: {str(e)}")

# Tab 4: Person Info
with tab4:
    st.header("Person Information")
    
    # Input field for person ID
    person_id_info = st.number_input("Person ID", min_value=1, step=1, key="info_person_id")
    
    # Button to fetch person info
    if st.button("Get Person Info", key="info_btn"):
        with st.spinner("Fetching person information..."):
            try:
                person_info = asyncio.run(get_person_info(person_id_info))
                
                if person_info:
                    # Layout the information
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Basic Information")
                        st.write(f"*ID:* {person_info['person']['id']}")
                        st.write(f"*Height:* {person_info['person'].get('height', 'Unknown')} cm")
                        st.write(f"*Glasses:* {'Yes' if person_info['person'].get('glasses') else 'No/Unknown'}")
                        
                        gender_value = "Unknown"
                        if person_info.get('gender') and person_info['gender'].get('value'):
                            gender_value = person_info['gender']['value']
                        st.write(f"*Gender:* {gender_value}")
                        
                        age_value = "Unknown"
                        if person_info.get('age') and person_info['age'].get('value'):
                            age_value = person_info['age']['value']
                        st.write(f"*Age Group:* {age_value}")
                        
                        race_value = "Unknown"
                        if person_info.get('race') and person_info['race'].get('value'):
                            race_value = person_info['race']['value']
                        st.write(f"*Race:* {race_value}")
                        
                        hairline_type = "Unknown"
                        if person_info.get('hairline') and person_info['hairline'].get('type'):
                            hairline_type = person_info['hairline']['type']
                        st.write(f"*Hairline:* {hairline_type}")
                        
                        # if person_info['person'].get('feature'):
                        #     st.write(f"*Notable Feature:* {person_info['person']['feature']}")
                        # Replace the feature text display with base64 image display
                        # if person_info['person'].get('base64'):
                        #     st.subheader("Person Image")
                        #     st.image(person_info['person']['base64'], caption=f"Person ID: {person_info['person']['id']}")
                        if person_info['person'].get('base64'):
                            try:
                                import base64
                                from io import BytesIO
                                from PIL import Image
                                
                                # Add proper padding if needed
                                base64_data = person_info['person']['base64']
                                # Add padding if necessary
                                padding = len(base64_data) % 4
                                if padding:
                                    base64_data += '=' * (4 - padding)
                                    
                                # Try to display the image
                                image_bytes = base64.b64decode(base64_data)
                                image = Image.open(BytesIO(image_bytes))
                                st.image(image, caption=f"Person ID: {person_info['person']['id']}")
                            except Exception as e:
                                st.warning(f"Could not display image: {str(e)}")
                                # Fall back to displaying base64 data length
                                st.write(f"*Image data available:* {len(person_info['person']['base64'])} bytes")

                    with col2:
                        st.subheader("Apparel Information")
                        if person_info.get('apparel') and len(person_info['apparel']) > 0:
                            for item in person_info['apparel']:
                                st.write(f"*Shirt Color:* {item.get('shirt_colour', 'Unknown')}")
                                st.write(f"*Pant Color:* {item.get('pant_colour', 'Unknown')}")
                                if item.get('time'):
                                    st.write(f"*Logged at:* {item.get('time')}")
                        else:
                            st.write("No apparel information available")
                    
                    # Additional tracking information
                    st.subheader("Recent Tracking Data")
                    x_coords, y_coords, timestamps = asyncio.run(track_person(person_id_info))
                    
                    if len(x_coords) > 0:
                        # Create a small tracking visualization
                        fig = px.scatter(
                            x=x_coords, 
                            y=y_coords, 
                            title="Recent Movement",
                            labels={'x': 'X Coordinate', 'y': 'Y Coordinate'}
                        )
                        fig.add_trace(go.Scatter(
                            x=x_coords, 
                            y=y_coords,
                            mode='lines',
                            line=dict(color='green', width=2)
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show tracking data table
                        with st.expander("View Detailed Tracking Data"):
                            track_df = pd.DataFrame({
                                'Time': timestamps,
                                'X': x_coords,
                                'Y': y_coords
                            })
                            st.dataframe(track_df)
                    else:
                        st.info("No tracking data available for this person")
                else:
                    st.warning(f"No information found for Person ID: {person_id_info}")
                    
            except Exception as e:
                st.error(f"Error fetching person information: {str(e)}")