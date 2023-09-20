from geopy.geocoders import GoogleV3
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pickle import load
import re
import calendar

# Set Webpage Title
st.set_page_config(page_title="US Accidents Severity Predicament - Cesar Tellechea and Gustavo Lima")

# Load the Model
model = load(open('../models/randomforest_boost_default_42.sav', 'rb')) 

original_data = pd.read_csv(r'../data/processed/US_Accidents_2019_Clean.csv')

# Numeric Columns for Standarization
num_variables = ['city_n', 'state_n', 
       'winddirection_n', 'sunrisesunset_n', 'month_n',
       'weekday_n', "startlat", "startlng", 'temperaturef', 'humidity', 'pressurein', 'visibilitymi',
       'windspeedmph', 'precipitationin', 'amenity', 'bump',
       'crossing', 'giveway', 'junction', 'noexit', 'railway', 'roundabout',
       'station', 'stop', 'trafficcalming', 'trafficsignal', 'day', 'minute', 'clear', 'cloud', 'rain',
       'heavyrain', 'snow', 'heavysnow', 'fog', 'Rd', 'I-', 'St', 'Dr', 'Ave', 'Blvd']

severity_dict = {
    "1": "Level 1",
    "2": "Level 2",
    "3": "Level 3",
    "4": "Level 4"
}

scaler = StandardScaler()
scaler.fit(original_data[num_variables])

# Configure Google API
GOOGLE_API_KEY = 'AIzaSyAk6abh1mmzH1aDXUmWDXypFEEoRPW-1Zk'  # Reemplaza con tu clave de API
geolocator = GoogleV3(api_key=GOOGLE_API_KEY)

# Website Title
st.title("US Traffic Accidents Severity Predictament")

# 1st page
def display_greeting():
    st.write('### The Project Proposition ')
    st.write(''' 
         The project goal is to predict the severity of a traffic accident in United States.
         
         With this model we can understand how to improve roads and save financial, human and time
         resources, along improving road security for US Drivers. By identifying the features that
         add up to increased accident severity, it's possible to then take action upon them.
         
         Our methodology for the Model started with a Descriptive Analysis of our data,
         followed by an Exploratory Data Analysis where we further understood the data we were
         working on, followed by Data Cleaning, Organizing and Feature Engineering to have the
         cleanest data possible to have a good working model. There was a lot of chaotic data
         that needed to be simplified, which helped achieving a way better model. Plus a lot
         of features that were deleted as they didn't had any effect on the models, which helps
         improve the performance. 
         
         As for Modelling we trained a Logistic Regression model, a Random Forest model and a
         Gradient Boosting (XGBClassifier) model to get the most of our data. In the end we 
         committed to Random Forest with better hyperparameter boosting as it scored at top
         in accuracy score (83%).
         '''
         )

# Define a function to display your app with buttons and sliders
def display_app():
    # Declare open variable
    startlat_var = 0
    startlng_var = 0
    city_var = 0
    state_var = 0
    
    # Widget to find direction
    address = st.text_input("Input your address:")
    
    # Address Finding Button 
    if st.button("Search Address"):
        location = geolocator.geocode(address)

        if location is not None:
            st.write(f"Address Found: {location.address}")
            st.write(f"Latitude: {location.latitude}")
            startlat_var = location.latitude
            st.write(f"Longitude: {location.longitude}")
            startlng_var = location.longitude

            parts = location.address.split(', ')
            if len(parts) >= 3:
                city = parts[-3]
                city = re.sub(r'[^a-zA-Z\s]', '', city)
                st.write(f"City: {city}")
                city_var = original_data.loc[original_data['city'] == city, 'city_n'].iloc[0]

                state = parts[-2]
                state = re.sub(r'[^a-zA-Z\s]', '', state)
                st.write(f"State: {state}")
                state_var = original_data.loc[original_data['state'] == state, 'state_n'].iloc[0]
            else:
                st.warning("Couldn't extract city and state from the address.")
        else:
            st.warning("Address not found. Try again.")
    
    # Road Type Dictionary 
    street_mapping = {
        'Rd': 'Road',
        'I-': 'Interstate',
        'St': 'Street',
        'Dr': 'Drive',
        'Ave': 'Avenue',
        'Blvd': 'Boulevard'
    }

    # Initialize the variables _val as false
    rd_val = False
    i_val = False
    st_val = False
    dr_val = False
    ave_val = False
    blvd_val = False

    # Road Type
    street_type = st.selectbox("Choose Street Type", list(street_mapping.keys()))
    full_street_name = street_mapping[street_type]  
    # True Condition
    if street_type == 'Rd':
        rd_val = True
    elif street_type == 'I-':
        i_val = True
    elif street_type == 'St':
        st_val = True
    elif street_type == 'Dr':
        dr_val = True
    elif street_type == 'Ave':
        ave_val = True
    elif street_type == 'Blvd':
        blvd_val = True

    st.markdown("""
        <style>
            .radio-columns > * {
                display: inline-block;
                margin-right: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='radio-columns'>", unsafe_allow_html=True)
        station_var = st.radio("Is there a Public Transportation Station nearby?", ("Yes", "No"))
        station_var = 1 if station_var == "Yes" else 0
        amenity_var = st.radio("Are there amenities nearby?", ("Yes", "No"))
        amenity_var = 1 if amenity_var == "Yes" else 0
        bump_var = st.radio("Is there a Bump nearby?", ("Yes", "No"))
        bump_var = 1 if bump_var == "Yes" else 0
        crossing_var = st.radio("Is there a Crossing nearby?", ("Yes", "No"))
        crossing_var = 1 if crossing_var == "Yes" else 0
        give_way_var = st.radio("Is there a Give Way nearby?", ("Yes", "No"))
        give_way_var = 1 if give_way_var == "Yes" else 0
        junction_var = st.radio("Is there a Junction nearby?", ("Yes", "No"))
        junction_var = 1 if junction_var == "Yes" else 0
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='radio-columns'>", unsafe_allow_html=True)
        no_exit_var = st.radio("Is there a No Exit nearby?", ("Yes", "No"))
        no_exit_var = 1 if no_exit_var == "Yes" else 0
        railway_var = st.radio("Is there a Railway nearby?", ("Yes", "No"))
        railway_var = 1 if railway_var == "Yes" else 0
        roundabout_var = st.radio("Is there a Roundabout nearby?", ("Yes", "No"))
        roundabout_var = 1 if roundabout_var == "Yes" else 0
        stop_var = st.radio("Is there a Stop nearby?", ("Yes", "No"))
        stop_var = 1 if stop_var == "Yes" else 0
        traffic_calming_var = st.radio("Is there Traffic Calming nearby?", ("Yes", "No"))
        traffic_calming_var = 1 if traffic_calming_var == "Yes" else 0
        traffic_signal_var = st.radio("Is there a Traffic Signal nearby?", ("Yes", "No"))
        traffic_signal_var = 1 if traffic_signal_var == "Yes" else 0
        st.markdown("</div>", unsafe_allow_html=True)

    # Time Variables
    
    # Months By Order
    def custom_sort_month(month):
        month_mapping = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        return month_mapping[month]

    sorted_months = sorted(original_data['month'].unique(), key=custom_sort_month)
    month_var = st.selectbox("Select Month", sorted_months)
    month_var = original_data.loc[original_data['month'] == month_var, 'month_n'].iloc[0]
    
    # WeekDays By Order
    def custom_sort(weekday):
        weekdays_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        return weekdays_order.index(weekday)

    # Sort the unique weekdays using the custom sorting function
    sorted_weekdays = sorted(original_data['weekday'].unique(), key=custom_sort)
    weekday_var = st.selectbox("Select Weekday", sorted_weekdays)
    weekday_var = original_data.loc[original_data['weekday'] == weekday_var, 'weekday_n'].iloc[0]

    # Sort the unique days of the month
    sorted_days = sorted(original_data['day'].unique())
    day_var = st.selectbox("Select Day of the Month", sorted_days)
    
    # Widgets to establish Accident Time
    selected_hour = st.number_input("Enter Accident Hour (0-23)", min_value=0, max_value=23, step=1)
    selected_minute = st.number_input("Enter Accident Minute (0-59)", min_value=0, max_value=59, step=1)
    
    # Validate that Time is correct
    if selected_hour < 0 or selected_hour > 23:
        st.warning("Hour must be between 0 and 23.")
    if selected_minute < 0 or selected_minute > 59:
        st.warning("Minutes must be between 0 and 59.")

    # Calculate the minute time of the day with Hour.
    minute_var = selected_hour * 60 + selected_minute

    # Weather Conditions Dictionary Mapping
    weather_mapping = {
        'clear': 'Clear',
        'cloud': 'Cloudy',
        'rain': 'Rainy',
        'heavyrain': 'Heavy Rain',
        'snow': 'Snowy',
        'heavysnow': 'Heavy Snow',
        'fog': 'Foggy'
    }

    # Initialize all weather conditions _val as false
    clear_val = False
    cloud_val = False
    rain_val = False
    heavyrain_val = False
    snow_val = False
    heavysnow_val = False
    fog_val = False

    # Weather Condtion Type
    selected_weather = st.selectbox("Select Weather Condition", list(weather_mapping.keys()))

    # Obtain the full name of the weather condition
    full_weather_name = weather_mapping[selected_weather]

    #  True Conditioning for 
    if selected_weather == 'clear':
        clear_val = True
    elif selected_weather == 'cloud':
        cloud_val = True
    elif selected_weather == 'rain':
        rain_val = True
    elif selected_weather == 'heavyrain':
        heavyrain_val = True
    elif selected_weather == 'snow':
        snow_val = True
    elif selected_weather == 'heavysnow':
        heavysnow_val = True
    elif selected_weather == 'fog':
        fog_val = True

    sunrisesunset_var = st.selectbox("Select Day or Night", original_data['sunrisesunset'].unique())
    sunrisesunset_var = original_data.loc[original_data['sunrisesunset'] == sunrisesunset_var, 'sunrisesunset_n'].iloc[0]

    winddirection_var = st.selectbox("Select Wind Direction", original_data['winddirection'].unique())
    winddirection_var = original_data.loc[original_data['winddirection'] == winddirection_var, 'winddirection_n'].iloc[0]

    # Widget para que el usuario elija las unidades
    unit_system = st.radio("Seleccione Unit System:", ("Metric", "Imperial"))

    # Climate Variables
    if unit_system == "Metric":
        temperature_var = st.number_input("Temperature (°C)", min_value=-40, max_value=49, value=21)
        humidity_var = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
        pressure_var = st.number_input("Pressure (mb)", min_value=0, max_value=3000, value=1013)
        visibility_var = st.number_input("Visibility (km)", min_value=0, max_value=16, value=16)
        windspeed_var = st.number_input("Wind Speed (km/h)", min_value=0, max_value=241, value=0)
        precipitation_var = st.number_input("Precipitation (mm)", min_value=0, max_value=254, value=0)
    else:
        temperature_var = st.number_input("Temperature (°F)", min_value=-40, max_value=120, value=70)
        humidity_var = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
        pressure_var = st.number_input("Pressure (inHg)", min_value=0, max_value=118, value=29)
        visibility_var = st.number_input("Visibility (mi)", min_value=0, max_value=10, value=10)
        windspeed_var = st.number_input("Wind Speed (mph)", min_value=0, max_value=150, value=0)
        precipitation_var = st.number_input("Precipitation (in)", min_value=0, max_value=10, value=0)

    # Convert Units if needed
    if unit_system == "Metric":
        temperature_var = (temperature_var * 9/5) + 32
        pressure_var = pressure_var * 0.02953
        visibility_var = visibility_var * 0.621371
        windspeed_var = windspeed_var / 1.60934
        precipitation_var = precipitation_var * 0.0393701

    # Predictament Button
    data_norm = None
    
    if st.button("Predict Accident Severity"):  
        data = np.array([[city_var, state_var, winddirection_var, sunrisesunset_var, 
                          month_var, weekday_var, startlat_var, startlng_var, temperature_var,
                          humidity_var, pressure_var, visibility_var, windspeed_var,
                          precipitation_var, amenity_var, bump_var, crossing_var,
                          give_way_var, junction_var, no_exit_var, railway_var,
                          roundabout_var, station_var,  stop_var, traffic_calming_var, 
                          traffic_signal_var, day_var, minute_var, clear_val, cloud_val,
                          rain_val, heavyrain_val, snow_val, heavysnow_val,fog_val, rd_val,
                          i_val, st_val, dr_val, ave_val, blvd_val,
                          ]
                         ]
                        )
    
        data_norm = scaler.transform(data)
            
    # Check if data_norm is not None before using it
    if data_norm is not None:
        # Obtain the Severity Level
        prediction = str(model.predict(data_norm)[0])
        pred_class = severity_dict[prediction]
        # Show Prediction
        st.write("Severity Level:", pred_class)

# Initialize session state to keep track of page
if 'page' not in st.session_state:
    st.session_state.page = 1

# Show pages based on user interaction
if st.session_state.page == 1:
    display_greeting()
    if st.button("Next"):
        st.session_state.page = 2
        st.experimental_rerun()

elif st.session_state.page == 2:
    display_app()