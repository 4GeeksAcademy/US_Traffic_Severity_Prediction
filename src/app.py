from geopy.geocoders import GoogleV3
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re

# Set Webpage Title
st.set_page_config(page_title="US Accidents Severity Predicament - Cesar Tellechea and Gustavo Lima")

# Load the Model
model = pickle.load(open('../models/randomforest_boost_default_42.pkl', 'rb')) 

unique_cities = pd.read_csv('../data/processed/unique_cities.csv')
unique_states = pd.read_csv('../data/processed/unique_states.csv')
unique_weekdays = pd.read_csv('../data/processed/unique_weekdays.csv')
unique_months = pd.read_csv('../data/processed/unique_months.csv')
unique_sunrisesunset = pd.read_csv('../data/processed/unique_sunrisesunset.csv')
unique_winddirection = pd.read_csv('../data/processed/unique_winddirection.csv')
unique_days = pd.read_csv('../data/processed/unique_days.csv')


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

with open('../models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Configure Google API
GOOGLE_API_KEY = 'AIzaSyAk6abh1mmzH1aDXUmWDXypFEEoRPW-1Zk'
geolocator = GoogleV3(api_key=GOOGLE_API_KEY)

# Website Title
st.title("US Traffic Accidents Severity Predictament")

# Create buttons

def display_navigation_buttons():
    # Check if it's the last page
    if st.session_state.page == 1:
         if st.button("Next"):
                st.session_state.page += 1
                st.experimental_rerun()
    elif st.session_state.page == 7:
        pass
    else:
        but1, but2 = st.columns(2)
        with but1:
            if st.button("Previous"):
                st.session_state.page -= 1
                st.experimental_rerun()

        with but2:
            if st.button("Next"):
                st.session_state.page += 1
                st.experimental_rerun()
                

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
         
         Our data was all accident data from 2019, with 954302 accidents reports distributed
         over 56 Features as Weather, Time, Points of Interest, Location Data. 
         
         As for Modelling we trained a Logistic Regression model, a Random Forest model and a
         Gradient Boosting (XGBClassifier) model to get the most of our data. In the end we 
         committed to Random Forest with better hyperparameter boosting as it scored at top
         in accuracy score (83%).
         '''
         )
 
# 2nd Page   
def display_accidentsmap():
    st.write('### Accidents Distribution')
    st.write('''
        In this image we can analyze the distribution of 2019 accidents across US territory. 
        Most of the accidents happen in coastal areas of the US, which is where the
        majority of population is located.
        ''')
    st.image('../assets/01_accidents_map.png', caption='US Accidents Distribution', use_column_width=True)
    st.write('''
        In this histogram we fact check the map data with the distribution by state. As expected
        the 3 top states are 3 of the biggest and most populated states of the US, California, Texas
        and Florida. A curious observation is that New York, despite being the 4th most populated state, 
        has much lower number of accidents when compared to the the top 3. We know that in NY people use more 
        transportation and have less own vehicles when compared to CA, TX, and Florida. 
        Looking at Accidents per Popualation, this data completely changes, South Carolina, Oregon and Utah
        are the states where more accidents happen per population.
        ''')
    st.image('../assets/02_Hist_State.png', caption='US States Accidents Distribution', use_column_width=True)

# 3rd Page 
def display_accidentsdata():
    st.write('### Accidents Distribution by Time and Weather')
    st.write('''
        Accidents happen the most around Winter months (November, December) and for 2019 we can 
        also observe that March, Spring Month, there's also quite a lot of accidents. 
        ''')
    st.image('../assets/03_accidents_month.png', caption='US Accidents Distribution by Months', use_column_width=True)
    st.write('''
       There's an equal distribution of accidents for Week Days, minus Wednesday and Thursday where there
       seems to be less traffic issues. 
        ''')
    st.image('../assets/04_accidents_dayweek.png', caption='US Accidents Distribution distribution by Week Day', use_column_width=True)
    st.write('''
       Majority of accidents happen during the day, despite people thinking that night time is more
       prone to severe accidents and road dangers. 
        ''')
    st.image('../assets/05_accidents_timeofday.png', caption='US Accidents Distribution by Time of Day', use_column_width=True)
    st.write('''
       There's an idea that the harsher the weather, more prone to accidents. With this data we show that
       majority of the accidents happen in Clear / Cloudy weather conditions. Nonetheless, rain, fog and snow
       do have an impact on accidents severity
        ''')
    st.image('../assets/06_accidents_weather.png', caption='US Accidents Distribution by Weather Conditions', use_column_width=True)
 
# 4th page 
def display_holidayaccidents():
    st.write('### Accidents Data on Top US Holidays')
    st.write('''
        We picked up few random days to check the accidents numbers. These days are consider
        as regular normal day, without anything in special tied to them. 
        ''')
    st.image('../assets/12_accidents_rngday.png', caption='Random Normal Days Accidents', use_column_width=True)
    st.write('''
        Comparing the top US Holidays with regular days Data, we can conclude that there are small
        increases in the number of Accidents, specially in New Years Eve. We know that the majority
        of these days are associated with deviant behaviour (alcohol, drugs, etc...) which can lead to 
        less driving capacity and obviously more accidents. Still these days don't mean that there will be
        an increased severity level of accidents. 
        ''')
    st.image('../assets/13_accidentsholiday.png', caption='Top US Holiday Accident Data', use_column_width=True)
 
# 5th Page    
def display_accidentsroad():
    st.write('### Road Dangers vs Accident Severity')
    st.write('''
        We analyzed what kind of roads are more inclined to more severe accidents. We found out
        that Innerstates are the type of road that has more dangers to drivers. Many times the 
        road conditions, driver's speed and other road dangers, increase the risk of accidents in I- Roads
        ''')
    st.image('../assets/07_danger_by_roadtype.png', caption='Roads more prone to Accidents', use_column_width=True)
    st.write('''
        Crossings are very prone to severe accidents. It's plausible because for many drivers crossings without
        traffic lights are very confusing, and thus making them more open to accidents. Also people 
        lacking the full understanding of priority rules help increase the numbers. 
        ''')
    st.image('../assets/08_danger_crossing.png', caption='Number of Accidents on Crossings', use_column_width=True)
    st.write('''
       Traffic Signals are supposed to help drivers understand better their surroundings and help avoid
       accidents. With this data we can conclude that they are very unefficient. Perhaps it's time to 
       find better ways to signal road dangers or rules in ways drivers will respect / understand them better.
        ''')
    st.image('../assets/09_danger_trafficsignal.png', caption='Number of Accidents with nearby Traffic Signals', use_column_width=True)
    
# 6th Page
def display_accidentssafety():
    st.write('### Safety Mechanism on the Road')
    st.write('''
        We observed in our analysis that Road Bumps do serve an important role on avoiding 
        accidents. They do warn drivers to lower speed and also act as a speed braker. 
        ''')
    st.image('../assets/10_safe_bump.png', caption='Number of Accidents nearby / on Bumps', use_column_width=True)
    st.write('''
        Roundabouts are one of the safests mechanisms on the Road. There's barely no accidents
        nearby or on Roundabouts. It's true that they are annoying as a driver, but they do
        help reduce accident severity.
        ''')
    st.image('../assets/11_safe_roundabout.png', caption='Number of Accidents nearby / on Roundabouts', use_column_width=True)
   

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
                city = re.sub(r'[^a-zA-Z\s]+', '', city).strip()
                st.write(f"City: {city}")
                if city in unique_cities['city'].values:
                    city_var = unique_cities.loc[unique_cities['city'] == city, 'city_n'].iloc[0]
                else:
                    st.warning(f"Sorry, {city} has no accident records but that's a good thing :)")

                state = parts[-2]
                state = re.sub(r'[^a-zA-Z]+', '', state)
                st.write(f"State: {state}")
                if state in unique_states['state'].values:
                    state_var = unique_states.loc[unique_states['state'] == state, 'state_n'].iloc[0]
                else:
                    st.warning(f"Sorry, {state} has no accident records but that's a good thing :)")
            
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
    selected_street_type = st.selectbox("Choose Street Type", list(street_mapping.values()))
    abbreviated_street_type = {v: k for k, v in street_mapping.items()}[selected_street_type]

    # True Condition
    if abbreviated_street_type == 'Rd':
        rd_val = True
    elif abbreviated_street_type == 'I-':
        i_val = True
    elif abbreviated_street_type == 'St':
        st_val = True
    elif abbreviated_street_type == 'Dr':
        dr_val = True
    elif abbreviated_street_type == 'Ave':
        ave_val = True
    elif abbreviated_street_type == 'Blvd':
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
    selected_date = st.date_input("Select Date", min_value=None, max_value=None, value=None, key=None, help=None)

    if selected_date:
        month_var = selected_date.strftime("%b")
        month_var = unique_months.loc[unique_months['month'] == month_var, 'month_n'].iloc[0]
        weekday_var = selected_date.strftime("%a")
        weekday_var = unique_weekdays.loc[unique_weekdays['weekday'] == weekday_var, 'weekday_n'].iloc[0]
        day_var = selected_date.day 
    
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
    selected_weather = st.selectbox("Select Weather Condition", list(weather_mapping.values()))

    # Obtain the abbreviated weather name
    abbreviated_weather_name = {v: k for k, v in weather_mapping.items()}[selected_weather]

    # Set the corresponding condition to True
    if abbreviated_weather_name == 'clear':
        clear_val = True
    elif abbreviated_weather_name == 'cloud':
        cloud_val = True
    elif abbreviated_weather_name == 'rain':
        rain_val = True
    elif abbreviated_weather_name == 'heavy rain':
        heavyrain_val = True
    elif abbreviated_weather_name == 'snow':
        snow_val = True
    elif abbreviated_weather_name == 'heavy snow':
        heavysnow_val = True
    elif abbreviated_weather_name == 'fog':
        fog_val = True
    

    sunrisesunset_var = st.selectbox("Select Day or Night", unique_sunrisesunset['sunrisesunset'].unique())
    sunrisesunset_var = unique_sunrisesunset.loc[unique_sunrisesunset['sunrisesunset'] == sunrisesunset_var, 'sunrisesunset_n'].iloc[0]

    winddirection_var = st.selectbox("Select Wind Direction", unique_winddirection['winddirection'].unique())
    winddirection_var = unique_winddirection.loc[unique_winddirection['winddirection'] == winddirection_var, 'winddirection_n'].iloc[0]

    unit_system = st.toggle("Unit System (Metric/Imperial)", False)  # False para Metric, True para Imperial

    # Climate Variables
    if unit_system:
        temperature_var = st.number_input("Temperature (°F)", min_value=-40, max_value=120, value=70)
        humidity_var = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
        pressure_var = st.number_input("Pressure (inHg)", min_value=0, max_value=118, value=29)
        visibility_var = st.number_input("Visibility (mi)", min_value=0, max_value=10, value=10)
        windspeed_var = st.number_input("Wind Speed (mph)", min_value=0, max_value=150, value=0)
        precipitation_var = st.number_input("Precipitation (in)", min_value=0, max_value=10, value=0)
    else:
        temperature_var = st.number_input("Temperature (°C)", min_value=-40, max_value=49, value=21)
        humidity_var = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
        pressure_var = st.number_input("Pressure (mb)", min_value=0, max_value=3000, value=1013)
        visibility_var = st.number_input("Visibility (km)", min_value=0, max_value=16, value=16)
        windspeed_var = st.number_input("Wind Speed (km/h)", min_value=0, max_value=241, value=0)
        precipitation_var = st.number_input("Precipitation (mm)", min_value=0, max_value=254, value=0)

    # Convert Units if needed
    if not unit_system:
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
        
    if st.session_state.page == 7:    
        if st.button("Previous"):
    # Decrement the page counter in session state
            st.session_state.page -= 1
    # Rerun the app to reflect the new page
            st.experimental_rerun()

# Update the main section to switch between pages
if 'page' not in st.session_state:
    st.session_state.page = 1

if st.session_state.page == 1:
    display_greeting()
    if st.button("Next"):
        st.session_state.page = 2
        st.experimental_rerun()

elif st.session_state.page == 2:
    display_accidentsmap()
    display_navigation_buttons()

elif st.session_state.page == 3:
    display_accidentsdata()
    display_navigation_buttons()

elif st.session_state.page == 4:
    display_holidayaccidents()
    display_navigation_buttons()

elif st.session_state.page == 5:
    display_accidentsroad()
    display_navigation_buttons()
    
elif st.session_state.page == 6:
    display_accidentssafety()
    display_navigation_buttons()

elif st.session_state.page == 7:
    display_app()
    display_navigation_buttons()