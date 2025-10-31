import streamlit as st
import joblib
import numpy as np
import pandas as pd



model = joblib.load("random_forest_model_of_accident_project.pkl")

st.set_page_config(page_title="🚗 Road Accident Severity Predictor", page_icon="🚦", layout="wide")

st.title("🚗 Road Accident Severity Prediction System")
st.caption("Developed by **Yuvraj** | Powered by Streamlit & Scikit-learn")
st.markdown("---")




def encode_choice(label, mapping):
    choice = st.selectbox(label, list(mapping.keys()))
    return mapping[choice]



st.markdown("---")
with st.expander("📘 VIEW FULL ENCODING REFERENCE (Click to Expand)", expanded=False):
    st.markdown("""
    ## 🚦 ENCODING REFERENCE — ACCIDENT DATA COLUMNS

    ---
    ### 📍 Type_of_vehicle
    | Category | Code |
    |-----------|------|
    | Automobile | 0.1 |
    | Public (> 45 seats) | 0.2 |
    | Lorry (41–100Q) | 0.3 |
    | Public (13–45 seats) | 0.4 |
    | Lorry (11–40Q) | 0.5 |
    | Long lorry | 0.6 |
    | Public (12 seats) | 0.7 |
    | Taxi | 0.8 |
    | Pick up upto 10Q | 0.9 |
    | Stationwagen | 1.0 |
    | Ridden horse | 1.1 |
    | Other | 1.2 |
    | Bajaj | 1.3 |
    | Turbo | 1.4 |
    | Motorcycle | 1.5 |
    | Special vehicle | 1.6 |
    | Bicycle | 1.7 |

    ---
    ### 📍 Defect_of_vehicle
    | Category | Code |
    |-----------|------|
    | No defect | 0.1 |
    | 5 | 0.2 |
    | 7 | 0.3 |

    ---
    ### 📍 Area_accident_occured
    | Category | Code |
    |-----------|------|
    | Residential areas | 0.1 |
    | Office areas | 0.2 |
    | Recreational areas | 0.3 |
    | Industrial areas | 0.4 |
    | Other | 0.5 |
    | Church areas | 0.6 |
    | Market areas | 0.7 |
    | Rural village areas | 0.8 |
    | Outside rural areas | 0.9 |
    | Hospital areas | 1.0 |
    | School areas | 1.1 |
    | Rural+Office mixed | 1.2 |
    | Recreational extended | 1.3 |

    ---
    ### 📍 Lanes_or_Medians
    | Type | Code |
    |-------|------|
    | Undivided Two way | 0 |
    | Two-way (broken lines) | 1 |
    | Two-way (solid lines) | 2 |
    | Double carriageway (median) | 3 |
    | One way | 4 |
    | Other | 5 |

    ---
    ### 📍 Road_allignment
    | Type | Code |
    |-------|------|
    | Tangent road (flat) | 0.1 |
    | Tangent road (mild grade) | 0.2 |
    | Escarpments | 0.3 |
    | Rolling terrain | 0.4 |
    | Gentle curve | 0.5 |
    | Mountainous | 0.6 |
    | Steep downward | 0.7 |
    | Sharp curve | 0.8 |
    | Steep upward | 0.9 |

    ---
    ### 📍 Types_of_Junction
    | Type | Code |
    |-------|------|
    | No junction | 0.1 |
    | Y Shape | 0.2 |
    | Crossing | 0.3 |
    | O Shape | 0.4 |
    | Other | 0.5 |
    | T Shape | 0.6 |
    | X Shape | 0.7 |

    ---
    ### 📍 Road_surface_type
    | Type | Code |
    |-------|------|
    | Asphalt | 0.1 |
    | Earth | 0.2 |
    | Asphalt (some distress) | 0.3 |
    | Gravel | 0.4 |
    | Other | 0.5 |

    ---
    ### 📍 Road_surface_conditions
    | Condition | Code |
    |------------|------|
    | Dry | 0.1 |
    | Wet/Damp | 0.2 |
    | Snow | 0.3 |
    | Flooded >3cm | 0.4 |

    ---
    ### 📍 Weather_conditions
    | Condition | Code |
    |------------|------|
    | Normal | 0.1 |
    | Raining | 0.2 |
    | Raining & Windy | 0.3 |
    | Cloudy | 0.4 |
    | Other | 0.5 |
    | Windy | 0.6 |
    | Snow | 0.7 |
    | Fog/Mist | 0.8 |

    ---
    ### 📍 Type_of_collision
    | Type | Code |
    |-------|------|
    | Parked vehicle | 0.1 |
    | Vehicle-to-vehicle | 0.2 |
    | Roadside object | 0.3 |
    | Animal | 0.4 |
    | Other | 0.5 |
    | Rollover | 0.6 |
    | Fall from vehicle | 0.7 |
    | Pedestrian | 0.8 |
    | Train | 0.9 |

    ---
    ### 📍 Vehicle_movement
    | Type | Code |
    |-------|------|
    | Going straight | 0.1 |
    | U-turn | 0.2 |
    | Moving backward | 0.3 |
    | Turnover | 0.4 |
    | Waiting to go | 0.5 |
    | Getting off | 0.6 |
    | Reversing | 0.7 |
    | Parked | 0.8 |
    | Stopping | 0.9 |
    | Overtaking | 1.0 |
    | Other | 1.1 |
    | Entering junction | 1.2 |

    ---
    ### 📍 Pedestrian_movement
    | Type | Code |
    |-------|------|
    | Not a pedestrian | 0.1 |
    | Crossing nearside | 0.2 |
    | Masked crossing | 0.3 |
    | Unknown/Other | 0.4 |
    | Crossing offside | 0.5 |
    | Standing in carriageway | 0.6 |
    | Walking back to traffic | 0.7 |
    | Facing traffic | 0.8 |
    | Standing masked | 0.9 |

    ---
    ### 📍 Cause_of_accident
    | Cause | Code |
    |--------|------|
    | Moving backward | 0.1 |
    | Overtaking | 0.2 |
    | Changing lane left | 0.3 |
    | Changing lane right | 0.4 |
    | Overloading | 0.5 |
    | Other | 0.6 |
    | No priority to vehicle | 0.7 |
    | No priority to pedestrian | 0.8 |
    | No distancing | 0.9 |
    | Improper exit | 1.0 |
    | Improper parking | 1.1 |
    | Overspeed | 1.2 |
    | Careless driving | 1.3 |
    | High speed | 1.4 |
    | Driving to left | 1.5 |
    | Overturning | 1.6 |
    | Turnover | 1.7 |
    | Under drugs | 1.8 |
    | Drunk driving | 1.9 |

    ---
    ### 📍 Age_band_of_driver
    | Age Group | Code |
    |------------|------|
    | Under 18 | 0 |
    | 18–30 | 1 |
    | 31–50 | 2 |
    | Over 51 | 3 |

    ---
    ### 📍 Educational_level
    | Level | Code |
    |--------|------|
    | Illiterate | 0 |
    | Writing & Reading | 1 |
    | Elementary | 2 |
    | Junior high | 3 |
    | High school | 4 |
    | Above high school | 5 |

    ---
    ### 📍 Driving_experience
    | Experience | Code |
    |-------------|------|
    | No licence | 0 |
    | Below 1yr | 1 |
    | 1–2yrs | 2 |
    | 2–5yrs | 3 |
    | 5–10yrs | 4 |
    | Above 10yrs | 5 |

    ---
    ### 📍 Service_year_of_vehicle
    | Years | Code |
    |--------|------|
    | Below 1yr | 0 |
    | 1–2yrs | 1 |
    | 2–5yrs | 2 |
    | 5–10yrs | 3 |
    | Above 10yrs | 4 |

    ---
    ### 📍 Light_conditions
    | Condition | Code |
    |------------|------|
    | Darkness - no lighting | 0 |
    | Darkness - unlit | 1 |
    | Darkness - lights lit | 2 |
    | Daylight | 3 |

    ---
    ### 📍 Accident_severity (Output)
    | Severity | Code |
    |-----------|------|
    | Slight Injury | 0 |
    | Serious Injury | 1 |
    | Fatal Injury | 2 |
    """)









age_band = {
    "Under 18": 0, "18–30": 1, "31–50": 2, "Over 51": 3
}

education = {
    "Illiterate": 0, "Writing & Reading": 1, "Elementary": 2,
    "Junior high": 3, "High school": 4, "Above high school": 5
}

experience = {
    "No licence": 0, "Below 1 yr": 1, "1–2 yrs": 2, "2–5 yrs": 3,
    "5–10 yrs": 4, "Above 10 yrs": 5
}

vehicle_type = {
    "Automobile": 0.1, "Public (>45 seats)": 0.2, "Lorry (41–100Q)": 0.3,
    "Public (13–45 seats)": 0.4, "Lorry (11–40Q)": 0.5, "Long lorry": 0.6,
    "Public (12 seats)": 0.7, "Taxi": 0.8, "Pick up upto 10Q": 0.9,
    "Stationwagen": 1.0, "Ridden horse": 1.1, "Other": 1.2, "Bajaj": 1.3,
    "Turbo": 1.4, "Motorcycle": 1.5, "Special vehicle": 1.6, "Bicycle": 1.7
}

service_year = {
    "Below 1 yr": 0, "1–2 yrs": 1, "2–5 yrs": 2, "5–10 yrs": 3, "Above 10 yrs": 4
}

vehicle_defect = {
    "No defect": 0.1, "Minor defect": 0.2, "Major defect": 0.3
}

area = {
    "Residential": 0.1, "Office": 0.2, "Recreational": 0.3, "Industrial": 0.4,
    "Other": 0.5, "Church": 0.6, "Market": 0.7, "Rural village": 0.8,
    "Outside rural": 0.9, "Hospital": 1.0, "School": 1.1
}

lanes = {
    "Undivided two-way": 0, "Two-way (broken)": 1, "Two-way (solid)": 2,
    "Double carriageway": 3, "One-way": 4, "Other": 5
}

alignment = {
    "Flat (tangent)": 0.1, "Mild grade": 0.2, "Escarpments": 0.3,
    "Rolling": 0.4, "Gentle curve": 0.5, "Mountainous": 0.6,
    "Steep downward": 0.7, "Sharp curve": 0.8, "Steep upward": 0.9
}

junction = {
    "No junction": 0.1, "Y-shape": 0.2, "Crossing": 0.3, "O-shape": 0.4,
    "Other": 0.5, "T-shape": 0.6, "X-shape": 0.7
}

surface_type = {
    "Asphalt": 0.1, "Earth": 0.2, "Asphalt (distressed)": 0.3,
    "Gravel": 0.4, "Other": 0.5
}

surface_cond = {
    "Dry": 0.1, "Wet/Damp": 0.2, "Snow": 0.3, "Flooded": 0.4
}

light_cond = {
    "Dark - no lighting": 0, "Dark - unlit": 1,
    "Dark - lights lit": 2, "Daylight": 3
}

weather = {
    "Normal": 0.1, "Raining": 0.2, "Raining & Windy": 0.3, "Cloudy": 0.4,
    "Other": 0.5, "Windy": 0.6, "Snow": 0.7, "Fog/Mist": 0.8
}

collision = {
    "Parked vehicle": 0.1, "Vehicle-to-vehicle": 0.2, "Roadside object": 0.3,
    "Animal": 0.4, "Other": 0.5, "Rollover": 0.6, "Fall from vehicle": 0.7,
    "Pedestrian": 0.8, "Train": 0.9
}

movement = {
    "Going straight": 0.1, "U-turn": 0.2, "Moving backward": 0.3,
    "Turnover": 0.4, "Waiting to go": 0.5, "Getting off": 0.6,
    "Reversing": 0.7, "Parked": 0.8, "Stopping": 0.9,
    "Overtaking": 1.0, "Other": 1.1, "Entering junction": 1.2
}

pedestrian = {
    "Not a pedestrian": 0.1, "Crossing nearside": 0.2,
    "Masked crossing": 0.3, "Unknown/Other": 0.4,
    "Crossing offside": 0.5, "Standing in carriageway": 0.6,
    "Walking back to traffic": 0.7, "Facing traffic": 0.8,
    "Standing masked": 0.9
}

cause = {
    "Moving backward": 0.1, "Overtaking": 0.2, "Changing lane left": 0.3,
    "Changing lane right": 0.4, "Overloading": 0.5, "Other": 0.6,
    "No priority to vehicle": 0.7, "No priority to pedestrian": 0.8,
    "No distancing": 0.9, "Improper exit": 1.0, "Improper parking": 1.1,
    "Overspeed": 1.2, "Careless driving": 1.3, "High speed": 1.4,
    "Driving to left": 1.5, "Overturning": 1.6, "Turnover": 1.7,
    "Under drugs": 1.8, "Drunk driving": 1.9
}






st.header("🧾 Enter Accident Details")

col1, col2, col3 = st.columns(3)

with col1:
    Age_band_of_driver = encode_choice("Age Band of Driver", age_band)
    Educational_level = encode_choice("Educational Level", education)
    Driving_experience = encode_choice("Driving Experience", experience)
    Type_of_vehicle = encode_choice("Type of Vehicle", vehicle_type)
    Service_year_of_vehicle = encode_choice("Service Year of Vehicle", service_year)
    Defect_of_vehicle = encode_choice("Defect of Vehicle", vehicle_defect)

with col2:
    Area_accident_occured = encode_choice("Area Accident Occurred", area)
    Lanes_or_Medians = encode_choice("Lanes or Medians", lanes)
    Road_allignment = encode_choice("Road Alignment", alignment)
    Types_of_Junction = encode_choice("Types of Junction", junction)
    Road_surface_type = encode_choice("Road Surface Type", surface_type)
    Road_surface_conditions = encode_choice("Road Surface Conditions", surface_cond)

with col3:
    Light_conditions = encode_choice("Light Conditions", light_cond)
    Weather_conditions = encode_choice("Weather Conditions", weather)
    Type_of_collision = encode_choice("Type of Collision", collision)
    Number_of_vehicles_involved = st.slider("Number of Vehicles Involved", 1, 10, 1)
    Vehicle_movement = encode_choice("Vehicle Movement", movement)
    Pedestrian_movement = encode_choice("Pedestrian Movement", pedestrian)
    Cause_of_accident = encode_choice("Cause of Accident", cause)

# =============================================
# 🔍 Prediction
# =============================================
if st.button("🚦 Predict Accident Severity"):
    features = np.array([[Age_band_of_driver, Educational_level, Driving_experience,
                          Type_of_vehicle, Service_year_of_vehicle, Defect_of_vehicle,
                          Area_accident_occured, Lanes_or_Medians, Road_allignment,
                          Types_of_Junction, Road_surface_type, Road_surface_conditions,
                          Light_conditions, Weather_conditions, Type_of_collision,
                          Number_of_vehicles_involved, Vehicle_movement, Pedestrian_movement,
                          Cause_of_accident]])
    
    result = model.predict(features)[0]

    if result == 0:
        st.success("🟢 **Slight Injury**")
    elif result == 1:
        st.warning("🟡 **Serious Injury**")
    else:
        st.error("🔴 **Fatal Injury**")

st.markdown("---")
st.caption("© 2025 Yuvraj | Accident Severity ML Predictor")









