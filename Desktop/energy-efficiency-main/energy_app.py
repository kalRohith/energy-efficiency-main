import streamlit as st
import pickle
import numpy as np
from streamlit_extras.metric_cards import style_metric_cards
import requests
import json
import math
import re
import plotly.graph_objects as go

# Appliance data for Indian households (typical values)
APPLIANCE_DATA = {
    # Cooling appliances
    "ac": {"power_kw": 1.5, "cost_inr": 35000, "desc": "Split AC (1.5kW)", "type": "cooling"},
    "cooler": {"power_kw": 0.18, "cost_inr": 7000, "desc": "Air Cooler (180W)", "type": "cooling"},
    "window ac": {"power_kw": 1.2, "cost_inr": 28000, "desc": "Window AC (1.2kW)", "type": "cooling"},
    # Heating appliances
    "heater": {"power_kw": 2.0, "cost_inr": 2500, "desc": "Room Heater (2kW)", "type": "heating"},
    "oil heater": {"power_kw": 2.5, "cost_inr": 6000, "desc": "Oil Heater (2.5kW)", "type": "heating"},
    "blow heater": {"power_kw": 1.5, "cost_inr": 1800, "desc": "Blow Heater (1.5kW)", "type": "heating"},
    # Add more appliances as needed
}

# Carbon emission factor for Indian grid (kg CO₂ per kWh)
INDIA_GRID_EMISSION_FACTOR = 0.727  # 0.727 tCO2/MWh = 0.727 kg/kWh

# Conversion factor
EURO_TO_INR = 90

# --- Custom CSS for high-end look ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Montserrat', sans-serif;
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ef 100%);
    }
    .main {
        background: rgba(255,255,255,0.85) !important;
        border-radius: 18px;
        padding: 2rem 2rem 1rem 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
    }
    .stButton>button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.5em 2em;
    }
    .stSlider>div>div>div>div {
        background: #2a5298 !important;
    }
    .stSelectbox>div>div>div>div {
        color: #1e3c72 !important;
    }
    .metric-card {
        background: #f1f5fa;
        border-radius: 12px;
        padding: 1.2em 1em;
        box-shadow: 0 2px 8px 0 rgba(31, 38, 135, 0.07);
        margin-bottom: 1em;
    }
    .result-card {
        background: linear-gradient(90deg, #e0e7ef 0%, #f8fafc 100%);
        border-radius: 16px;
        padding: 1.5em 1em;
        box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.10);
        margin-bottom: 1.5em;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e3c72;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Load models ---
heating_model = pickle.load(open("models/xgboost_regression_model1.pkl","rb"))
cooling_model = pickle.load(open("models/model_cooling.pkl", "rb"))

# --- Sidebar for Inputs ---
st.sidebar.image("https://img.icons8.com/ios-filled/100/1e3c72/heat-map.png", width=80)
st.sidebar.markdown("<h2 style='color:#1e3c72;'>Building Parameters</h2>", unsafe_allow_html=True)

X5 = st.sidebar.selectbox("Number of floors", ["One", "Two"])
if X5 == "One":
    X2 = st.sidebar.slider("Surface area (m²)", 686, 808, 720)
else:
    X2 = st.sidebar.slider("Surface area (m²)", 514, 661, 540)
X7 = st.sidebar.selectbox("Window size", ["Small", "Medium", "Large"])
X6 = st.sidebar.selectbox("Building Orientation", ["North", "East", "South", "West"])

# New synthetic features
insulation_quality = st.sidebar.selectbox("Insulation Quality", ["Poor", "Average", "Good"])
building_age = st.sidebar.slider("Building Age (years)", 1, 100, 20)
occupancy_level = st.sidebar.slider("Occupancy Level (people)", 1, 10, 3)
hvac_system_type = st.sidebar.selectbox("HVAC System Type", ["Central", "Split", "Window", "None"])
climate_zone = st.sidebar.selectbox("Climate Zone", ["Cold", "Temperate", "Hot"])
appliance_efficiency = st.sidebar.selectbox("Appliance Efficiency", ["Low", "Medium", "High"])

st.markdown("""
<div style='display:flex; align-items:center; gap:1em;'>
    <img src='https://img.icons8.com/ios-filled/100/1e3c72/heat-map.png' width='48'/>
    <h1 style='margin-bottom:0; color:#1e3c72;'>Energy Load Predictor</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("<p style='color:#2a5298; font-size:1.1rem;'>Predict the heating and cooling loads of your building and estimate your investment and annual costs with a high-end, data-driven tool.</p>", unsafe_allow_html=True)

st.markdown("---")

# --- Advanced 3D Simulation Model of the Building (Plotly) ---

# Calculate building dimensions
floor_height = 7.5  # meters
num_floors = 1 if X5 == "One" else 2
height = num_floors * floor_height
area = X2
width = depth = area ** 0.5  # Assume square footprint

# Window size mapping for color and transparency
window_color_map = {"Small": "#b0c4de", "Medium": "#4682b4", "Large": "#1e3c72"}
window_alpha_map = {"Small": 0.3, "Medium": 0.5, "Large": 0.7}
window_color = window_color_map[X7]
window_alpha = window_alpha_map[X7]

# Orientation arrow direction
orientation_map = {"North": (0, 1), "East": (1, 0), "South": (0, -1), "West": (-1, 0)}
arrow_dx, arrow_dy = orientation_map[X6]
arrow_length = width * 0.8

# Vertices of the building block (rectangular prism)
x = [0, width, width, 0, 0, width, width, 0]
y = [0, 0, depth, depth, 0, 0, depth, depth]
z = [0, 0, 0, 0, height, height, height, height]

# Faces of the building (each as a list of vertex indices)
faces = [
    [0, 1, 2, 3],  # bottom
    [4, 5, 6, 7],  # top
    [0, 1, 5, 4],  # front
    [1, 2, 6, 5],  # right
    [2, 3, 7, 6],  # back
    [3, 0, 4, 7],  # left
]

# Color faces: front and back faces (windows) colored by window size and transparency, others light gray
face_colors = ["#e0e7ef", "#e0e7ef", window_color, "#e0e7ef", window_color, "#e0e7ef"]
face_opacities = [0.85, 0.85, window_alpha, 0.85, window_alpha, 0.85]

# Create mesh for each face
mesh_traces = []
for i, face in enumerate(faces):
    mesh_traces.append(go.Mesh3d(
        x=[x[j] for j in face],
        y=[y[j] for j in face],
        z=[z[j] for j in face],
        color=face_colors[i],
        opacity=face_opacities[i],
        i=[0, 1, 2],
        j=[1, 2, 3],
        k=[2, 3, 0],
        showscale=False,
        name=f"Face {i+1}"
    ))

# Add a door on the front face (centered, 1m wide, 2.1m high)
door_width = width * 0.2
if door_width < 1: door_width = 1
if door_width > width * 0.5: door_width = width * 0.5

door_height = 2.1

door_x = [width/2 - door_width/2, width/2 + door_width/2, width/2 + door_width/2, width/2 - door_width/2]
door_y = [0, 0, 0, 0]
door_z = [0, 0, door_height, door_height]
door_trace = go.Mesh3d(
    x=door_x,
    y=door_y,
    z=door_z,
    color="#8B5A2B",
    opacity=0.95,
    i=[0, 1, 2],
    j=[1, 2, 3],
    k=[2, 3, 0],
    showscale=False,
    name="Door"
)

# Add a ground plane
ground_x = [-(width*0.2), width*1.2, width*1.2, -(width*0.2)]
ground_y = [-(depth*0.2), -(depth*0.2), depth*1.2, depth*1.2]
ground_z = [0, 0, 0, 0]
ground_trace = go.Mesh3d(
    x=ground_x,
    y=ground_y,
    z=ground_z,
    color="#b7e0cd",
    opacity=0.5,
    i=[0, 1, 2],
    j=[1, 2, 3],
    k=[2, 3, 0],
    showscale=False,
    name="Ground"
)

# Add stick-figure people (occupancy)
people_traces = []
people_per_row = int(np.ceil(np.sqrt(occupancy_level)))
spacing_x = width / (people_per_row + 1)
spacing_y = depth / (people_per_row + 1)
count = 0
for i in range(people_per_row):
    for j in range(people_per_row):
        if count >= occupancy_level:
            break
        px = (i + 1) * spacing_x
        py = (j + 1) * spacing_y
        # Body (vertical line)
        people_traces.append(go.Scatter3d(
            x=[px, px],
            y=[py, py],
            z=[0.5, 2.0],
            mode="lines",
            line=dict(color="#111", width=6),
            showlegend=False,
            name="Person"
        ))
        # Head (marker)
        people_traces.append(go.Scatter3d(
            x=[px],
            y=[py],
            z=[2.3],
            mode="markers",
            marker=dict(size=8, color="#777", symbol="circle"),
            showlegend=False,
            name="Head"
        ))
        count += 1
    if count >= occupancy_level:
        break

# Add orientation arrow (as a 3D scatter line)
arrow_x = [width/2, width/2 + arrow_dx * arrow_length]
arrow_y = [depth/2, depth/2 + arrow_dy * arrow_length]
arrow_z = [height + 0.5, height + 0.5]
arrow_trace = go.Scatter3d(
    x=arrow_x,
    y=arrow_y,
    z=arrow_z,
    mode="lines+markers+text",
    line=dict(color="#ff6600", width=9),
    marker=dict(size=8, color="#ff6600"),
    text=["", X6],
    textposition="top center",
    name="Orientation"
)

# Add a sun marker for orientation
sun_x = width/2 + arrow_dx * (width*0.9)
sun_y = depth/2 + arrow_dy * (depth*0.9)
sun_z = height + 2
sun_trace = go.Scatter3d(
    x=[sun_x],
    y=[sun_y],
    z=[sun_z],
    mode="markers+text",
    marker=dict(size=18, color="#FFD700", symbol="circle"),
    text=["☀️"],
    textposition="bottom center",
    name="Sun"
)

# Add a sunlight beam (cone) from sun to window face
beam_x = [sun_x, width/2]
beam_y = [sun_y, depth/2]
beam_z = [sun_z, height/2]
beam_trace = go.Scatter3d(
    x=beam_x,
    y=beam_y,
    z=beam_z,
    mode="lines",
    line=dict(color="#FFD700", width=12, dash="dot"),
    showlegend=False,
    name="Sunlight"
)

# Compose the figure
fig3d = go.Figure(data=mesh_traces + [door_trace, ground_trace] + people_traces + [arrow_trace, sun_trace, beam_trace])
fig3d.update_layout(
    scene=dict(
        xaxis_title="Width (m)",
        yaxis_title="Depth (m)",
        zaxis_title="Height (m)",
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=height/max(width, depth)),
        bgcolor="#222"
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    showlegend=False,
    title="3D Simulation of Your Building"
)

st.markdown("<div class='section-title'>3D Simulation of Your Building</div>", unsafe_allow_html=True)
st.plotly_chart(fig3d, use_container_width=True)

# --- Prediction Button ---
if st.button("🔮 Predict", use_container_width=True):
    # Map categorical features to numbers as in training
    insulation_map = {"Poor": 0, "Average": 1, "Good": 2}
    hvac_map = {"Central": 0, "Split": 1, "Window": 2, "None": 3}
    climate_map = {"Cold": 0, "Temperate": 1, "Hot": 2}
    appliance_map = {"Low": 0, "Medium": 1, "High": 2}

    X1 = -0.00119112 * X2 + 1.5642495965572887
    X8 = 3  # Example default value for X8
    if X5 == "One" :
        X3 = X2 - 441
        X4 = 220.5
        X5_num = 3.5
    else:
        X3 = 0.68571429 * X2 - 78
        X4 = 110.25
        X5_num = 7
    X6_num = {"North": 2, "East": 3, "South": 4, "West": 5}[X6]
    X7_num = {"Small": 0.1, "Medium": 0.25, "Large": 0.4}[X7]

    # Prepare feature vector for model (order: X1, X2, X3, X4, X5, X6, X7, X8, insulation, age, occupancy, hvac, climate, appliance)
    features = np.array([
        X1, X2, X3, X4, X5_num, X6_num, X7_num, X8,
        insulation_map[insulation_quality],
        building_age,
        occupancy_level,
        hvac_map[hvac_system_type],
        climate_map[climate_zone],
        appliance_map[appliance_efficiency]
    ]).reshape(1, -1)

    # For now, use only the first 8 features for prediction (since model expects 8), but keep the rest for future use/visualization
    hpred = heating_model.predict(features[:, :8].astype(np.float64))
    cpred = cooling_model.predict(features[:, :8].astype(np.float64))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(label="🔥 Heating Load (kVA)", value=int(hpred[0]))
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(label="❄️ Cooling Load (kVA)", value=int(cpred[0]))
        st.markdown("</div>", unsafe_allow_html=True)

    # --- System Sizing and Cost Calculation ---
    HEATING_PUMP_CAPACITY = 14  # in kWh
    AC_CAPACITY = 9  # in kWh
    HEATING_PUMP_COST = 11000 * EURO_TO_INR  # in ₹
    AC_COST = 6000 * EURO_TO_INR  # in ₹
    heating_load = int(hpred[0]) 
    cooling_load = int(cpred[0])
    num_heating_pumps = math.ceil(heating_load / HEATING_PUMP_CAPACITY)
    num_acs_for_heating = math.ceil(max(0, heating_load - num_heating_pumps * HEATING_PUMP_CAPACITY) / AC_CAPACITY)
    num_acs_for_cooling = math.ceil(cooling_load / AC_CAPACITY)
    total_acs_needed = max(num_acs_for_heating, num_acs_for_cooling)
    total_cost = (num_heating_pumps * HEATING_PUMP_COST) + (total_acs_needed * AC_COST)

    # st.markdown("<div class='section-title'>System Recommendation</div>", unsafe_allow_html=True)
    # st.markdown(f"""
    # <div class='result-card'>
    # <ul style='font-size:1.1em;'>
    #     <li><b>Number of heating pumps needed:</b> <span style='color:#1e3c72;'>{num_heating_pumps}</span></li>
    #     <li><b>Total number of additional ACs needed:</b> <span style='color:#1e3c72;'>{total_acs_needed}</span></li>
    #     <li><b>Estimated investment needed:</b> <span style='color:#2a5298;'>₹{total_cost:,.0f}</span></li>
    # </ul>
    # </div>
    # """, unsafe_allow_html=True)

    # --- Annual Cost Estimates ---
    cost_per_unit = 0.31 * EURO_TO_INR  # ₹/kWh
    average_annual_consumption_h = 95 * X2
    yearly_heating_bill = average_annual_consumption_h * cost_per_unit
    average_annual_consumption_c = 48 * X2
    yearly_cooling_bill = average_annual_consumption_c * cost_per_unit

    st.markdown("<div class='section-title'>Annual Cost Estimates</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='result-card'>
    <ul style='font-size:1.1em;'>
        <li>💡 <b>Estimated annual costs for heating:</b> <span style='color:#1e3c72;'>₹{yearly_heating_bill:,.0f}</span></li>
        <li>💧 <b>Estimated annual costs for cooling:</b> <span style='color:#2a5298;'>₹{yearly_cooling_bill:,.0f}</span></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # --- Advanced Visualizations ---
    fig = go.Figure(data=[
        go.Bar(name='Predicted', x=['Heating', 'Cooling'], y=[heating_load, cooling_load], marker_color=['#1e3c72', '#2a5298']),
        go.Bar(name='Typical', x=['Heating', 'Cooling'], y=[120, 80], marker_color=['#b0c4de', '#b0e0e6'])
    ])
    fig.update_layout(barmode='group', xaxis_title='Load Type', yaxis_title='Load (kVA)',
                      legend_title_text='Legend', template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    pie_fig = go.Figure(data=[
        go.Pie(labels=['Heating', 'Cooling'], values=[yearly_heating_bill, yearly_cooling_bill],
               marker_colors=['#1e3c72', '#2a5298'], hole=0.4)
    ])
    pie_fig.update_traces(textinfo='percent+label', pull=[0.05, 0.05])
    st.plotly_chart(pie_fig, use_container_width=True)

    with st.expander("See detailed summary"):
        st.write(f"**Inputs:**")
        st.write(f"- Surface Area: {X2} m²")
        st.write(f"- Number of Floors: {X5}")
        st.write(f"- Window Size: {X7}")
        st.write(f"- Building Orientation: {X6}")
        st.write(f"- Insulation Quality: {insulation_quality}")
        st.write(f"- Building Age: {building_age}")
        st.write(f"- Occupancy Level: {occupancy_level}")
        st.write(f"- HVAC System Type: {hvac_system_type}")
        st.write(f"- Climate Zone: {climate_zone}")
        st.write(f"- Appliance Efficiency: {appliance_efficiency}")
        st.write(f"\n**Predicted Heating Load:** {heating_load} kVA")
        st.write(f"**Predicted Cooling Load:** {cooling_load} kVA")
        st.write(f"**Estimated Annual Heating Cost:** ₹{yearly_heating_bill:,.0f}")
        st.write(f"**Estimated Annual Cooling Cost:** ₹{yearly_cooling_bill:,.0f}")

    # --- Appliance Recommendation for Indian Household ---
    st.markdown("<div class='section-title'>Appliance Options for Your Load (India)</div>", unsafe_allow_html=True)
    appliance_results = []
    for a, data in APPLIANCE_DATA.items():
        # For heating, use only heating appliances; for cooling, use only cooling appliances
        if heating_load > cooling_load:
            if data['type'] == 'heating':
                req_load = heating_load
            else:
                continue
        else:
            if data['type'] == 'cooling':
                req_load = cooling_load
            else:
                continue
        num = math.ceil(req_load / data['power_kw'])
        total_cost = num * data['cost_inr']
        # Assume 8 hours/day, 300 days/year usage for annual energy (customize as needed)
        annual_kwh = num * data['power_kw'] * 8 * 300
        annual_co2 = annual_kwh * INDIA_GRID_EMISSION_FACTOR
        appliance_results.append({
            'appliance': data['desc'],
            'num': num,
            'unit_kw': data['power_kw'],
            'unit_cost': data['cost_inr'],
            'total_cost': total_cost,
            'type': data['type'],
            'annual_kwh': annual_kwh,
            'annual_co2': annual_co2
        })
    # Sort by total cost, then by number of units
    appliance_results = sorted(appliance_results, key=lambda x: (x['total_cost'], x['num']))
    # Display as table
    st.markdown("<div style='overflow-x:auto;'>", unsafe_allow_html=True)
    st.write("**Appliance Requirement Table:**")
    st.dataframe(
        {"Appliance": [r['appliance'] for r in appliance_results],
         "Units Needed": [r['num'] for r in appliance_results],
         "Unit Power (kW)": [r['unit_kw'] for r in appliance_results],
         "Unit Cost (₹)": [r['unit_cost'] for r in appliance_results],
         "Total Cost (₹)": [r['total_cost'] for r in appliance_results]},
        use_container_width=True
    )
    # Bar chart for cost comparison
    bar_fig = go.Figure(data=[
        go.Bar(
            x=[r['appliance'] for r in appliance_results],
            y=[r['total_cost'] for r in appliance_results],
            marker_color=['#2a5298' if i==0 else '#b0c4de' for i in range(len(appliance_results))],
            text=[f"{r['num']} units" for r in appliance_results],
            textposition='auto',
        )
    ])
    bar_fig.update_layout(
        xaxis_title='Appliance',
        yaxis_title='Total Cost (₹)',
        template='plotly_white',
        showlegend=False
    )
    st.plotly_chart(bar_fig, use_container_width=True)
    # Highlight the most cost-effective option for cooling and heating
    best_cooling = next((r for r in appliance_results if r['type'] == 'cooling'), None)
    best_heating = next((r for r in appliance_results if r['type'] == 'heating'), None)
    if best_cooling and best_heating:
        st.success(f"Best Cooling Option: {best_cooling['appliance']} (x{best_cooling['num']}) for ₹{best_cooling['total_cost']:,}")
        st.success(f"Best Heating Option: {best_heating['appliance']} (x{best_heating['num']}) for ₹{best_heating['total_cost']:,}")
    elif best_cooling:
        st.success(f"Best Cooling Option: {best_cooling['appliance']} (x{best_cooling['num']}) for ₹{best_cooling['total_cost']:,}")
    elif best_heating:
        st.success(f"Best Heating Option: {best_heating['appliance']} (x{best_heating['num']}) for ₹{best_heating['total_cost']:,}")

    # --- Carbon Footprint Section ---
    st.markdown("<div class='section-title'>Estimated Annual Carbon Footprint</div>", unsafe_allow_html=True)
    st.dataframe(
        {"Appliance": [r['appliance'] for r in appliance_results],
         "Units Needed": [r['num'] for r in appliance_results],
         "Annual Energy (kWh)": [round(r['annual_kwh']) for r in appliance_results],
         "Annual CO₂ (kg)": [round(r['annual_co2']) for r in appliance_results]},
        use_container_width=True
    )
    co2_fig = go.Figure(data=[
        go.Bar(
            x=[r['appliance'] for r in appliance_results],
            y=[r['annual_co2'] for r in appliance_results],
            marker_color=['#2a5298' if i==0 else '#b0c4de' for i in range(len(appliance_results))],
            text=[f"{round(r['annual_co2'])} kg" for r in appliance_results],
            textposition='auto',
        )
    ])
    co2_fig.update_layout(
        xaxis_title='Appliance',
        yaxis_title='Annual CO₂ Emissions (kg)',
        template='plotly_white',
        showlegend=False
    )
    st.plotly_chart(co2_fig, use_container_width=True)
    # Suggestions to lower emissions
    st.info("""
    **Suggestions to Lower Emissions:**
    - Choose appliances with lower power ratings and higher efficiency.
    - Reduce daily usage hours where possible.
    - Prefer cooling options like fans and coolers over ACs for moderate climates.
    - For heating, use efficient heaters and insulate your home to reduce demand.
    - Consider renewable energy sources (e.g., rooftop solar) to offset grid emissions.
    """)

