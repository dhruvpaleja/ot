"""
🚛 Urban Logistics Optimization — Full Stack VRP Dashboard
MPSTME NMIMS Mumbai
Built with Streamlit + OR-Tools + Plotly + Folium
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, AntPath
from streamlit_folium import st_folium
import json
import math
import io
from vrp_engine import (
    Warehouse, DeliveryPoint, Vehicle, TrafficConfig, TrafficManager,
    HeuristicSolver, ORToolsSolver, ScenarioComparator,
    build_distance_matrix, get_default_warehouses, get_default_deliveries,
    get_default_vehicles, haversine, road_distance
)

# ─── HELPER: Convert hex to rgba for Plotly Sankey ───────────────────────────
def hex_to_rgba(hex_color, alpha=0.4):
    """Convert hex color (#RRGGBB) to rgba string with transparency for Plotly"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f'rgba({r}, {g}, {b}, {alpha})'
    return hex_color  # fallback

# ─── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="Urban Logistics VRP Optimizer",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600:700&family=Inter:wght@300:400:500:600&display=swap');

/* Dark theme overrides */
.stApp {
    background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 100%);
}

/* Header styling */
.main-header {
    background: linear-gradient(135deg, #1a1d2e 0%, #252a3a 100%);
    border: 1px solid #2d3348;
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.main-header h1 {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 2.4rem;
    background: linear-gradient(135deg, #4A9EFF, #3DBA7E, #FF8C42);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.main-header p {
    color: #7a849a;
    font-size: 0.95rem;
    margin-top: 4px;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1d2e, #1f2233);
    border: 1px solid #2d3348;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    transition: transform 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
}
.metric-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    margin: 4px 0;
}
.metric-label {
    color: #7a849a;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Section headers */
.section-header {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    font-size: 1.4rem;
    color: #e2e6f0;
    border-left: 4px solid #4A9EFF;
    padding-left: 12px;
    margin: 24px 0 16px 0;
}

/* Route badge */
.route-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px 4px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1117 0%, #151823 100%);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
}
</style>
""", unsafe_allow_html=True)

# ─── HEADER ────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🚛 Urban Logistics VRP Optimizer</h1>
    <p>MPSTME NMIMS Mumbai — Vehicle Routing Problem with Traffic & Signal Intelligence</p>
</div>
""", unsafe_allow_html=True)

# ─── SESSION STATE INITIALIZATION ──────────────────────────
if 'warehouses' not in st.session_state:
    st.session_state.warehouses = get_default_warehouses()
if 'deliveries' not in st.session_state:
    st.session_state.deliveries = get_default_deliveries()
if 'vehicles' not in st.session_state:
    st.session_state.vehicles = get_default_vehicles()
if 'solution' not in st.session_state:
    st.session_state.solution = None
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = None

# ─── SIDEBAR: CONFIGURATION PANEL ─────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration Panel")
    
    # ── SOLVER SETTINGS ──
    st.markdown("### 🧮 Solver Settings")
    solver_choice = st.selectbox("Solver Engine", 
        ["OR-Tools (Optimal)", "Heuristic (Fast)", "Both (Compare)"],
        help="OR-Tools: Google's constraint solver for optimal solutions\nHeuristic: Clarke-Wright Savings + 2-opt for fast solutions")

    time_limit = st.slider("Solver Time Limit (sec)", 5, 120, 30, 5) if "OR-Tools" in solver_choice else 30

    st.markdown("---")

    # ── TRAFFIC SETTINGS ──
    st.markdown("### 🚦 Traffic Settings")
    time_of_day = st.selectbox("Time of Day",
        list(TrafficManager.TIME_PRESETS.keys()),
        format_func=lambda x: TrafficManager.TIME_PRESETS[x]["label"],
        index=0)

    st.markdown("**Zone Traffic Intensity:**")
    zone_multipliers = {}
    zone_cols = st.columns(2)
    for idx, (zone, info) in enumerate(TrafficManager.MUMBAI_ZONES.items()):
        with zone_cols[idx % 2]:
            zone_multipliers[zone] = st.slider(
                f"{zone}", 0.5, 3.0, 1.0, 0.1,
                key=f"zone_{zone}",
                help=f"Base traffic: {info['base_traffic']}x")

    st.markdown("---")

    # ── SIGNAL SETTINGS ──
    st.markdown("### 🚥 Signal Management")
    signal_cycle = st.slider("Signal Cycle Time (sec)", 60, 240, 120, 10)
    green_ratio = st.slider("Green Phase Ratio", 0.2, 0.7, 0.45, 0.05)
    signals_per_km = st.slider("Avg Signals/km", 0.5, 5.0, 2.5, 0.5)

    st.markdown("---")

    # ── WAREHOUSE MANAGEMENT ──
    st.markdown("### 🏭 Warehouses")

    with st.expander("➕ Add New Warehouse", expanded=False):
        wh_name = st.text_input("Name", "New Warehouse", key="wh_name")
        wh_col1, wh_col2 = st.columns(2)
        wh_lat = wh_col1.number_input("Latitude", 18.8, 19.5, 19.1, 0.001, key="wh_lat")
        wh_lon = wh_col2.number_input("Longitude", 72.7, 73.2, 72.85, 0.001, key="wh_lon")
        wh_cap = st.number_input("Max Capacity", 100, 5000, 1000, 100, key="wh_cap")
        wh_cost = st.number_input("Daily Operating Cost (₹)", 100, 5000, 500, 50, key="wh_cost")
        if st.button("Add Warehouse", width="stretch", type="primary"):
            st.session_state.warehouses.append(
                Warehouse(wh_name, wh_lat, wh_lon, 
                         TrafficManager.get_zone(wh_name), wh_cap, wh_cost))
            st.rerun()

    for i, wh in enumerate(st.session_state.warehouses):
        col1, col2 = st.columns([3, 1])
        col1.markdown(f"🏭 **{wh.name}** (Cap: {wh.max_capacity})")
        if col2.button("🗑️", key=f"del_wh_{i}"):
            st.session_state.warehouses.pop(i)
            st.rerun()

    st.markdown("---")

    # ── VEHICLE MANAGEMENT ──
    st.markdown("### 🚛 Vehicles")

    with st.expander("➕ Add New Vehicle", expanded=False):
        v_name = st.text_input("Name", "Van-New", key="v_name")
        v_cap = st.number_input("Capacity (units)", 100, 2000, 400, 50, key="v_cap")
        v_cost = st.number_input("Cost/km (₹)", 1.0, 50.0, 12.0, 0.5, key="v_cost")
        v_co2 = st.number_input("CO₂/km (kg)", 0.01, 1.0, 0.21, 0.01, key="v_co2")
        v_speed = st.number_input("Speed (km/h)", 10.0, 80.0, 28.0, 1.0, key="v_speed")
        v_fuel = st.selectbox("Fuel Type", ["Diesel", "CNG", "Electric", "Petrol", "Hybrid"], key="v_fuel")
        if st.button("Add Vehicle", width="stretch", type="primary"):
            st.session_state.vehicles.append(
                Vehicle(v_name, v_cap, v_cost, v_co2, v_speed, v_fuel))
            st.rerun()

    for i, v in enumerate(st.session_state.vehicles):
        fuel_icon = {"Diesel": "⛽", "CNG": "🟢", "Electric": "⚡", "Petrol": "🔴", "Hybrid": "🔵"}.get(v.fuel_type, "🚛")
        col1, col2 = st.columns([3, 1])
        col1.markdown(f"{fuel_icon} **{v.name}** (Cap: {v.capacity})")
        if col2.button("🗑️", key=f"del_v_{i}"):
            st.session_state.vehicles.pop(i)
            st.rerun()

    st.markdown("---")

    # ── DELIVERY POINT MANAGEMENT ──
    st.markdown("### 📦 Delivery Points")

    with st.expander("➕ Add Delivery Point", expanded=False):
        dp_name = st.text_input("Name", "New Location", key="dp_name")
        dp_col1, dp_col2 = st.columns(2)
        dp_lat = dp_col1.number_input("Latitude", 18.8, 19.5, 19.05, 0.001, key="dp_lat")
        dp_lon = dp_col2.number_input("Longitude", 72.7, 73.2, 72.85, 0.001, key="dp_lon")
        dp_demand = st.number_input("Demand (units)", 10, 1000, 100, 10, key="dp_demand")
        dp_priority = st.selectbox("Priority", [1, 2, 3], 
            format_func=lambda x: {1: "Normal", 2: "High", 3: "Urgent"}[x], key="dp_priority")
        if st.button("Add Delivery Point", width="stretch", type="primary"):
            st.session_state.deliveries.append(
                DeliveryPoint(dp_name, dp_lat, dp_lon,
                            TrafficManager.get_zone(dp_name), dp_demand, (8, 18), dp_priority))
            st.rerun()

    st.markdown(f"**{len(st.session_state.deliveries)} delivery points loaded**")
    with st.expander("View All Delivery Points"):
        for i, dp in enumerate(st.session_state.deliveries):
            priority_badge = {1: "🟢", 2: "🟡", 3: "🔴"}[dp.priority]
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"{priority_badge} **{dp.name}** (D: {dp.demand})")
            if col2.button("🗑️", key=f"del_dp_{i}"):
                st.session_state.deliveries.pop(i)
                st.rerun()

    st.markdown("---")

    # ── LOAD/SAVE PROBLEM ──
    st.markdown("### 💾 Save/Load Problem")

    if st.button("📥 Export Problem as JSON", width="stretch"):
        problem_data = {
            "warehouses": [{"name": w.name, "lat": w.lat, "lon": w.lon, 
                           "capacity": w.max_capacity, "cost": w.operating_cost}
                          for w in st.session_state.warehouses],
            "deliveries": [{"name": d.name, "lat": d.lat, "lon": d.lon,
                           "demand": d.demand, "priority": d.priority}
                          for d in st.session_state.deliveries],
            "vehicles": [{"name": v.name, "capacity": v.capacity, "cost_per_km": v.cost_per_km,
                          "co2_per_km": v.co2_per_km, "speed": v.speed_kmh, "fuel": v.fuel_type}
                        for v in st.session_state.vehicles]
        }
        st.download_button("⬇️ Download JSON", json.dumps(problem_data, indent=2),
                          "vrp_problem.json", "application/json", width="stretch")

    uploaded = st.file_uploader("📤 Load Problem JSON", type=["json"])
    if uploaded:
        try:
            data = json.loads(uploaded.read())
            st.session_state.warehouses = [
                Warehouse(w["name"], w["lat"], w["lon"], 
                         TrafficManager.get_zone(w["name"]), w["capacity"], w.get("cost", 500))
                for w in data.get("warehouses", [])]
            st.session_state.deliveries = [
                DeliveryPoint(d["name"], d["lat"], d["lon"],
                            TrafficManager.get_zone(d["name"]), d["demand"], (8, 18), d.get("priority", 1))
                for d in data.get("deliveries", [])]
            st.session_state.vehicles = [
                Vehicle(v["name"], v["capacity"], v["cost_per_km"], v["co2_per_km"],
                       v.get("speed", 28), v.get("fuel", "Diesel"))
                for v in data.get("vehicles", [])]
            st.success("✅ Problem loaded!")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading: {e}")

# ─── BUILD TRAFFIC CONFIG ─────────────────────────────────
traffic_config = TrafficConfig(
    zone_multipliers=zone_multipliers,
    time_of_day=time_of_day,
    signal_cycle_time=signal_cycle,
    green_ratio=green_ratio,
    signals_per_km=signals_per_km
)

# ─── SOLVE BUTTON ─────────────────────────────────────────
col_solve1, col_solve2, col_solve3 = st.columns([1, 2, 1])
with col_solve2:
    solve_btn = st.button("🚀 OPTIMIZE ROUTES", width="stretch", type="primary")

if solve_btn:
    with st.spinner("🧮 Solving VRP... This may take a moment for large problems"):
        wh = st.session_state.warehouses
        dl = st.session_state.deliveries
        vh = st.session_state.vehicles
        
        if not wh or not dl or not vh:
            st.error("❌ Need at least 1 warehouse, 1 delivery point, and 1 vehicle!")
        else:
            if solver_choice == "OR-Tools (Optimal)":
                st.session_state.solution = ORToolsSolver.solve(wh, dl, vh, traffic_config, time_limit)
            elif solver_choice == "Heuristic (Fast)":
                st.session_state.solution = HeuristicSolver.solve(wh, dl, vh, traffic_config)
            else:
                st.session_state.solution = ORToolsSolver.solve(wh, dl, vh, traffic_config, time_limit)
            
            # Also run scenarios
            sc = "OR-Tools" if "OR-Tools" in solver_choice else "Heuristic"
            st.session_state.scenarios = ScenarioComparator.run_scenarios(
                wh, dl, vh, traffic_config, sc, min(time_limit, 15))
            
            st.success("✅ Optimization complete!")
            st.rerun()

# ─── DISPLAY RESULTS ──────────────────────────────────────
solution = st.session_state.solution
scenarios = st.session_state.scenarios

if solution and solution.routes:
    # ── TOP-LEVEL METRICS ──
    st.markdown('<div class="section-header">📊 Optimization Summary</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)

    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Cost</div>
            <div class="metric-value" style="color: #4A9EFF;">₹{solution.total_cost:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Distance</div>
            <div class="metric-value" style="color: #FF8C42;">{solution.total_distance:,.1f} km</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">CO₂ Emissions</div>
            <div class="metric-value" style="color: #3DBA7E;">{solution.total_co2:,.2f} kg</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Time</div>
            <div class="metric-value" style="color: #e8c76d;">{solution.total_time:,.0f} min</div>
        </div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Active Vehicles</div>
            <div class="metric-value" style="color: #BB86FC;">{len(solution.routes)}</div>
        </div>""", unsafe_allow_html=True)
    with m6:
        avg_util = np.mean([r.load_carried/r.capacity*100 for r in solution.routes]) if solution.routes else 0
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Avg Utilization</div>
            <div class="metric-value" style="color: #FF6B9D;">{avg_util:.0f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"<p style='text-align:center;color:#7a849a;margin-top:8px;'>Solver: <b>{solution.solver_used}</b></p>", unsafe_allow_html=True)

    if solution.unserved:
        st.warning(f"⚠️ Unserved deliveries: {', '.join(solution.unserved)}")

    # ─── MAIN TABS ─────────────────────────────────────────
    tabs = st.tabs([
        "🗺️ Route Map", "📊 Route Analysis", "🌿 Emissions", 
        "🚦 Traffic & Signals", "📈 Advanced Analytics",
        "📋 Delivery Plan", "⚡ Scenario Comparison"
    ])

    VEHICLE_COLORS = ['#4A9EFF', '#FF8C42', '#3DBA7E', '#BB86FC', '#FF6B9D', 
                      '#e8c76d', '#00BCD4', '#FF5252', '#69F0AE', '#FFD740']

    all_locations = st.session_state.warehouses + st.session_state.deliveries

    # ═══════════════════════════════════════════════════════
    # TAB 1: ROUTE MAP
    # ═══════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown('<div class="section-header">🗺️ Interactive Route Map</div>', unsafe_allow_html=True)
        
        map_style = st.radio("Map Style", ["Standard", "Satellite", "Dark"], horizontal=True)
        tile_map = {
            "Standard": "OpenStreetMap",
            "Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "Dark": "CartoDB dark_matter"
        }
        
        center_lat = np.mean([loc.lat for loc in all_locations])
        center_lon = np.mean([loc.lon for loc in all_locations])
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12,
                      tiles=tile_map[map_style] if map_style != "Satellite" else None)
        
        if map_style == "Satellite":
            folium.TileLayer(
                tiles=tile_map["Satellite"],
                attr="Esri", name="Satellite"
            ).add_to(m)
        
        # Add warehouse markers
        for wh in st.session_state.warehouses:
            folium.Marker(
                [wh.lat, wh.lon],
                popup=folium.Popup(f"""<div style='font-family:Inter;min-width:200px;'>
                    <h4 style='color:#4A9EFF;margin:0;'>🏭 {wh.name}</h4>
                    <hr style='margin:4px 0;'>
                    <b>Capacity:</b> {wh.max_capacity} units<br>
                    <b>Zone:</b> {wh.zone}<br>
                    <b>Daily Cost:</b> ₹{wh.operating_cost}
                </div>""", max_width=300),
                icon=folium.Icon(color='blue', icon='industry', prefix='fa'),
                tooltip=f"🏭 {wh.name}"
            ).add_to(m)
        
        # Add delivery markers
        for dp in st.session_state.deliveries:
            priority_color = {1: 'green', 2: 'orange', 3: 'red'}[dp.priority]
            folium.CircleMarker(
                [dp.lat, dp.lon],
                radius=max(6, dp.demand / 20),
                color=priority_color,
                fill=True,
                fillColor=priority_color,
                fillOpacity=0.7,
                popup=folium.Popup(f"""<div style='font-family:Inter;min-width:180px;'>
                    <h4 style='margin:0;'>📦 {dp.name}</h4>
                    <hr style='margin:4px 0;'>
                    <b>Demand:</b> {dp.demand} units<br>
                    <b>Priority:</b> {['', 'Normal', 'High', 'Urgent'][dp.priority]}<br>
                    <b>Zone:</b> {dp.zone}<br>
                    <b>Window:</b> {dp.time_window[0]}:00 - {dp.time_window[1]}:00
                </div>""", max_width=300),
                tooltip=f"📦 {dp.name} (D:{dp.demand})"
            ).add_to(m)
        
        # Draw routes
        wh_list = st.session_state.warehouses
        dl_list = st.session_state.deliveries
        all_locs = list(wh_list) + list(dl_list)
        
        for r_idx, route in enumerate(solution.routes):
            color = VEHICLE_COLORS[r_idx % len(VEHICLE_COLORS)]
            coords = []
            for loc_name in route.route_names:
                for loc in all_locs:
                    if loc.name == loc_name:
                        coords.append([loc.lat, loc.lon])
                        break
            
            if len(coords) > 1:
                # Animated path
                AntPath(
                    coords,
                    color=color,
                    weight=4,
                    opacity=0.8,
                    dash_array=[10, 20],
                    delay=1000,
                    tooltip=f"{route.vehicle_name}: {' → '.join(route.route_names)}"
                ).add_to(m)
                
                # Add route number labels at midpoints
                for k in range(len(coords)-1):
                    mid_lat = (coords[k][0] + coords[k+1][0]) / 2
                    mid_lon = (coords[k][1] + coords[k+1][1]) / 2
                    folium.Marker(
                        [mid_lat, mid_lon],
                        icon=folium.DivIcon(html=f"""
                            <div style='background:{color};color:white;border-radius:50%;
                                width:20px;height:20px;text-align:center;line-height:20px;
                                font-size:10px;font-weight:bold;'>{r_idx+1}</div>""")
                    ).add_to(m)
        
        # Traffic heatmap overlay
        show_traffic_heat = st.checkbox("Show Traffic Heatmap", value=False)
        if show_traffic_heat:
            heat_data = []
            for loc in all_locs:
                zone = TrafficManager.get_zone(loc.name)
                intensity = TrafficManager.MUMBAI_ZONES.get(zone, {"base_traffic": 1.0})["base_traffic"]
                mult = zone_multipliers.get(zone, 1.0)
                time_mult = TrafficManager.TIME_PRESETS.get(time_of_day, {"base": 1.0})["base"]
                total_intensity = intensity * mult * time_mult
                heat_data.append([loc.lat, loc.lon, total_intensity])
            HeatMap(heat_data, radius=30, blur=20, max_zoom=13).add_to(m)
        
        st_folium(m, width=None, height=600, returned_objects=[])
        
        # Route summary below map
        st.markdown("**Route Details:**")
        for r_idx, route in enumerate(solution.routes):
            color = VEHICLE_COLORS[r_idx % len(VEHICLE_COLORS)]
            st.markdown(f"""<div style='background:#1a1d2e;border-left:4px solid {color};
                padding:12px;border-radius:8px;margin:8px 0;'>
                <b style='color:{color};'>{route.vehicle_name}</b> | 
                {' → '.join(route.route_names)} | 
                📏 {route.total_distance:.1f} km | 
                💰 ₹{route.total_cost:,.0f} | 
                🌿 {route.total_co2:.2f} kg CO₂ | 
                ⏱️ {route.total_time:.0f} min |
                📦 {route.load_carried}/{route.capacity} units ({route.load_carried/route.capacity*100:.0f}%)
             </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # TAB 2: ROUTE ANALYSIS
    # ═══════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown('<div class="section-header">📊 Route Analysis Dashboard</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost per vehicle
            fig = go.Figure()
            names = [r.vehicle_name for r in solution.routes]
            costs = [r.total_cost for r in solution.routes]
            colors = [VEHICLE_COLORS[i % len(VEHICLE_COLORS)] for i in range(len(solution.routes))]
            
            fig.add_trace(go.Bar(x=names, y=costs, marker_color=colors,
                               text=[f'₹{c:,.0f}' for c in costs], textposition='outside'))
            fig.update_layout(title="💰 Transportation Cost per Vehicle",
                            xaxis_title="Vehicle", yaxis_title="Cost (₹)",
                            template="plotly_dark", height=400,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            # Distance per vehicle
            fig = go.Figure()
            dists = [r.total_distance for r in solution.routes]
            fig.add_trace(go.Bar(x=names, y=dists, marker_color=colors,
                               text=[f'{d:.1f} km' for d in dists], textposition='outside'))
            fig.update_layout(title="📏 Distance per Vehicle",
                            xaxis_title="Vehicle", yaxis_title="Distance (km)",
                            template="plotly_dark", height=400,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
            st.plotly_chart(fig, width="stretch")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Load distribution pie
            loads = [r.load_carried for r in solution.routes]
            fig = go.Figure(data=[go.Pie(
                labels=names, values=loads,
                marker=dict(colors=colors),
                hole=0.4,
                textinfo='label+percent+value'
            )])
            fig.update_layout(title="📦 Load Distribution",
                            template="plotly_dark", height=400,
                            paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width="stretch")
        
        with col4:
            # Vehicle utilization gauge
            fig = go.Figure()
            for i, route in enumerate(solution.routes):
                util = route.load_carried / route.capacity * 100
                fig.add_trace(go.Bar(
                    name=route.vehicle_name,
                    x=[route.vehicle_name],
                    y=[util],
                    marker_color=colors[i],
                    text=f'{util:.0f}%',
                    textposition='outside'
                ))
            fig.add_hline(y=80, line_dash="dash", line_color="#e8c76d", 
                         annotation_text="Target 80%")
            fig.update_layout(title="📊 Vehicle Utilization (%)",
                            yaxis_range=[0, 110], showlegend=False,
                            template="plotly_dark", height=400,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
            st.plotly_chart(fig, width="stretch")
        
        # Time breakdown stacked bar
        st.markdown("---")
        fig = go.Figure()
        base_times = [r.total_time - r.signal_delays - r.traffic_delay for r in solution.routes]
        fig.add_trace(go.Bar(name='Base Travel', x=names, y=base_times, 
                           marker_color='#4A9EFF'))
        fig.add_trace(go.Bar(name='Traffic Delay', x=names, 
                           y=[r.traffic_delay for r in solution.routes],
                           marker_color='#FF8C42'))
        fig.add_trace(go.Bar(name='Signal Delay', x=names, 
                           y=[r.signal_delays for r in solution.routes],
                           marker_color='#FF5252'))
        fig.update_layout(title="⏱️ Time Breakdown per Vehicle (minutes)",
                        barmode='stack', template="plotly_dark", height=400,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
        st.plotly_chart(fig, width="stretch")

    # ═══════════════════════════════════════════════════════
    # TAB 3: EMISSIONS
    # ═══════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown('<div class="section-header">🌿 CO₂ Emissions Dashboard</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CO2 per vehicle
            co2s = [r.total_co2 for r in solution.routes]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=names, y=co2s, marker_color=colors,
                               text=[f'{c:.2f} kg' for c in co2s], textposition='outside'))
            fig.update_layout(title="🌿 CO₂ per Vehicle",
                            xaxis_title="Vehicle", yaxis_title="CO₂ (kg)",
                            template="plotly_dark", height=400,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            # CO2 by fuel type
            fuel_co2 = {}
            for route in solution.routes:
                for v in st.session_state.vehicles:
                    if v.name == route.vehicle_name:
                        fuel_co2[v.fuel_type] = fuel_co2.get(v.fuel_type, 0) + route.total_co2
                        break
            
            fuel_colors = {"Diesel": "#FF5252", "CNG": "#3DBA7E", "Electric": "#4A9EFF", 
                          "Petrol": "#FF8C42", "Hybrid": "#BB86FC"}
            fig = go.Figure(data=[go.Pie(
                labels=list(fuel_co2.keys()),
                values=list(fuel_co2.values()),
                marker=dict(colors=[fuel_colors.get(f, '#999') for f in fuel_co2.keys()]),
                hole=0.45,
                textinfo='label+percent+value'
            )])
            fig.update_layout(title="⛽ CO₂ by Fuel Type",
                            template="plotly_dark", height=400,
                            paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width="stretch")
        
        # CO2 vs Cost scatter (Pareto)
        fig = go.Figure()
        for i, route in enumerate(solution.routes):
            fig.add_trace(go.Scatter(
                x=[route.total_cost], y=[route.total_co2],
                mode='markers+text',
                marker=dict(size=max(15, route.load_carried/10), color=colors[i], opacity=0.8),
                text=[route.vehicle_name],
                textposition='top center',
                name=route.vehicle_name
            ))
        fig.update_layout(title="💰 Cost vs CO₂ Tradeoff (Pareto View)",
                        xaxis_title="Cost (₹)", yaxis_title="CO₂ (kg)",
                        template="plotly_dark", height=450,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
        st.plotly_chart(fig, width="stretch")
        
        # Emission intensity (CO2 per unit delivered)
        st.markdown("**Emission Intensity (CO₂ per unit delivered):**")
        intensity_data = []
        for route in solution.routes:
            if route.load_carried > 0:
                intensity_data.append({
                    "Vehicle": route.vehicle_name,
                    "CO₂/unit": round(route.total_co2 / route.load_carried * 1000, 2),
                    "Units": f"{route.load_carried} / {route.capacity}"
                })
        if intensity_data:
            df_int = pd.DataFrame(intensity_data)
            fig = px.bar(df_int, x="Vehicle", y="CO₂/unit", color="Vehicle",
                        color_discrete_sequence=VEHICLE_COLORS,
                        title="🌱 CO₂ Intensity (g per unit delivered)")
            fig.update_layout(template="plotly_dark", height=350,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
            st.plotly_chart(fig, width="stretch")

    # ═══════════════════════════════════════════════════════
    # TAB 4: TRAFFIC & SIGNALS
    # ═══════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown('<div class="section-header">🚦 Traffic & Signal Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Zone traffic levels
            zones = list(TrafficManager.MUMBAI_ZONES.keys())
            time_mult = TrafficManager.TIME_PRESETS.get(time_of_day, {"base": 1.0})["base"]
            base_traffics = [TrafficManager.MUMBAI_ZONES[z]["base_traffic"] * time_mult * 
                           zone_multipliers.get(z, 1.0) for z in zones]
            
            zone_colors = ['#FF5252' if t > 2.0 else '#FF8C42' if t > 1.5 else '#3DBA7E' 
                          for t in base_traffics]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=zones, y=base_traffics, marker_color=zone_colors,
                               text=[f'{t:.2f}x' for t in base_traffics], textposition='outside'))
            fig.add_hline(y=1.5, line_dash="dash", line_color="#e8c76d",
                         annotation_text="Moderate Congestion")
            fig.add_hline(y=2.0, line_dash="dash", line_color="#FF5252",
                         annotation_text="Heavy Congestion")
            fig.update_layout(title="🚗 Traffic Multiplier by Zone",
                            yaxis_title="Traffic Multiplier",
                            template="plotly_dark", height=400,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            # Signal density
            signal_densities = [TrafficManager.MUMBAI_ZONES[z]["signals_per_km"] for z in zones]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=zones, y=signal_densities,
                               marker_color=['#4A9EFF' if s < 2.5 else '#FF8C42' if s < 3 else '#FF5252' 
                                            for s in signal_densities],
                               text=[f'{s:.1f}/km' for s in signal_densities], textposition='outside'))
            fig.update_layout(title="🚥 Signal Density by Zone",
                            yaxis_title="Signals per km",
                            template="plotly_dark", height=400,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
            st.plotly_chart(fig, width="stretch")
        
        # Traffic heatmap by zone and time
        st.markdown("---")
        time_periods = list(TrafficManager.TIME_PRESETS.keys())
        heatmap_data = []
        for z in zones:
            row = []
            for t in time_periods:
                base = TrafficManager.MUMBAI_ZONES[z]["base_traffic"]
                tmult = TrafficManager.TIME_PRESETS[t]["base"]
                row.append(round(base * tmult * zone_multipliers.get(z, 1.0), 2))
            heatmap_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=[TrafficManager.TIME_PRESETS[t]["label"] for t in time_periods],
            y=zones,
            colorscale=[[0, '#1a1d2e'], [0.3, '#3DBA7E'], [0.6, '#FF8C42'], [1.0, '#FF5252']],
            text=[[f'{v:.2f}x' for v in row] for row in heatmap_data],
            texttemplate='%{text}',
            textfont={"size": 12}
        ))
        fig.update_layout(title="🌡️ Traffic Intensity Heatmap (Zone × Time)",
                        template="plotly_dark", height=400,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
        st.plotly_chart(fig, width="stretch")
        
        # Green wave analysis for each route
        st.markdown("---")
        st.markdown('<div class="section-header">🟢 Green Wave Signal Optimization</div>', unsafe_allow_html=True)
        
        for r_idx, route in enumerate(solution.routes):
            with st.expander(f"🚛 {route.vehicle_name} — Green Wave Analysis"):
                # Calculate segment distances
                seg_dists = []
                for k in range(len(route.route_indices)-1):
                    d = solution.distance_matrix[route.route_indices[k]][route.route_indices[k+1]]
                    seg_dists.append(d)
                
                gw = TrafficManager.calculate_green_wave(route.route_names, seg_dists, traffic_config)
                
                if gw:
                    gw_df = pd.DataFrame(gw)
                    st.dataframe(gw_df, width="stretch", hide_index=True)
                    
                    # Signal timing visualization
                    fig = go.Figure()
                    for i, seg in enumerate(gw):
                        fig.add_trace(go.Bar(
                            name=f"{seg['from']} → {seg['to']}",
                            x=[f"Seg {i+1}"],
                            y=[seg['green_phase_sec']],
                            marker_color='#3DBA7E',
                            showlegend=False
                        ))
                        fig.add_trace(go.Bar(
                            name="Red Phase",
                            x=[f"Seg {i+1}"],
                            y=[signal_cycle - seg['green_phase_sec']],
                            marker_color='#FF5252',
                            showlegend=False
                        ))
                    fig.update_layout(barmode='stack', title=f"Signal Phases — {route.vehicle_name}",
                                    template="plotly_dark", height=300,
                                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
                    st.plotly_chart(fig, width="stretch")

    # ═══════════════════════════════════════════════════════
    # TAB 5: ADVANCED ANALYTICS
    # ═══════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown('<div class="section-header">📈 Advanced Analytics</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distance matrix heatmap
            all_locs = list(st.session_state.warehouses) + list(st.session_state.deliveries)
            loc_names = [l.name for l in all_locs]
            
            # Rebuild clean distance matrix for display
            display_matrix = build_distance_matrix(all_locs, traffic_config)
            
            fig = go.Figure(data=go.Heatmap(
                z=display_matrix,
                x=loc_names, y=loc_names,
                colorscale=[[0, '#0f1117'], [0.3, '#4A9EFF'], [0.7, '#FF8C42'], [1.0, '#FF5252']],
                text=np.round(display_matrix, 1),
                texttemplate='%{text}',
                textfont={"size": 8}
            ))
            fig.update_layout(title="📍 Distance Matrix Heatmap (km)",
                            template="plotly_dark", height=500,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            # Shadow prices
            if solution.shadow_prices:
                sp_names = list(solution.shadow_prices.keys())
                sp_values = list(solution.shadow_prices.values())
                sorted_idx = np.argsort(sp_values)[::-1]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[sp_values[i] for i in sorted_idx],
                    y=[sp_names[i] for i in sorted_idx],
                    orientation='h',
                    marker_color=['#FF5252' if sp_values[i] > np.median(sp_values) else '#3DBA7E' 
                                 for i in sorted_idx]
                ))
                fig.update_layout(title="💲 Shadow Prices (₹/unit) — Delivery Cost Efficiency",
                                xaxis_title="Shadow Price (₹/unit)",
                                template="plotly_dark", height=500,
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
                st.plotly_chart(fig, width="stretch")
        
        # Demand vs Capacity analysis
        st.markdown("---")
        col3, col4 = st.columns(2)
        
        with col3:
            # Demand per delivery point
            dp_names = [d.name for d in st.session_state.deliveries]
            dp_demands = [d.demand for d in st.session_state.deliveries]
            dp_colors = ['#FF5252' if d.priority == 3 else '#FF8C42' if d.priority == 2 else '#3DBA7E' 
                        for d in st.session_state.deliveries]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=dp_names, y=dp_demands, marker_color=dp_colors,
                               text=dp_demands, textposition='outside'))
            fig.update_layout(title="📦 Demand per Delivery Point",
                            xaxis_title="Location", yaxis_title="Demand (units)",
                            template="plotly_dark", height=400,
                            xaxis_tickangle=-45,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
            st.plotly_chart(fig, width="stretch")
        
        with col4:
            # Cumulative distance along routes
            fig = go.Figure()
            for r_idx, route in enumerate(solution.routes):
                cum_dist = [0]
                for k in range(len(route.route_indices)-1):
                    d = solution.distance_matrix[route.route_indices[k]][route.route_indices[k+1]]
                    cum_dist.append(cum_dist[-1] + d)
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(cum_dist))), y=cum_dist,
                    mode='lines+markers',
                    name=route.vehicle_name,
                    line=dict(color=VEHICLE_COLORS[r_idx % len(VEHICLE_COLORS)], width=2)
                ))
            fig.update_layout(title="📈 Cumulative Distance Along Route",
                            xaxis_title="Stop Number", yaxis_title="Cumulative Distance (km)",
                            template="plotly_dark", height=400,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
            st.plotly_chart(fig, width="stretch")
        
        # Sankey diagram: Warehouse → Delivery flow (✅ FIXED)
        st.markdown("---")
        sankey_labels = [wh.name for wh in st.session_state.warehouses]
        sankey_labels += [dp.name for dp in st.session_state.deliveries]
        
        sources = []
        targets = []
        values = []
        link_colors = []
        
        wh_offset = 0
        dp_offset = len(st.session_state.warehouses)
        
        for r_idx, route in enumerate(solution.routes):
            color = VEHICLE_COLORS[r_idx % len(VEHICLE_COLORS)]
            # Source is depot (warehouse 0)
            src_idx = 0
            for stop_name in route.route_names[1:-1]:  # skip depot at start and end
                for dp_idx, dp in enumerate(st.session_state.deliveries):
                    if dp.name == stop_name:
                        sources.append(src_idx)
                        targets.append(dp_offset + dp_idx)
                        values.append(dp.demand)
                        link_colors.append(color)
                        break
        
        if sources:
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=20, thickness=20,
                    line=dict(color="#2d3348", width=1),
                    label=sankey_labels,
                    color=['#4A9EFF'] * len(st.session_state.warehouses) + 
                          ['#3DBA7E'] * len(st.session_state.deliveries)
                ),
                link=dict(
                    source=sources, target=targets, value=values,
                    color=[hex_to_rgba(c, alpha=0.4) for c in link_colors]  # ✅ FIXED: rgba format
                )
            )])
            fig.update_layout(title="🔀 Warehouse → Delivery Flow (Sankey)",
                            template="plotly_dark", height=500,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
            st.plotly_chart(fig, width="stretch")
        
        # Radar chart per vehicle
        st.markdown("---")
        categories = ['Distance', 'Cost', 'CO₂', 'Time', 'Utilization']
        
        fig = go.Figure()
        for i, route in enumerate(solution.routes):
            max_dist = max(r.total_distance for r in solution.routes) or 1
            max_cost = max(r.total_cost for r in solution.routes) or 1
            max_co2 = max(r.total_co2 for r in solution.routes) or 1
            max_time = max(r.total_time for r in solution.routes) or 1
            
            vals = [
                route.total_distance / max_dist * 100,
                route.total_cost / max_cost * 100,
                route.total_co2 / max_co2 * 100,
                route.total_time / max_time * 100,
                route.load_carried / route.capacity * 100
            ]
            vals.append(vals[0])  # close the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                fill='toself',
                name=route.vehicle_name,
                line_color=VEHICLE_COLORS[i % len(VEHICLE_COLORS)],
                opacity=0.6
            ))
        
        fig.update_layout(
            title="🕸️ Vehicle Performance Radar",
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 110]),
                bgcolor='rgba(26,29,46,0.8)'
            ),
            template="plotly_dark", height=500,
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, width="stretch")

    # ═══════════════════════════════════════════════════════
    # TAB 6: DELIVERY PLAN
    # ═══════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown('<div class="section-header">📋 Delivery Plan & Recommendations</div>', unsafe_allow_html=True)
        
        # Full delivery plan table
        plan_rows = []
        for r_idx, route in enumerate(solution.routes):
            cum_dist = 0
            cum_time = 0
            for k in range(1, len(route.route_indices) - 1):
                stop_idx = route.route_indices[k]
                prev_idx = route.route_indices[k-1]
                
                seg_dist = solution.distance_matrix[prev_idx][stop_idx]
                cum_dist += seg_dist
                seg_time = (seg_dist / 30) * 60
                cum_time += seg_time
                
                # Find delivery point
                loc = None
                for dp in st.session_state.deliveries:
                    if dp.name == route.route_names[k]:
                        loc = dp
                        break
                
                if loc:
                    plan_rows.append({
                        "Vehicle": route.vehicle_name,
                        "Stop #": k,
                        "Location": loc.name,
                        "Zone": loc.zone,
                        "Demand": loc.demand,
                        "Priority": {1: "🟢 Normal", 2: "🟡 High", 3: "🔴 Urgent"}[loc.priority],
                        "Dist from Prev (km)": round(seg_dist, 1),
                        "Cum. Distance (km)": round(cum_dist, 1),
                        "Est. Arrival (min)": round(cum_time, 0),
                        "Time Window": f"{loc.time_window[0]}:00-{loc.time_window[1]}:00"
                    })
        
        if plan_rows:
            df_plan = pd.DataFrame(plan_rows)
            st.dataframe(df_plan, width="stretch", hide_index=True, height=500)
            
            # Download as CSV
            csv_buf = io.StringIO()
            df_plan.to_csv(csv_buf, index=False)
            st.download_button("📥 Download Delivery Plan (CSV)", csv_buf.getvalue(),
                              "delivery_plan.csv", "text/csv", width="stretch")
        
        # Recommendations
        st.markdown("---")
        st.markdown('<div class="section-header">💡 Delivery Recommendations</div>', unsafe_allow_html=True)
        
        recommendations = []
        
        # Check underutilized vehicles
        for route in solution.routes:
            util = route.load_carried / route.capacity * 100
            if util < 50:
                recommendations.append(
                    f"⚠️ **{route.vehicle_name}** is underutilized ({util:.0f}%). "
                    f"Consider using a smaller vehicle or consolidating routes.")
        
        # Check high-cost routes
        if solution.routes:
            avg_cost_per_km = np.mean([r.total_cost/max(r.total_distance,0.1) for r in solution.routes])
            for route in solution.routes:
                cpk = route.total_cost / max(route.total_distance, 0.1)
                if cpk > avg_cost_per_km * 1.3:
                    recommendations.append(
                        f"💰 **{route.vehicle_name}** has high cost/km (₹{cpk:.1f}/km vs avg ₹{avg_cost_per_km:.1f}/km). "
                        f"Route optimization or vehicle swap recommended.")
        
        # Check emissions
        for route in solution.routes:
            for v in st.session_state.vehicles:
                if v.name == route.vehicle_name and v.fuel_type == "Diesel":
                    recommendations.append(
                        f"🌿 Consider replacing **{route.vehicle_name}** (Diesel) with CNG/Electric "
                        f"to reduce CO₂ by ~{route.total_co2 * 0.6:.1f} kg on this route.")
                    break
        
        # Shadow price insights
        if solution.shadow_prices:
            expensive = max(solution.shadow_prices, key=solution.shadow_prices.get)
            cheapest = min(solution.shadow_prices, key=solution.shadow_prices.get)
            recommendations.append(
                f"📍 **{expensive}** has the highest shadow price (₹{solution.shadow_prices[expensive]:.1f}/unit). "
                f"Consider adding a closer warehouse.")
            recommendations.append(
                f"✅ **{cheapest}** is the most cost-efficient delivery point (₹{solution.shadow_prices[cheapest]:.1f}/unit).")
        
        if not recommendations:
            recommendations.append("✅ All routes look optimally configured!")
        
        for rec in recommendations:
            st.markdown(f"- {rec}")

    # ═══════════════════════════════════════════════════════
    # TAB 7: SCENARIO COMPARISON
    # ═══════════════════════════════════════════════════════
    with tabs[6]:
        st.markdown('<div class="section-header">⚡ Scenario Comparison</div>', unsafe_allow_html=True)
        
        if scenarios:
            sc_names = list(scenarios.keys())
            sc_costs = [scenarios[s].total_cost for s in sc_names]
            sc_dists = [scenarios[s].total_distance for s in sc_names]
            sc_co2s = [scenarios[s].total_co2 for s in sc_names]
            sc_times = [scenarios[s].total_time for s in sc_names]
            
            sc_colors = ['#4A9EFF', '#FF8C42', '#3DBA7E']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=sc_names, y=sc_costs, marker_color=sc_colors,
                                   text=[f'₹{c:,.0f}' for c in sc_costs], textposition='outside'))
                fig.update_layout(title="💰 Cost Comparison",
                                yaxis_title="Total Cost (₹)",
                                template="plotly_dark", height=400,
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
                st.plotly_chart(fig, width="stretch")
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=sc_names, y=sc_co2s, marker_color=sc_colors,
                                   text=[f'{c:.2f} kg' for c in sc_co2s], textposition='outside'))
                fig.update_layout(title="🌿 CO₂ Comparison",
                                yaxis_title="Total CO₂ (kg)",
                                template="plotly_dark", height=400,
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
                st.plotly_chart(fig, width="stretch")
            
            # Grouped comparison
            fig = go.Figure()
            metrics = ['Cost (₹)', 'Distance (km)', 'CO₂ (kg)', 'Time (min)']
            
            for i, sc_name in enumerate(sc_names):
                sc = scenarios[sc_name]
                # Normalize to percentage of baseline
                if sc_costs[0] > 0:
                    vals = [
                        sc.total_cost / sc_costs[0] * 100,
                        sc.total_distance / max(sc_dists[0], 0.1) * 100,
                        sc.total_co2 / max(sc_co2s[0], 0.1) * 100,
                        sc.total_time / max(sc_times[0], 0.1) * 100
                    ]
                else:
                    vals = [100, 100, 100, 100]
                
                fig.add_trace(go.Bar(
                    name=sc_name, x=metrics, y=vals,
                    marker_color=sc_colors[i],
                    text=[f'{v:.0f}%' for v in vals],
                    textposition='outside'
                ))
            
            fig.add_hline(y=100, line_dash="dash", line_color="#e8c76d",
                         annotation_text="Baseline (100%)")
            fig.update_layout(title="📊 Normalized Scenario Comparison (% of Baseline)",
                            barmode='group',
                            yaxis_title="% of Baseline",
                            template="plotly_dark", height=450,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
            st.plotly_chart(fig, width="stretch")
            
            # Detailed comparison table
            st.markdown("---")
            comp_data = []
            for sc_name in sc_names:
                sc = scenarios[sc_name]
                comp_data.append({
                    "Scenario": sc_name,
                    "Total Cost (₹)": f"₹{sc.total_cost:,.0f}",
                    "Total Distance (km)": f"{sc.total_distance:,.1f}",
                    "CO₂ (kg)": f"{sc.total_co2:,.2f}",
                    "Time (min)": f"{sc.total_time:,.0f}",
                    "Active Vehicles": len(sc.routes),
                    "Unserved": len(sc.unserved),
                    "Solver": sc.solver_used
                })
            st.dataframe(pd.DataFrame(comp_data), width="stretch", hide_index=True)
        else:
            st.info("Run optimization first to see scenario comparisons.")

else:
    # No solution yet — show getting started
    st.markdown(f"""
    <div style='text-align:center;padding:40px;'>
        <h2>🚛 Ready to Optimize!</h2>
        <p style='color:#7a849a;margin:16px 0;'>Configure your warehouses, vehicles, delivery points, and traffic settings in the sidebar.<br>
        Then click 🚀 OPTIMIZE ROUTES to solve the Vehicle Routing Problem.</p>
        <div style='display:flex;justify-content:center;gap:24px;margin-top:24px;'>
            <div style='text-align:center;'>
                <div style='font-size:2rem;'>🏭</div>
                <div>{len(st.session_state.warehouses)} Warehouses</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:2rem;'>📦</div>
                <div>{len(st.session_state.deliveries)} Delivery Points</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:2rem;'>🚛</div>
                <div>{len(st.session_state.vehicles)} Vehicles</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#7a849a;font-size:0.85rem;'>
🚛 Urban Logistics VRP Optimizer — MPSTME NMIMS Mumbai<br>
Built with Streamlit • OR-Tools • Plotly • Folium | 
Vehicle Routing Problem with Traffic & Signal Intelligence
</div>
""", unsafe_allow_html=True)
