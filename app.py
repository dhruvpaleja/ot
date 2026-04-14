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
from datetime import datetime

from vrp_engine import (
    Warehouse, DeliveryPoint, Vehicle, TrafficConfig, TrafficManager,
    HeuristicSolver, ORToolsSolver, ScenarioComparator,
    build_distance_matrix, get_default_warehouses, get_default_deliveries,
    get_default_vehicles, haversine, road_distance,
)
from advanced_features import (
    Depot, DriverAssignment, DriverManager,
    MultiDepotAllocator, DeliveryClusterer,
    PDFRouteExporter, AdvancedVisualizations,
    DemandPredictor, TravelTimeForecaster,
    PickupDeliveryRequest, PickupDeliverySolver,
    SplitDeliveryOptimizer,
    RLDispatcher, RealTimeReoptimizer,
)

# ─── HELPERS ───────────────────────────────────────────────

def hex_to_rgba(hex_color: str, alpha: float = 0.4) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return hex_color

VEHICLE_COLORS = [
    "#4A9EFF", "#FF8C42", "#3DBA7E", "#BB86FC", "#FF6B9D",
    "#e8c76d", "#00BCD4", "#FF5252", "#69F0AE", "#FFD740",
]

# ─── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="Urban Logistics VRP Optimizer",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

.stApp { background: linear-gradient(135deg,#0f1117 0%,#1a1d2e 100%); }

.main-header {
    background: linear-gradient(135deg,#1a1d2e,#252a3a);
    border: 1px solid #2d3348; border-radius: 16px;
    padding: 24px 32px; margin-bottom: 24px; text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,.3);
}
.main-header h1 {
    font-family: 'Rajdhani', sans-serif; font-weight: 700; font-size: 2.4rem;
    background: linear-gradient(135deg,#4A9EFF,#3DBA7E,#FF8C42);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
}
.main-header p { color: #7a849a; font-size: .95rem; margin-top: 4px; }

.metric-card {
    background: linear-gradient(135deg,#1a1d2e,#1f2233);
    border: 1px solid #2d3348; border-radius: 12px;
    padding: 20px; text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,.2); transition: transform .2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-value { font-family:'Rajdhani',sans-serif; font-size:2rem; font-weight:700; margin:4px 0; }
.metric-label { color:#7a849a; font-size:.8rem; text-transform:uppercase; letter-spacing:1px; }

.section-header {
    font-family:'Rajdhani',sans-serif; font-weight:600; font-size:1.4rem;
    color:#e2e6f0; border-left:4px solid #4A9EFF;
    padding-left:12px; margin:24px 0 16px 0;
}
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f1117,#151823); }
.stTabs [data-baseweb="tab-list"] { gap:4px; }
.stTabs [data-baseweb="tab"] { border-radius:8px; padding:8px 16px; }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🚛 Urban Logistics VRP Optimizer</h1>
    <p>MPSTME NMIMS Mumbai — Vehicle Routing Problem with Traffic, ML & Signal Intelligence</p>
</div>
""", unsafe_allow_html=True)

# ─── SESSION STATE ─────────────────────────────────────────
defaults = {
    "warehouses": get_default_warehouses(),
    "deliveries": get_default_deliveries(),
    "vehicles": get_default_vehicles(),
    "solution": None,
    "scenarios": None,
    "drivers": [
        DriverAssignment("DRV001", "Rajesh Kumar",   "Van-A (Diesel)", 8.0, 20.0, "9876540001", "LMV", "", 10.0, 0, 88.0),
        DriverAssignment("DRV002", "Amit Sharma",    "Van-B (CNG)",    8.0, 20.0, "9876540002", "LMV", "", 10.0, 0, 75.0),
        DriverAssignment("DRV003", "Priya Nair",     "Van-C (Electric)",9.0,18.0, "9876540003", "LMV", "", 9.0,  0, 92.0),
        DriverAssignment("DRV004", "Suresh Patil",   "Van-D (Diesel)", 8.0, 20.0, "9876540004", "LMV", "", 10.0, 0, 81.0),
        DriverAssignment("DRV005", "Kavya Reddy",    "Van-E (CNG)",    8.0, 20.0, "9876540005", "LMV", "", 10.0, 0, 69.0),
    ],
    "pdp_requests": [],
    "rl_decisions": [],
    "demand_predictor": None,
    "travel_forecaster": None,
    "reoptimizer": RealTimeReoptimizer(),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration Panel")

    # Solver
    st.markdown("### 🧮 Solver")
    solver_choice = st.selectbox("Engine",
        ["OR-Tools (Optimal)", "Heuristic (Fast)", "Both (Compare)"])
    time_limit = st.slider("Time Limit (sec)", 5, 120, 30, 5) if "OR-Tools" in solver_choice else 30

    st.markdown("---")

    # Traffic
    st.markdown("### 🚦 Traffic")
    time_of_day = st.selectbox("Time of Day",
        list(TrafficManager.TIME_PRESETS.keys()),
        format_func=lambda x: TrafficManager.TIME_PRESETS[x]["label"])
    st.markdown("**Zone Multipliers:**")
    zone_multipliers = {}
    zc = st.columns(2)
    for idx, (zone, info) in enumerate(TrafficManager.MUMBAI_ZONES.items()):
        with zc[idx % 2]:
            zone_multipliers[zone] = st.slider(zone, 0.5, 3.0, 1.0, 0.1, key=f"z_{zone}")

    st.markdown("---")

    # Signals
    st.markdown("### 🚥 Signals")
    signal_cycle = st.slider("Cycle Time (sec)", 60, 240, 120, 10)
    green_ratio  = st.slider("Green Phase Ratio", 0.2, 0.7, 0.45, 0.05)
    signals_per_km = st.slider("Signals/km", 0.5, 5.0, 2.5, 0.5)

    st.markdown("---")

    # Warehouses
    st.markdown("### 🏭 Warehouses")
    with st.expander("➕ Add Warehouse"):
        wn = st.text_input("Name", "New Hub", key="wh_n")
        wlat = st.number_input("Lat", 18.8, 19.5, 19.1, 0.001, key="wh_lat")
        wlon = st.number_input("Lon", 72.7, 73.2, 72.85, 0.001, key="wh_lon")
        wcap = st.number_input("Capacity", 100, 5000, 1000, 100, key="wh_cap")
        wcos = st.number_input("Daily Cost (₹)", 100, 5000, 500, 50, key="wh_cos")
        if st.button("Add Warehouse", type="primary"):
            st.session_state.warehouses.append(
                Warehouse(wn, wlat, wlon, TrafficManager.get_zone(wn), wcap, wcos))
            st.rerun()
    for i, wh in enumerate(st.session_state.warehouses):
        c1, c2 = st.columns([3, 1])
        c1.markdown(f"🏭 **{wh.name}**")
        if c2.button("🗑️", key=f"del_wh_{i}"):
            st.session_state.warehouses.pop(i); st.rerun()

    st.markdown("---")

    # Vehicles
    st.markdown("### 🚛 Vehicles")
    with st.expander("➕ Add Vehicle"):
        vn = st.text_input("Name", "Van-New", key="v_n")
        vc = st.number_input("Capacity", 100, 2000, 400, 50, key="v_c")
        vcost = st.number_input("Cost/km (₹)", 1.0, 50.0, 12.0, 0.5, key="v_cost")
        vco2 = st.number_input("CO₂/km (kg)", 0.01, 1.0, 0.21, 0.01, key="v_co2")
        vspd = st.number_input("Speed (km/h)", 10.0, 80.0, 28.0, 1.0, key="v_spd")
        vfuel = st.selectbox("Fuel", ["Diesel", "CNG", "Electric", "Petrol", "Hybrid"], key="v_fuel")
        if st.button("Add Vehicle", type="primary"):
            st.session_state.vehicles.append(Vehicle(vn, vc, vcost, vco2, vspd, vfuel))
            st.rerun()
    for i, v in enumerate(st.session_state.vehicles):
        icon = {"Diesel":"⛽","CNG":"🟢","Electric":"⚡","Petrol":"🔴","Hybrid":"🔵"}.get(v.fuel_type,"🚛")
        c1, c2 = st.columns([3, 1])
        c1.markdown(f"{icon} **{v.name}** ({v.capacity})")
        if c2.button("🗑️", key=f"del_v_{i}"):
            st.session_state.vehicles.pop(i); st.rerun()

    st.markdown("---")

    # Delivery Points
    st.markdown("### 📦 Deliveries")
    with st.expander("➕ Add Delivery Point"):
        dpn = st.text_input("Name", "New Location", key="dp_n")
        dplat = st.number_input("Lat", 18.8, 19.5, 19.05, 0.001, key="dp_lat")
        dplon = st.number_input("Lon", 72.7, 73.2, 72.85, 0.001, key="dp_lon")
        dpd = st.number_input("Demand (units)", 10, 1000, 100, 10, key="dp_d")
        dpp = st.selectbox("Priority", [1, 2, 3],
            format_func=lambda x: {1:"Normal",2:"High",3:"Urgent"}[x], key="dp_p")
        if st.button("Add Delivery", type="primary"):
            st.session_state.deliveries.append(
                DeliveryPoint(dpn, dplat, dplon, TrafficManager.get_zone(dpn), dpd, (8,18), dpp))
            st.rerun()

    st.markdown(f"**{len(st.session_state.deliveries)} delivery points**")
    with st.expander("View / Remove"):
        for i, dp in enumerate(st.session_state.deliveries):
            badge = {1:"🟢",2:"🟡",3:"🔴"}[dp.priority]
            c1, c2 = st.columns([3,1])
            c1.markdown(f"{badge} **{dp.name}** (D:{dp.demand})")
            if c2.button("🗑️", key=f"del_dp_{i}"):
                st.session_state.deliveries.pop(i); st.rerun()

    st.markdown("---")

    # Save / Load
    st.markdown("### 💾 Save / Load")
    if st.button("📥 Export Problem JSON"):
        pd_data = {
            "warehouses": [{"name":w.name,"lat":w.lat,"lon":w.lon,"capacity":w.max_capacity,"cost":w.operating_cost} for w in st.session_state.warehouses],
            "deliveries": [{"name":d.name,"lat":d.lat,"lon":d.lon,"demand":d.demand,"priority":d.priority} for d in st.session_state.deliveries],
            "vehicles":   [{"name":v.name,"capacity":v.capacity,"cost_per_km":v.cost_per_km,"co2_per_km":v.co2_per_km,"speed":v.speed_kmh,"fuel":v.fuel_type} for v in st.session_state.vehicles],
        }
        st.download_button("⬇️ Download JSON", json.dumps(pd_data, indent=2),
                           "vrp_problem.json", "application/json")
    uploaded = st.file_uploader("📤 Load JSON", type=["json"])
    if uploaded:
        try:
            data = json.loads(uploaded.read())
            st.session_state.warehouses = [Warehouse(w["name"],w["lat"],w["lon"],TrafficManager.get_zone(w["name"]),w["capacity"],w.get("cost",500)) for w in data.get("warehouses",[])]
            st.session_state.deliveries = [DeliveryPoint(d["name"],d["lat"],d["lon"],TrafficManager.get_zone(d["name"]),d["demand"],(8,18),d.get("priority",1)) for d in data.get("deliveries",[])]
            st.session_state.vehicles   = [Vehicle(v["name"],v["capacity"],v["cost_per_km"],v["co2_per_km"],v.get("speed",28),v.get("fuel","Diesel")) for v in data.get("vehicles",[])]
            st.success("✅ Loaded!"); st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# ─── TRAFFIC CONFIG ────────────────────────────────────────
traffic_config = TrafficConfig(
    zone_multipliers=zone_multipliers,
    time_of_day=time_of_day,
    signal_cycle_time=signal_cycle,
    green_ratio=green_ratio,
    signals_per_km=signals_per_km,
)

# ─── SOLVE BUTTON ──────────────────────────────────────────
_, sc, _ = st.columns([1, 2, 1])
with sc:
    solve_btn = st.button("🚀 OPTIMIZE ROUTES", use_container_width=True, type="primary")

if solve_btn:
    with st.spinner("🧮 Solving VRP..."):
        wh, dl, vh = st.session_state.warehouses, st.session_state.deliveries, st.session_state.vehicles
        if not wh or not dl or not vh:
            st.error("❌ Need at least 1 warehouse, 1 delivery point, and 1 vehicle!")
        else:
            if "OR-Tools" in solver_choice:
                st.session_state.solution = ORToolsSolver.solve(wh, dl, vh, traffic_config, time_limit)
            elif "Heuristic" in solver_choice:
                st.session_state.solution = HeuristicSolver.solve(wh, dl, vh, traffic_config)
            else:
                st.session_state.solution = ORToolsSolver.solve(wh, dl, vh, traffic_config, time_limit)

            sc_key = "OR-Tools" if "OR-Tools" in solver_choice else "Heuristic"
            st.session_state.scenarios = ScenarioComparator.run_scenarios(
                wh, dl, vh, traffic_config, sc_key, min(time_limit, 15))
            st.success("✅ Optimization complete!")
            st.rerun()

# ─── RESULTS ───────────────────────────────────────────────
solution  = st.session_state.solution
scenarios = st.session_state.scenarios

if solution and solution.routes:
    # ── Metrics row ──
    st.markdown('<div class="section-header">📊 Optimization Summary</div>', unsafe_allow_html=True)
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    avg_util = np.mean([r.load_carried/r.capacity*100 for r in solution.routes]) if solution.routes else 0

    for col, label, value, color in [
        (m1, "Total Cost",       f"₹{solution.total_cost:,.0f}",      "#4A9EFF"),
        (m2, "Total Distance",   f"{solution.total_distance:,.1f} km","#FF8C42"),
        (m3, "CO₂ Emissions",    f"{solution.total_co2:,.2f} kg",     "#3DBA7E"),
        (m4, "Total Time",       f"{solution.total_time:,.0f} min",   "#e8c76d"),
        (m5, "Active Vehicles",  str(len(solution.routes)),           "#BB86FC"),
        (m6, "Avg Utilization",  f"{avg_util:.0f}%",                  "#FF6B9D"),
    ]:
        col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color};">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"<p style='text-align:center;color:#7a849a;margin-top:8px;'>Solver: <b>{solution.solver_used}</b></p>", unsafe_allow_html=True)
    if solution.unserved:
        st.warning(f"⚠️ Unserved: {', '.join(solution.unserved)}")

    # ── TABS ──
    tabs = st.tabs([
        "🗺️ Route Map",
        "📊 Route Analysis",
        "🌿 Emissions",
        "🚦 Traffic & Signals",
        "📈 Advanced Analytics",
        "📋 Delivery Plan",
        "⚡ Scenarios",
        "🎯 Clustering",
        "🤖 ML & AI",
        "🚚 P&D / Split",
        "👨‍✈️ Drivers",
        "🔄 Real-Time Ops",
        "📄 PDF Reports",
    ])

    all_locs = list(st.session_state.warehouses) + list(st.session_state.deliveries)
    names = [r.vehicle_name for r in solution.routes]
    colors = [VEHICLE_COLORS[i % len(VEHICLE_COLORS)] for i in range(len(solution.routes))]

    # ════════════════════════════════════════════════════════
    # TAB 0 — ROUTE MAP
    # ════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown('<div class="section-header">🗺️ Interactive Route Map</div>', unsafe_allow_html=True)
        map_style = st.radio("Map Style", ["Standard", "Satellite", "Dark"], horizontal=True)
        tile = {"Standard":"OpenStreetMap","Dark":"CartoDB dark_matter"}.get(map_style,"OpenStreetMap")

        clat = np.mean([l.lat for l in all_locs])
        clon = np.mean([l.lon for l in all_locs])
        m = folium.Map(location=[clat, clon], zoom_start=12,
                       tiles=tile if map_style != "Satellite" else None)
        if map_style == "Satellite":
            folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri").add_to(m)

        for wh in st.session_state.warehouses:
            folium.Marker([wh.lat, wh.lon],
                popup=folium.Popup(f"<b>🏭 {wh.name}</b><br>Cap:{wh.max_capacity}<br>Zone:{wh.zone}", max_width=200),
                icon=folium.Icon(color="blue", icon="industry", prefix="fa"),
                tooltip=f"🏭 {wh.name}").add_to(m)

        for dp in st.session_state.deliveries:
            pc = {1:"green",2:"orange",3:"red"}[dp.priority]
            folium.CircleMarker([dp.lat, dp.lon], radius=max(6,dp.demand/20),
                color=pc, fill=True, fillColor=pc, fillOpacity=0.7,
                popup=folium.Popup(f"<b>📦 {dp.name}</b><br>D:{dp.demand}<br>Pri:{dp.priority}", max_width=180),
                tooltip=f"📦 {dp.name} (D:{dp.demand})").add_to(m)

        for r_idx, route in enumerate(solution.routes):
            color = VEHICLE_COLORS[r_idx % len(VEHICLE_COLORS)]
            coords = []
            for loc_name in route.route_names:
                for loc in all_locs:
                    if loc.name == loc_name:
                        coords.append([loc.lat, loc.lon]); break
            if len(coords) > 1:
                AntPath(coords, color=color, weight=4, opacity=0.8,
                        dash_array=[10,20], delay=1000,
                        tooltip=f"{route.vehicle_name}").add_to(m)

        show_heat = st.checkbox("Show Traffic Heatmap", False)
        if show_heat:
            hd = []
            for loc in all_locs:
                z = TrafficManager.get_zone(loc.name)
                intensity = TrafficManager.MUMBAI_ZONES.get(z,{"base_traffic":1.0})["base_traffic"]
                tmult = TrafficManager.TIME_PRESETS.get(time_of_day,{"base":1.0})["base"]
                hd.append([loc.lat, loc.lon, intensity * zone_multipliers.get(z,1.0) * tmult])
            HeatMap(hd, radius=30, blur=20).add_to(m)

        st_folium(m, width=None, height=600, returned_objects=[])

        st.markdown("**Route Details:**")
        for r_idx, route in enumerate(solution.routes):
            color = VEHICLE_COLORS[r_idx % len(VEHICLE_COLORS)]
            st.markdown(f"""<div style='background:#1a1d2e;border-left:4px solid {color};
                padding:12px;border-radius:8px;margin:6px 0;'>
                <b style='color:{color};'>{route.vehicle_name}</b> | {' → '.join(route.route_names)} |
                📏 {route.total_distance:.1f} km | 💰 ₹{route.total_cost:,.0f} |
                🌿 {route.total_co2:.2f} kg | ⏱️ {route.total_time:.0f} min |
                📦 {route.load_carried}/{route.capacity} ({route.load_carried/route.capacity*100:.0f}%)
            </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # TAB 1 — ROUTE ANALYSIS
    # ════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown('<div class="section-header">📊 Route Analysis Dashboard</div>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=names, y=[r.total_cost for r in solution.routes],
                marker_color=colors, text=[f'₹{c:,.0f}' for c in [r.total_cost for r in solution.routes]], textposition='outside'))
            fig.update_layout(title="💰 Cost per Vehicle", xaxis_title="Vehicle", yaxis_title="Cost (₹)",
                template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=names, y=[r.total_distance for r in solution.routes],
                marker_color=colors, text=[f'{d:.1f}km' for d in [r.total_distance for r in solution.routes]], textposition='outside'))
            fig.update_layout(title="📏 Distance per Vehicle", xaxis_title="Vehicle", yaxis_title="km",
                template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
            st.plotly_chart(fig, use_container_width=True)

        c3,c4 = st.columns(2)
        with c3:
            fig = go.Figure(data=[go.Pie(labels=names, values=[r.load_carried for r in solution.routes],
                marker=dict(colors=colors), hole=0.4, textinfo='label+percent+value')])
            fig.update_layout(title="📦 Load Distribution", template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = go.Figure()
            for i, route in enumerate(solution.routes):
                util = route.load_carried / route.capacity * 100
                fig.add_trace(go.Bar(x=[route.vehicle_name], y=[util], marker_color=colors[i],
                    text=f'{util:.0f}%', textposition='outside', showlegend=False))
            fig.add_hline(y=80, line_dash="dash", line_color="#e8c76d", annotation_text="Target 80%")
            fig.update_layout(title="📊 Vehicle Utilization", yaxis_range=[0,110],
                template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
            st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        base_times = [r.total_time - r.signal_delays - r.traffic_delay for r in solution.routes]
        fig.add_trace(go.Bar(name='Base Travel', x=names, y=base_times, marker_color='#4A9EFF'))
        fig.add_trace(go.Bar(name='Traffic Delay', x=names, y=[r.traffic_delay for r in solution.routes], marker_color='#FF8C42'))
        fig.add_trace(go.Bar(name='Signal Delay', x=names, y=[r.signal_delays for r in solution.routes], marker_color='#FF5252'))
        fig.update_layout(title="⏱️ Time Breakdown (min)", barmode='stack',
            template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
        st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════
    # TAB 2 — EMISSIONS
    # ════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown('<div class="section-header">🌿 CO₂ Emissions Dashboard</div>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=names, y=[r.total_co2 for r in solution.routes],
                marker_color=colors, text=[f'{c:.2f}kg' for c in [r.total_co2 for r in solution.routes]], textposition='outside'))
            fig.update_layout(title="🌿 CO₂ per Vehicle", yaxis_title="CO₂ (kg)",
                template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fuel_co2 = {}
            for route in solution.routes:
                for v in st.session_state.vehicles:
                    if v.name == route.vehicle_name:
                        fuel_co2[v.fuel_type] = fuel_co2.get(v.fuel_type, 0) + route.total_co2; break
            fc = {"Diesel":"#FF5252","CNG":"#3DBA7E","Electric":"#4A9EFF","Petrol":"#FF8C42","Hybrid":"#BB86FC"}
            fig = go.Figure(data=[go.Pie(labels=list(fuel_co2.keys()), values=list(fuel_co2.values()),
                marker=dict(colors=[fc.get(f,'#999') for f in fuel_co2]), hole=0.45)])
            fig.update_layout(title="⛽ CO₂ by Fuel Type", template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        for i, route in enumerate(solution.routes):
            fig.add_trace(go.Scatter(x=[route.total_cost], y=[route.total_co2], mode='markers+text',
                marker=dict(size=max(15, route.load_carried/10), color=colors[i], opacity=0.8),
                text=[route.vehicle_name], textposition='top center', name=route.vehicle_name))
        fig.update_layout(title="💰 Cost vs CO₂ Pareto", xaxis_title="Cost (₹)", yaxis_title="CO₂ (kg)",
            template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
        st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════
    # TAB 3 — TRAFFIC & SIGNALS
    # ════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown('<div class="section-header">🚦 Traffic & Signal Analysis</div>', unsafe_allow_html=True)
        zones = list(TrafficManager.MUMBAI_ZONES.keys())
        tmult = TrafficManager.TIME_PRESETS.get(time_of_day,{"base":1.0})["base"]
        base_ts = [TrafficManager.MUMBAI_ZONES[z]["base_traffic"] * tmult * zone_multipliers.get(z,1.0) for z in zones]
        sig_dens = [TrafficManager.MUMBAI_ZONES[z]["signals_per_km"] for z in zones]

        c1,c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=zones, y=base_ts,
                marker_color=['#FF5252' if t>2 else '#FF8C42' if t>1.5 else '#3DBA7E' for t in base_ts],
                text=[f'{t:.2f}x' for t in base_ts], textposition='outside'))
            fig.add_hline(y=1.5, line_dash="dash", line_color="#e8c76d")
            fig.add_hline(y=2.0, line_dash="dash", line_color="#FF5252")
            fig.update_layout(title="🚗 Traffic Multiplier by Zone",
                template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=zones, y=sig_dens,
                marker_color=['#4A9EFF' if s<2.5 else '#FF8C42' if s<3 else '#FF5252' for s in sig_dens],
                text=[f'{s:.1f}/km' for s in sig_dens], textposition='outside'))
            fig.update_layout(title="🚥 Signal Density by Zone",
                template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
            st.plotly_chart(fig, use_container_width=True)

        # Heatmap
        time_periods = list(TrafficManager.TIME_PRESETS.keys())
        hm = [[round(TrafficManager.MUMBAI_ZONES[z]["base_traffic"] * TrafficManager.TIME_PRESETS[t]["base"] * zone_multipliers.get(z,1.0),2)
               for t in time_periods] for z in zones]
        fig = go.Figure(data=go.Heatmap(z=hm,
            x=[TrafficManager.TIME_PRESETS[t]["label"] for t in time_periods], y=zones,
            colorscale=[[0,'#1a1d2e'],[0.3,'#3DBA7E'],[0.6,'#FF8C42'],[1,'#FF5252']],
            text=[[f'{v:.2f}x' for v in row] for row in hm], texttemplate='%{text}'))
        fig.update_layout(title="🌡️ Traffic Intensity Heatmap", template="plotly_dark", height=420,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
        st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════
    # TAB 4 — ADVANCED ANALYTICS
    # ════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown('<div class="section-header">📈 Advanced Analytics</div>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            locs = list(st.session_state.warehouses) + list(st.session_state.deliveries)
            dm = build_distance_matrix(locs, traffic_config)
            loc_names = [l.name for l in locs]
            fig = go.Figure(data=go.Heatmap(z=dm, x=loc_names, y=loc_names,
                colorscale=[[0,'#0f1117'],[0.3,'#4A9EFF'],[0.7,'#FF8C42'],[1,'#FF5252']],
                text=np.round(dm,1), texttemplate='%{text}', textfont={"size":7}))
            fig.update_layout(title="📍 Distance Matrix Heatmap (km)",
                template="plotly_dark", height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if solution.shadow_prices:
                sp_names = list(solution.shadow_prices.keys())
                sp_vals  = list(solution.shadow_prices.values())
                si = np.argsort(sp_vals)[::-1]
                fig = go.Figure()
                fig.add_trace(go.Bar(x=[sp_vals[i] for i in si], y=[sp_names[i] for i in si],
                    orientation='h', marker_color=['#FF5252' if sp_vals[i] > np.median(sp_vals) else '#3DBA7E' for i in si]))
                fig.update_layout(title="💲 Shadow Prices (₹/unit)",
                    template="plotly_dark", height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
                st.plotly_chart(fig, use_container_width=True)

        # Sankey
        wh_list = st.session_state.warehouses
        dp_list  = st.session_state.deliveries
        s_labels = [w.name for w in wh_list] + [d.name for d in dp_list]
        dp_off = len(wh_list)
        sources, targets, values, link_cols = [], [], [], []
        for r_idx, route in enumerate(solution.routes):
            color = VEHICLE_COLORS[r_idx % len(VEHICLE_COLORS)]
            for stop_name in route.route_names[1:-1]:
                for di, dp in enumerate(dp_list):
                    if dp.name == stop_name:
                        sources.append(0); targets.append(dp_off + di)
                        values.append(dp.demand); link_cols.append(color); break
        if sources:
            fig = go.Figure(data=[go.Sankey(
                node=dict(pad=15, thickness=20,
                    label=s_labels,
                    color=['#4A9EFF']*len(wh_list)+['#3DBA7E']*len(dp_list)),
                link=dict(source=sources, target=targets, value=values,
                    color=[hex_to_rgba(c, 0.4) for c in link_cols]))])
            fig.update_layout(title="🔀 Warehouse → Delivery Flow (Sankey)",
                template="plotly_dark", height=500, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        # Radar
        categories = ['Distance','Cost','CO₂','Time','Utilization']
        fig = go.Figure()
        max_dist = max(r.total_distance for r in solution.routes) or 1
        max_cost = max(r.total_cost for r in solution.routes) or 1
        max_co2  = max(r.total_co2 for r in solution.routes) or 1
        max_time = max(r.total_time for r in solution.routes) or 1
        for i, route in enumerate(solution.routes):
            vals = [route.total_distance/max_dist*100, route.total_cost/max_cost*100,
                    route.total_co2/max_co2*100, route.total_time/max_time*100,
                    route.load_carried/route.capacity*100]
            vals.append(vals[0])
            fig.add_trace(go.Scatterpolar(r=vals, theta=categories+[categories[0]],
                fill='toself', name=route.vehicle_name,
                line_color=VEHICLE_COLORS[i % len(VEHICLE_COLORS)], opacity=0.6))
        fig.update_layout(title="🕸️ Vehicle Performance Radar",
            polar=dict(radialaxis=dict(visible=True, range=[0,110]), bgcolor='rgba(26,29,46,0.8)'),
            template="plotly_dark", height=500, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════
    # TAB 5 — DELIVERY PLAN
    # ════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown('<div class="section-header">📋 Delivery Plan</div>', unsafe_allow_html=True)
        plan_rows = []
        for route in solution.routes:
            cum_dist = cum_time = 0
            for k in range(1, len(route.route_indices)-1):
                seg = solution.distance_matrix[route.route_indices[k-1]][route.route_indices[k]]
                cum_dist += seg; cum_time += (seg/30)*60
                loc = next((d for d in st.session_state.deliveries if d.name == route.route_names[k]), None)
                if loc:
                    plan_rows.append({"Vehicle":route.vehicle_name,"Stop #":k,"Location":loc.name,
                        "Zone":loc.zone,"Demand":loc.demand,
                        "Priority":{1:"🟢 Normal",2:"🟡 High",3:"🔴 Urgent"}[loc.priority],
                        "Seg Dist (km)":round(seg,1),"Cum Dist (km)":round(cum_dist,1),
                        "Est Arrival (min)":round(cum_time,0),
                        "Time Window":f"{loc.time_window[0]}:00–{loc.time_window[1]}:00"})
        if plan_rows:
            df = pd.DataFrame(plan_rows)
            st.dataframe(df, use_container_width=True, hide_index=True, height=500)
            buf = io.StringIO(); df.to_csv(buf, index=False)
            st.download_button("📥 Download CSV", buf.getvalue(), "delivery_plan.csv", "text/csv")

        # Recommendations
        st.markdown("---")
        st.markdown('<div class="section-header">💡 Recommendations</div>', unsafe_allow_html=True)
        recs = []
        for route in solution.routes:
            util = route.load_carried/route.capacity*100
            if util < 50:
                recs.append(f"⚠️ **{route.vehicle_name}** underutilized ({util:.0f}%). Consider merging routes.")
        if solution.shadow_prices:
            exp = max(solution.shadow_prices, key=solution.shadow_prices.get)
            recs.append(f"📍 **{exp}** — highest shadow price (₹{solution.shadow_prices[exp]:.1f}/unit). Consider nearby warehouse.")
        for route in solution.routes:
            for v in st.session_state.vehicles:
                if v.name == route.vehicle_name and v.fuel_type == "Diesel":
                    recs.append(f"🌿 Replace **{route.vehicle_name}** (Diesel) with CNG/Electric to save ~{route.total_co2*0.6:.1f} kg CO₂."); break
        if not recs:
            recs = ["✅ Routes look optimally configured!"]
        for r in recs:
            st.markdown(f"- {r}")

    # ════════════════════════════════════════════════════════
    # TAB 6 — SCENARIOS
    # ════════════════════════════════════════════════════════
    with tabs[6]:
        st.markdown('<div class="section-header">⚡ Scenario Comparison</div>', unsafe_allow_html=True)
        if scenarios:
            sc_names = list(scenarios.keys())
            sc_colors = ["#4A9EFF","#FF8C42","#3DBA7E"]
            metrics_names = ["Cost (₹)","Distance (km)","CO₂ (kg)","Time (min)"]
            base_vals = [scenarios[sc_names[0]].total_cost, scenarios[sc_names[0]].total_distance,
                         scenarios[sc_names[0]].total_co2, scenarios[sc_names[0]].total_time]

            c1,c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=sc_names, y=[scenarios[s].total_cost for s in sc_names],
                    marker_color=sc_colors, text=[f'₹{scenarios[s].total_cost:,.0f}' for s in sc_names], textposition='outside'))
                fig.update_layout(title="💰 Cost Comparison", template="plotly_dark", height=380,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=sc_names, y=[scenarios[s].total_co2 for s in sc_names],
                    marker_color=sc_colors, text=[f'{scenarios[s].total_co2:.2f}kg' for s in sc_names], textposition='outside'))
                fig.update_layout(title="🌿 CO₂ Comparison", template="plotly_dark", height=380,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
                st.plotly_chart(fig, use_container_width=True)

            # Normalized comparison
            fig = go.Figure()
            for i, sn in enumerate(sc_names):
                sc = scenarios[sn]
                norm = [sc.total_cost/max(base_vals[0],1)*100, sc.total_distance/max(base_vals[1],0.1)*100,
                        sc.total_co2/max(base_vals[2],0.1)*100, sc.total_time/max(base_vals[3],0.1)*100]
                fig.add_trace(go.Bar(name=sn, x=metrics_names, y=norm, marker_color=sc_colors[i],
                    text=[f'{v:.0f}%' for v in norm], textposition='outside'))
            fig.add_hline(y=100, line_dash="dash", line_color="#e8c76d", annotation_text="Baseline 100%")
            fig.update_layout(title="📊 Normalized Comparison (% of Baseline)", barmode='group',
                yaxis_title="% of Baseline", template="plotly_dark", height=430,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
            st.plotly_chart(fig, use_container_width=True)

            comp_df = pd.DataFrame([{"Scenario":sn, "Cost (₹)":f"₹{scenarios[sn].total_cost:,.0f}",
                "Distance (km)":f"{scenarios[sn].total_distance:.1f}",
                "CO₂ (kg)":f"{scenarios[sn].total_co2:.2f}",
                "Time (min)":f"{scenarios[sn].total_time:.0f}",
                "Vehicles":len(scenarios[sn].routes)} for sn in sc_names])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════
    # TAB 7 — CLUSTERING
    # ════════════════════════════════════════════════════════
    with tabs[7]:
        st.markdown('<div class="section-header">🎯 K-Means Clustering Analysis</div>', unsafe_allow_html=True)
        dl = st.session_state.deliveries
        if len(dl) >= 3:
            cc1, cc2 = st.columns([2,1])
            with cc1:
                n_cl = st.slider("Number of Clusters", 2, min(8,len(dl)), min(4,len(dl)), key="n_cl")
                result_cl = DeliveryClusterer.cluster_deliveries(dl, n_cl)
                fig_cl = DeliveryClusterer.create_cluster_plot(dl, result_cl, st.session_state.warehouses)
                st.plotly_chart(fig_cl, use_container_width=True)
            with cc2:
                st.subheader("Cluster Summary")
                cl_rows = []
                for ci in range(n_cl):
                    info = result_cl["cluster_sizes"].get(ci, {})
                    cl_rows.append({
                        "Cluster": f"Cluster {ci}",
                        "Stops": info.get("count", 0),
                        "Total Demand": info.get("total_demand", 0),
                        "Members": ", ".join(info.get("members", [])),
                    })
                st.dataframe(pd.DataFrame(cl_rows), use_container_width=True, hide_index=True)

                st.subheader("Centroid Locations")
                for i, c in enumerate(result_cl["centroids"]):
                    st.write(f"**Cluster {i}:** ({c[0]:.4f}, {c[1]:.4f})")

                # Elbow — inertia per k
                st.subheader("Elbow Analysis")
                max_k = min(8, len(dl))
                ks = list(range(2, max_k+1))
                inertias = []
                for k in ks:
                    r = DeliveryClusterer.cluster_deliveries(dl, k)
                    inertias.append(r["inertia"])
                fig_el = go.Figure()
                fig_el.add_trace(go.Scatter(x=ks, y=inertias, mode='lines+markers',
                    line=dict(color='#4A9EFF', width=2), marker=dict(size=8)))
                fig_el.update_layout(title="📉 Elbow Curve", xaxis_title="k (clusters)",
                    yaxis_title="Inertia", template="plotly_dark", height=300,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
                st.plotly_chart(fig_el, use_container_width=True)
        else:
            st.warning("Add at least 3 delivery points for clustering analysis.")

    # ════════════════════════════════════════════════════════
    # TAB 8 — ML & AI
    # ════════════════════════════════════════════════════════
    with tabs[8]:
        st.markdown('<div class="section-header">🤖 ML & AI — Demand Prediction + Travel Time Forecasting + RL Dispatch</div>', unsafe_allow_html=True)

        ml_tabs = st.tabs(["📈 Demand Prediction", "🕐 Travel Time Forecasting", "🤖 RL Dispatch"])

        # ── Demand Prediction ──
        with ml_tabs[0]:
            st.markdown("### 📈 Demand Prediction (Polynomial Regression)")
            st.info("Synthetic historical data is auto-generated. In production, connect your order management system.")

            dl = st.session_state.deliveries
            if dl:
                # Train
                if st.session_state.demand_predictor is None or st.button("🔄 Retrain Model", key="retrain_dp"):
                    predictor = DemandPredictor(degree=2)
                    hist = DemandPredictor.generate_synthetic_history(dl)
                    predictor.train(hist)
                    st.session_state.demand_predictor = predictor
                    st.success("✅ Model trained on synthetic demand history.")

                predictor = st.session_state.demand_predictor
                if predictor:
                    dp_col1, dp_col2 = st.columns([1,2])
                    with dp_col1:
                        sel_loc = st.selectbox("Select Location", [d.name for d in dl], key="dp_sel")
                        pred_hour = st.slider("Prediction Hour", 6, 22, 14, key="dp_hour")
                        pred = predictor.predict(sel_loc, pred_hour)
                        st.metric("Predicted Demand", f"{pred.predicted_demand:.1f} units")
                        st.metric("Confidence", f"{pred.confidence:.0%}")
                        st.metric("Trend", pred.trend.capitalize())
                        st.metric("Historical Average", f"{pred.historical_average:.1f} units")
                    with dp_col2:
                        preds = predictor.predict_all_hours(sel_loc)
                        fig = DemandPredictor.create_demand_chart(preds, sel_loc)
                        st.plotly_chart(fig, use_container_width=True)

                    # All-location heatmap
                    st.markdown("#### 🌡️ Demand Heatmap (All Locations × Hour)")
                    hours_range = list(range(6, 22))
                    dm_data = []
                    for d in dl:
                        row = [predictor.predict(d.name, h).predicted_demand for h in hours_range]
                        dm_data.append(row)
                    fig_hm = go.Figure(data=go.Heatmap(
                        z=dm_data, x=hours_range, y=[d.name for d in dl],
                        colorscale=[[0,'#0f1117'],[0.4,'#4A9EFF'],[0.8,'#FF8C42'],[1,'#FF5252']],
                        text=[[f"{v:.0f}" for v in row] for row in dm_data],
                        texttemplate='%{text}'))
                    fig_hm.update_layout(title="📊 Predicted Demand Heatmap",
                        xaxis_title="Hour", yaxis_title="Location",
                        template="plotly_dark", height=450,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
                    st.plotly_chart(fig_hm, use_container_width=True)

        # ── Travel Time Forecasting ──
        with ml_tabs[1]:
            st.markdown("### 🕐 Travel Time Forecasting (Linear Regression)")
            wh = st.session_state.warehouses; dl = st.session_state.deliveries

            if st.session_state.travel_forecaster is None or st.button("🔄 Retrain Forecaster", key="retrain_tf"):
                forecaster = TravelTimeForecaster()
                synth_data = TravelTimeForecaster.generate_synthetic_data(wh, dl)
                forecaster.train(synth_data)
                st.session_state.travel_forecaster = forecaster
                st.success(f"✅ Trained on {len(synth_data)} origin-destination pairs.")

            forecaster = st.session_state.travel_forecaster
            if forecaster:
                tf1, tf2, tf3 = st.columns(3)
                origin      = tf1.selectbox("Origin", [w.name for w in wh] + [d.name for d in dl[:5]], key="tf_o")
                destination = tf2.selectbox("Destination", [d.name for d in dl], key="tf_d")
                tf_hour     = tf3.slider("Hour", 6, 22, 9, key="tf_h")

                traf_f  = st.slider("Traffic Factor", 0.5, 3.0, 1.5, 0.1, key="tf_tf")
                weath_f = st.slider("Weather Factor", 0.8, 1.5, 1.0, 0.05, key="tf_wf")

                fc = forecaster.predict(origin, destination, tf_hour, traf_f, weath_f)
                m1,m2,m3 = st.columns(3)
                m1.metric("Predicted Travel Time", f"{fc.predicted_time:.1f} min")
                m2.metric("Confidence", f"{fc.confidence:.0%}")
                m3.metric("Traffic × Weather", f"{traf_f:.1f}x × {weath_f:.2f}x")

                # Hourly forecast chart
                hourly = [forecaster.predict(origin, destination, h, traf_f, weath_f) for h in range(6, 23)]
                fig_tf = go.Figure()
                fig_tf.add_trace(go.Scatter(
                    x=[f.hour_of_day for f in hourly],
                    y=[f.predicted_time for f in hourly],
                    mode='lines+markers',
                    line=dict(color='#FF8C42', width=2),
                    fill='tozeroy', fillcolor='rgba(255,140,66,0.15)',
                    name='Predicted Time'))
                fig_tf.add_vline(x=tf_hour, line_dash="dash", line_color="#4A9EFF",
                                 annotation_text=f"Selected: {tf_hour}h")
                fig_tf.update_layout(
                    title=f"⏱️ Travel Time Forecast: {origin} → {destination}",
                    xaxis_title="Hour of Day", yaxis_title="Travel Time (min)",
                    template="plotly_dark", height=380,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,46,0.8)')
                st.plotly_chart(fig_tf, use_container_width=True)

        # ── RL Dispatch ──
        with ml_tabs[2]:
            st.markdown("### 🤖 Reinforcement Learning Dispatch (Q-Learning)")
            st.info("Q-Learning agent learns to assign deliveries to vehicles, maximizing utilization and minimizing cost.")

            rl_col1, rl_col2 = st.columns([1,2])
            with rl_col1:
                n_episodes = st.slider("Training Episodes", 1, 20, 5, key="rl_eps")
                if st.button("🚀 Run RL Dispatch", type="primary", key="run_rl"):
                    rl = RLDispatcher(lr=0.15, gamma=0.9, epsilon=0.15)
                    decisions = rl.dispatch_batch(
                        st.session_state.vehicles,
                        st.session_state.deliveries,
                        n_episodes)
                    st.session_state.rl_decisions = decisions
                    st.session_state["rl_agent"] = rl
                    st.success(f"✅ Dispatched {len(decisions)} deliveries. Q-table states: {len(rl.q_table)}")

            decisions = st.session_state.get("rl_decisions", [])
            rl_agent  = st.session_state.get("rl_agent", None)

            with rl_col2:
                if decisions:
                    rl_df = pd.DataFrame([{
                        "Delivery": d.assigned_delivery, "Vehicle": d.vehicle_id,
                        "Expected Reward": f"{d.expected_reward:.2f}",
                        "Confidence": f"{d.confidence_score:.0%}",
                        "Reason": d.decision_reason,
                    } for d in decisions])
                    st.dataframe(rl_df, use_container_width=True, hide_index=True, height=300)

            if decisions and rl_agent:
                fig_rl = rl_agent.visualize_rl_decisions(decisions)
                st.plotly_chart(fig_rl, use_container_width=True)
                fig_qt = rl_agent.visualize_q_table()
                st.plotly_chart(fig_qt, use_container_width=True)

    # ════════════════════════════════════════════════════════
    # TAB 9 — P&D / SPLIT DELIVERIES
    # ════════════════════════════════════════════════════════
    with tabs[9]:
        st.markdown('<div class="section-header">🚚 Pickup & Delivery + Split Deliveries</div>', unsafe_allow_html=True)
        pdp_tabs = st.tabs(["🔄 Pickup & Delivery", "📦 Split Deliveries"])

        with pdp_tabs[0]:
            st.markdown("### 🔄 Pickup & Delivery Problem (PDP)")

            with st.expander("➕ Add P&D Request"):
                pid   = st.text_input("Request ID", f"REQ-{len(st.session_state.pdp_requests)+1:03d}")
                p1,p2 = st.columns(2)
                p_loc = p1.text_input("Pickup Location", "Andheri Hub")
                d_loc = p2.text_input("Delivery Location", "Bandra")
                pc1,pc2 = st.columns(2)
                p_lat = pc1.number_input("Pickup Lat", 18.8, 19.5, 19.12, 0.001, key="pdp_plat")
                p_lon = pc2.number_input("Pickup Lon", 72.7, 73.2, 72.85, 0.001, key="pdp_plon")
                dc1,dc2 = st.columns(2)
                d_lat = dc1.number_input("Delivery Lat", 18.8, 19.5, 19.06, 0.001, key="pdp_dlat")
                d_lon = dc2.number_input("Delivery Lon", 72.7, 73.2, 72.83, 0.001, key="pdp_dlon")
                load = st.number_input("Load (units)", 10, 500, 100, 10, key="pdp_load")
                prio = st.selectbox("Priority", [1,2,3], format_func=lambda x:{1:"Normal",2:"High",3:"Urgent"}[x])
                if st.button("Add P&D Request", type="primary"):
                    st.session_state.pdp_requests.append(PickupDeliveryRequest(
                        pid, p_loc, p_lat, p_lon, d_loc, load, prio,
                        delivery_lat=d_lat, delivery_lon=d_lon))
                    st.success(f"✅ Request {pid} added.")

            reqs = st.session_state.pdp_requests
            if reqs:
                rdf = pd.DataFrame([{"ID":r.request_id, "Pickup":r.pickup_location,
                    "Delivery":r.delivery_location, "Load":r.load, "Priority":r.priority} for r in reqs])
                st.dataframe(rdf, use_container_width=True, hide_index=True)

                if st.button("🚀 Solve P&D Routes", type="primary"):
                    depots = [Depot(w.name, w.lat, w.lon, w.zone, w.max_capacity) for w in st.session_state.warehouses]
                    pdp_result = PickupDeliverySolver.solve(reqs, st.session_state.vehicles, depots)
                    st.success(f"✅ Served {pdp_result['served_requests']}/{pdp_result['total_requests']} requests.")
                    if pdp_result["unserved_requests"]:
                        st.warning(f"Unserved: {pdp_result['unserved_requests']}")
                    fig_pdp = PickupDeliverySolver.visualize_pdp_routes(reqs, pdp_result)
                    st.plotly_chart(fig_pdp, use_container_width=True)
            else:
                st.info("Add P&D requests above to enable pickup & delivery routing.")

        with pdp_tabs[1]:
            st.markdown("### 📦 Split Delivery Optimizer")
            caps = [v.capacity for v in st.session_state.vehicles]
            threshold = st.slider("Split Threshold (% of max capacity)", 50, 100, 80, 5, key="split_thr") / 100
            candidates = SplitDeliveryOptimizer.identify_split_candidates(
                st.session_state.deliveries, caps, threshold)

            if candidates:
                st.success(f"✅ {len(candidates)} deliveries require splitting.")
                c_rows = [{"Location":c.location_name, "Total Demand":c.total_demand,
                    "Number of Trips":len(c.splits),
                    "Split Quantities": " | ".join(f"{s['quantity']} units" for s in c.splits)}
                    for c in candidates]
                st.dataframe(pd.DataFrame(c_rows), use_container_width=True, hide_index=True)
                fig_sd = SplitDeliveryOptimizer.visualize_split_deliveries(candidates)
                st.plotly_chart(fig_sd, use_container_width=True)
            else:
                fig_sd = SplitDeliveryOptimizer.visualize_split_deliveries([])
                st.plotly_chart(fig_sd, use_container_width=True)
                st.success(f"✅ No deliveries exceed {threshold:.0%} of max capacity ({max(caps) if caps else 500} units). No splits needed.")

    # ════════════════════════════════════════════════════════
    # TAB 10 — DRIVERS
    # ════════════════════════════════════════════════════════
    with tabs[10]:
        st.markdown('<div class="section-header">👨‍✈️ Driver Assignment & Shift Management</div>', unsafe_allow_html=True)

        drv_tabs = st.tabs(["📋 Assignments", "🕐 Shift Timeline", "⭐ Performance", "⚠️ Validation"])

        with drv_tabs[0]:
            st.markdown("#### Current Driver Assignments")
            drv_df = pd.DataFrame([{
                "ID": d.driver_id, "Name": d.driver_name, "Vehicle": d.vehicle_name,
                "Shift": f"{d.shift_start:.0f}:00 – {d.shift_end:.0f}:00",
                "Max Hours": d.max_working_hours, "Phone": d.phone,
                "License": d.license_type, "Score": f"{d.performance_score:.0f}%",
            } for d in st.session_state.drivers])
            st.dataframe(drv_df, use_container_width=True, hide_index=True)

            st.markdown("#### ➕ Add Driver")
            d1,d2,d3 = st.columns(3)
            drv_id   = d1.text_input("Driver ID", f"DRV{len(st.session_state.drivers)+1:03d}", key="drv_id")
            drv_name = d2.text_input("Driver Name", "New Driver", key="drv_name")
            drv_veh  = d3.selectbox("Assign Vehicle", [v.name for v in st.session_state.vehicles], key="drv_veh")
            d4,d5,d6 = st.columns(3)
            drv_ss   = d4.number_input("Shift Start (h)", 0.0, 23.0, 8.0, 0.5, key="drv_ss")
            drv_se   = d5.number_input("Shift End (h)", 0.0, 24.0, 20.0, 0.5, key="drv_se")
            drv_ph   = d6.text_input("Phone", "98765XXXXX", key="drv_ph")
            drv_score = st.slider("Performance Score", 0, 100, 80, key="drv_sc")

            if st.button("➕ Add Driver", type="primary"):
                st.session_state.drivers.append(DriverAssignment(
                    drv_id, drv_name, drv_veh, drv_ss, drv_se, drv_ph, "LMV", "", drv_se-drv_ss, 0, float(drv_score)))
                st.success(f"✅ Driver {drv_name} added."); st.rerun()

            if st.button("🗑️ Remove Last Driver"):
                if st.session_state.drivers:
                    st.session_state.drivers.pop(); st.rerun()

        with drv_tabs[1]:
            fig_shift = DriverManager.create_shift_chart(st.session_state.drivers)
            st.plotly_chart(fig_shift, use_container_width=True)

            # Overtime check
            st.markdown("#### ⏰ Overtime Analysis")
            for d in st.session_state.drivers:
                ot = DriverManager.calculate_overtime(d, d.shift_end)
                if ot > 0:
                    st.warning(f"⚠️ {d.driver_name}: {ot:.1f}h potential overtime")
                else:
                    st.success(f"✅ {d.driver_name}: Within shift limits")

        with drv_tabs[2]:
            fig_perf = DriverManager.create_performance_chart(st.session_state.drivers)
            st.plotly_chart(fig_perf, use_container_width=True)

            st.markdown("#### 📊 Performance Details")
            perf_df = pd.DataFrame([{
                "Driver": d.driver_name,
                "Score": f"{d.performance_score:.0f}%",
                "Rating": "🌟 Excellent" if d.performance_score >= 90 else "✅ Good" if d.performance_score >= 75 else "⚠️ Needs Improvement",
                "Vehicle": d.vehicle_name,
            } for d in st.session_state.drivers])
            st.dataframe(perf_df, use_container_width=True, hide_index=True)

        with drv_tabs[3]:
            warnings = DriverManager.validate_assignments(st.session_state.drivers, st.session_state.vehicles)
            if warnings:
                for w in warnings:
                    if "⚠️" in w:
                        st.warning(w)
                    else:
                        st.info(w)
            else:
                st.success("✅ All driver assignments are valid!")

    # ════════════════════════════════════════════════════════
    # TAB 11 — REAL-TIME OPS
    # ════════════════════════════════════════════════════════
    with tabs[11]:
        st.markdown('<div class="section-header">🔄 Real-Time Re-optimization</div>', unsafe_allow_html=True)

        rt_col1, rt_col2 = st.columns([1,2])
        with rt_col1:
            st.markdown("#### Simulate Operational Changes")
            rt_traffic = st.slider("Current Traffic Factor", 0.5, 3.0, 1.0, 0.1, key="rt_trf")
            rt_orig_traffic = 1.0

            st.markdown("**Add New Urgent Delivery:**")
            rt_name  = st.text_input("Location Name", "Dharavi", key="rt_name")
            rt_lat   = st.number_input("Lat", 18.8, 19.5, 19.04, 0.001, key="rt_lat")
            rt_lon   = st.number_input("Lon", 72.7, 73.2, 72.86, 0.001, key="rt_lon")
            rt_dem   = st.number_input("Demand", 10, 500, 150, 10, key="rt_dem")

            new_deliveries = list(st.session_state.deliveries)
            if st.button("➕ Inject New Delivery", key="rt_inject"):
                new_dp = DeliveryPoint(rt_name, rt_lat, rt_lon, TrafficManager.get_zone(rt_name), rt_dem, (8,18), 3)
                new_deliveries.append(new_dp)
                st.success(f"✅ '{rt_name}' injected as urgent delivery.")

            if st.button("🔄 Check Re-optimization Need", type="primary", key="rt_check"):
                reopt = st.session_state.reoptimizer
                changes = reopt.detect_changes(new_deliveries, st.session_state.deliveries, rt_traffic, rt_orig_traffic)
                recommendation = reopt.get_recommendation(changes)

                st.markdown("#### Analysis Results")
                st.info(recommendation)

                fig_rt = reopt.visualize_changes(changes)
                st.plotly_chart(fig_rt, use_container_width=True)

                if reopt.should_reoptimize(changes):
                    st.warning("⚡ Re-optimization triggered! Click 🚀 OPTIMIZE ROUTES to update.")
                else:
                    st.success("✅ Current routes remain efficient. No re-optimization needed.")

        with rt_col2:
            st.markdown("#### Re-optimization History")
            hist = st.session_state.reoptimizer.history
            if hist:
                h_rows = [{"Time":h.get("timestamp","—"), "New":len(h["new_deliveries"]),
                    "Cancelled":len(h["cancelled_deliveries"]),
                    "Traffic Change":f"{h['traffic_change']:.2f}x",
                    "Total Changes":h["total_changes"]} for h in hist]
                st.dataframe(pd.DataFrame(h_rows), use_container_width=True, hide_index=True)
            else:
                st.info("Click 'Check Re-optimization Need' to generate history.")

            st.markdown("#### 📋 Active Route Snapshot")
            for route in solution.routes:
                st.markdown(f"""<div style='background:#1a1d2e;border-left:4px solid #4A9EFF;
                    padding:10px;border-radius:8px;margin:4px 0;font-size:0.85rem;'>
                    🚛 <b>{route.vehicle_name}</b> | {' → '.join(route.route_names[1:-1])} |
                    📏 {route.total_distance:.1f}km | ⏱️ {route.total_time:.0f}min
                </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # TAB 12 — PDF REPORTS
    # ════════════════════════════════════════════════════════
    with tabs[12]:
        st.markdown('<div class="section-header">📄 PDF Reports & Data Export</div>', unsafe_allow_html=True)
        pdf_tabs = st.tabs(["📄 PDF Route Sheets", "📊 All Routes Report", "📥 Data Export"])

        # ── Individual Route PDFs ──
        with pdf_tabs[0]:
            st.markdown("### 📄 Individual Driver Route Sheets")
            st.info("Each PDF contains: vehicle summary, stop sequence, arrival times, time window violations.")

            for r_idx, route in enumerate(solution.routes):
                color = VEHICLE_COLORS[r_idx % len(VEHICLE_COLORS)]
                c1, c2 = st.columns([3,1])
                c1.markdown(f"""<div style='background:#1a1d2e;border-left:4px solid {color};
                    padding:10px;border-radius:8px;'>
                    🚛 <b style='color:{color};'>{route.vehicle_name}</b> | {len(route.route_names)-2} stops |
                    📏 {route.total_distance:.1f}km | ⏱️ {route.total_time:.0f}min | 💰 ₹{route.total_cost:,.0f}
                </div>""", unsafe_allow_html=True)

                driver = next((d for d in st.session_state.drivers if d.vehicle_name == route.vehicle_name), None)
                pdf_bytes = PDFRouteExporter.generate_route_sheet(route, driver)
                if pdf_bytes:
                    c2.download_button(
                        f"📥 PDF",
                        data=pdf_bytes,
                        file_name=f"route_{route.vehicle_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        key=f"pdf_route_{r_idx}",
                    )
                else:
                    c2.warning("Install reportlab")

        # ── Full Report ──
        with pdf_tabs[1]:
            st.markdown("### 📊 Consolidated Routes Report")
            route_data_list = [{
                "vehicle_name": r.vehicle_name,
                "stops": r.route_names,
                "total_distance": r.total_distance,
                "total_time": r.total_time,
                "total_cost": r.total_cost,
                "total_co2": r.total_co2,
            } for r in solution.routes]

            if st.button("🖨️ Generate Full PDF Report", type="primary", key="gen_full_pdf"):
                with st.spinner("Generating PDF..."):
                    pdf_bytes = PDFRouteExporter.generate_pdf_report(
                        route_data=route_data_list,
                        warehouse_names=[w.name for w in st.session_state.warehouses],
                        delivery_names=[d.name for d in st.session_state.deliveries],
                        total_distance=solution.total_distance,
                        total_time=solution.total_time,
                        total_cost=solution.total_cost,
                        total_emissions=solution.total_co2,
                    )
                    if pdf_bytes:
                        st.download_button(
                            "📥 Download Full PDF Report",
                            data=pdf_bytes,
                            file_name=f"vrp_full_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                        )
                        st.success("✅ PDF report ready!")
                    else:
                        st.error("PDF generation failed. Ensure reportlab is installed: pip install reportlab")

        # ── Data Export ──
        with pdf_tabs[2]:
            st.markdown("### 📥 Export Data (JSON / CSV)")
            e1,e2 = st.columns(2)

            with e1:
                st.markdown("**📦 Routes JSON** — for API / integration")
                routes_json = json.dumps([{
                    "vehicle_name": r.vehicle_name,
                    "route": r.route_names,
                    "distance_km": r.total_distance,
                    "time_min": r.total_time,
                    "cost_inr": r.total_cost,
                    "co2_kg": r.total_co2,
                    "load_carried": r.load_carried,
                    "capacity": r.capacity,
                    "utilization_pct": round(r.load_carried/r.capacity*100, 1),
                } for r in solution.routes], indent=2)
                st.download_button("📥 Download Routes JSON", routes_json,
                    f"routes_{datetime.now().strftime('%Y%m%d_%H%M')}.json", "application/json")

            with e2:
                st.markdown("**📋 Stops CSV** — for Excel analysis")
                csv_rows = []
                for route in solution.routes:
                    for i, stop in enumerate(route.route_names):
                        arr = route.arrival_times[i] if i < len(route.arrival_times) else 0
                        wait = route.wait_times[i] if i < len(route.wait_times) else 0
                        csv_rows.append({
                            "Vehicle": route.vehicle_name,
                            "Stop_Seq": i,
                            "Location": stop,
                            "Type": "Depot" if i == 0 or i == len(route.route_names)-1 else "Delivery",
                            "Arrival_Hour": round(arr, 2),
                            "Wait_Min": round(wait, 1),
                        })
                csv_buf = io.StringIO()
                pd.DataFrame(csv_rows).to_csv(csv_buf, index=False)
                st.download_button("📥 Download Stops CSV", csv_buf.getvalue(),
                    f"stops_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

                st.markdown("**📈 Solution Summary JSON**")
                summary_json = json.dumps({
                    "total_cost_inr": solution.total_cost,
                    "total_distance_km": solution.total_distance,
                    "total_co2_kg": solution.total_co2,
                    "total_time_min": solution.total_time,
                    "active_vehicles": len(solution.routes),
                    "unserved": solution.unserved,
                    "solver": solution.solver_used,
                    "generated_at": datetime.now().isoformat(),
                }, indent=2)
                st.download_button("📥 Download Summary JSON", summary_json,
                    f"summary_{datetime.now().strftime('%Y%m%d_%H%M')}.json", "application/json")

else:
    # ── No solution — Getting Started ──
    st.markdown(f"""
    <div style='text-align:center;padding:60px 20px;'>
        <h2 style='color:#e2e6f0;'>🚛 Ready to Optimize!</h2>
        <p style='color:#7a849a;margin:16px 0 32px 0;'>
            Configure warehouses, vehicles, and delivery points in the sidebar,<br>
            then click <b>🚀 OPTIMIZE ROUTES</b> to solve the Vehicle Routing Problem.
        </p>
        <div style='display:flex;justify-content:center;gap:40px;flex-wrap:wrap;'>
            {''.join(f"""<div style='background:#1a1d2e;border:1px solid #2d3348;border-radius:12px;padding:20px 30px;text-align:center;'>
                <div style='font-size:2rem;margin-bottom:8px;'>{icon}</div>
                <div style='font-size:1.4rem;font-weight:700;color:#4A9EFF;'>{count}</div>
                <div style='color:#7a849a;font-size:0.85rem;'>{label}</div>
            </div>""" for icon, count, label in [
                ("🏭", len(st.session_state.warehouses), "Warehouses"),
                ("📦", len(st.session_state.deliveries), "Delivery Points"),
                ("🚛", len(st.session_state.vehicles), "Vehicles"),
                ("👨‍✈️", len(st.session_state.drivers), "Drivers"),
            ])}
        </div>
        <div style='margin-top:40px;color:#7a849a;font-size:0.9rem;'>
            <b>Features:</b> OR-Tools CVRP • Traffic Intelligence • ML Demand Prediction •
            RL Dispatch • Pickup & Delivery • Split Deliveries • Driver Management • PDF Export
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#7a849a;font-size:.85rem;'>
🚛 Urban Logistics VRP Optimizer — MPSTME NMIMS Mumbai<br>
OR-Tools • Plotly • Folium • scikit-learn • ReportLab | Vehicle Routing Problem with ML & Signal Intelligence
</div>
""", unsafe_allow_html=True)
