# 🚛 Urban Logistics VRP Optimizer

### MPSTME NMIMS Mumbai — Vehicle Routing Problem with Traffic & Signal Intelligence

A **full-stack production-grade** Streamlit application for solving the **Capacitated Vehicle Routing Problem (CVRP)** with real-time traffic integration, signal management, CO₂ tracking, and comprehensive analytics. Built for Mumbai urban logistics.

---

## 🚀 Features

### Core Optimization
- **Dual Solver Engine**: Google OR-Tools (CVRP + GLS) and Clarke-Wright Savings + 2-opt Heuristic
- **Dynamic Problem Configuration**: Add/remove warehouses, vehicles, delivery points on-the-fly
- **Multi-Vehicle Support**: Diesel, CNG, Electric, Hybrid with unique cost/emission profiles
- **Capacity Constraints**: Automatic vehicle capacity enforcement
- **Save/Load Problems**: Export and import problem configurations as JSON

### Traffic & Signal Intelligence
- **6 Mumbai Traffic Zones**: South Mumbai, Central, Western/Eastern Suburbs, Extended, Navi Mumbai
- **Manual Traffic Intensity Control**: Per-zone sliders (0.5x - 3.0x)
- **Time-of-Day Presets**: Morning rush, midday, evening rush, night, off-peak
- **Signal Cycle Optimization**: Green wave analysis per route
- **Signal Density Modeling**: Zone-aware traffic signal delays

### Visualizations (7 Dashboard Tabs)
1. 🗺️ **Interactive Route Map** — Folium with animated routes, traffic heatmap, 3 map styles
2. 📊 **Route Analysis** — Cost/distance bars, load pie, utilization gauges, time breakdown
3. 🌿 **Emissions Dashboard** — CO₂ per vehicle, fuel-type breakdown, Pareto front, intensity
4. 🚦 **Traffic & Signals** — Zone traffic heatmap, signal density, green wave optimization
5. 📈 **Advanced Analytics** — Distance heatmap, shadow prices, Sankey flow, radar charts
6. 📋 **Delivery Plan** — Full schedule table with arrival times, CSV export, AI recommendations
7. ⚡ **Scenario Comparison** — Baseline vs Traffic vs Emission Cap (normalized metrics)

### 15+ Chart Types
Bar, Pie, Heatmap, Scatter (Pareto), Line, Sankey, Radar/Spider, Stacked Bar, Gauge, Grouped Bar, Animated Map, Traffic Heatmap overlay, and more.

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

## ▶️ Run Locally

```bash
streamlit run app.py
```

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Deploy!

---

## 🏗️ Project Structure

```
vrp_project/
├── app.py                  # Main Streamlit application (UI + charts)
├── vrp_engine.py           # VRP solver engine + traffic + signals
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
└── README.md               # This file
```

---

## 🧮 Technical Details

### Solver Algorithms
- **OR-Tools**: Google's constraint programming solver with PATH_CHEAPEST_ARC first solution + GUIDED_LOCAL_SEARCH metaheuristic
- **Heuristic**: Clarke-Wright Savings algorithm for route merging + 2-opt local search improvement

### Distance Calculation
- Haversine formula × 1.35 road factor for Mumbai road network approximation
- Traffic multiplier adjusts effective distance based on congestion

### CO₂ Model
```
CO₂ = distance × co2_per_km × load_factor
load_factor = 1.0 + 0.15 × (load / capacity)
```

### Signal Delay Model
```
delay = num_signals × avg_wait_time
num_signals = distance × signals_per_km
avg_wait = cycle_time × (1 - green_ratio) × 0.5
```

---

## 🗺️ Mumbai Coverage

**15 delivery zones** across Mumbai:
Bandra, Worli, Colaba, Kurla, Malad, Borivali, Powai, Juhu, Lower Parel, Chembur, Ghatkopar, Thane, Vashi, Goregaon, Santacruz

**2 default warehouses**: Andheri Hub, Dadar Central

**5 default vehicles**: 2× Diesel, 2× CNG, 1× Electric

---

## 📝 License

MIT — MPSTME NMIMS Mumbai Research Project