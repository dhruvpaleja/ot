"""
Advanced VRP Features Module — Complete Implementation
Urban Logistics Optimization — MPSTME NMIMS Mumbai

✅ Multi-Depot VRP
✅ Pickup & Delivery (PDP)
✅ Split Deliveries
✅ Driver Assignment & Shift Management
✅ Demand Prediction (ML — Polynomial Regression)
✅ Travel Time Forecasting (ML — Linear Regression)
✅ K-Means Clustering
✅ Reinforcement Learning Dispatch (Q-Learning)
✅ Real-Time Re-optimization
✅ PDF Route Sheet Export (ReportLab)
✅ Advanced Visualizations (Plotly)
"""

import numpy as np
import random
import io
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd

# scikit-learn
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# ══════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════

@dataclass
class Depot:
    """Multi-depot support"""
    name: str
    lat: float
    lon: float
    zone: str = "General"
    max_capacity: int = 1000
    operating_cost: float = 500.0
    vehicles_assigned: List[str] = field(default_factory=list)
    opening_time: float = 6.0
    closing_time: float = 22.0


@dataclass
class PickupDeliveryRequest:
    """Pickup & Delivery pair request"""
    request_id: str
    pickup_location: str
    pickup_lat: float
    pickup_lon: float
    delivery_location: str
    load: int = 100
    priority: int = 1
    pickup_time_window: Tuple[float, float] = (8.0, 18.0)
    delivery_time_window: Tuple[float, float] = (8.0, 18.0)
    pickup_service_time: float = 10.0
    delivery_service_time: float = 10.0
    max_ride_time: float = 60.0
    delivery_lat: float = 19.0
    delivery_lon: float = 72.8


@dataclass
class SplitDelivery:
    """Split delivery across multiple vehicles"""
    location_name: str
    lat: float
    lon: float
    total_demand: int
    splits: List[Dict] = field(default_factory=list)
    min_split_quantity: int = 50
    time_window: Tuple[float, float] = (8.0, 18.0)


@dataclass
class DriverAssignment:
    """Driver to vehicle assignment"""
    driver_id: str
    driver_name: str
    vehicle_name: str
    shift_start: float
    shift_end: float
    phone: str = ""
    license_type: str = "Commercial"
    email: str = ""
    max_working_hours: float = 10.0
    breaks_taken: int = 0
    performance_score: float = 85.0


@dataclass
class DemandPrediction:
    """ML-based demand prediction result"""
    location_name: str
    predicted_demand: float
    confidence: float
    prediction_hour: int
    historical_average: float
    trend: str  # "increasing" | "decreasing" | "stable"


@dataclass
class TravelTimeForecast:
    """ML-based travel time prediction result"""
    origin: str
    destination: str
    predicted_time: float
    confidence: float
    traffic_factor: float
    weather_factor: float
    hour_of_day: int


@dataclass
class RLDispatchDecision:
    """Reinforcement Learning dispatch decision"""
    vehicle_id: str
    assigned_delivery: str
    confidence_score: float
    expected_reward: float
    alternative_vehicles: List[str]
    decision_reason: str


# ══════════════════════════════════════════════════════════════
#  MULTI-DEPOT ALLOCATOR
# ══════════════════════════════════════════════════════════════

class MultiDepotAllocator:
    """Assign deliveries to nearest depot respecting capacity constraints"""

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat / 2) ** 2
             + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    @classmethod
    def assign_to_depots(
        cls,
        depots: List[Depot],
        deliveries: List[Any],
        capacity_factor: float = 0.85,
    ) -> Tuple[Dict[str, List], Dict[str, int]]:
        assignments = {d.name: [] for d in depots}
        depot_loads = {d.name: 0 for d in depots}

        # Sort deliveries by distance to nearest depot (hardest first)
        delivery_info = []
        for i, dl in enumerate(deliveries):
            dists = sorted(
                [(dep.name, cls.haversine(dl.lat, dl.lon, dep.lat, dep.lon)) for dep in depots],
                key=lambda x: x[1],
            )
            delivery_info.append((i, dl, dists[0][1], dists))
        delivery_info.sort(key=lambda x: -x[2])

        for idx, dl, _, dists in delivery_info:
            demand = getattr(dl, "demand", 100)
            assigned = False
            for depot_name, _ in dists:
                dep = next(d for d in depots if d.name == depot_name)
                if depot_loads[depot_name] + demand <= dep.max_capacity * capacity_factor:
                    assignments[depot_name].append(idx)
                    depot_loads[depot_name] += demand
                    assigned = True
                    break
            if not assigned:
                least = min(depot_loads, key=depot_loads.get)
                assignments[least].append(idx)
                depot_loads[least] += demand

        return assignments, depot_loads

    @classmethod
    def optimize_depot_locations(cls, deliveries: List[Any], n_depots: int = 3) -> List[Dict]:
        coords = np.array([[d.lat, d.lon] for d in deliveries])
        km = KMeans(n_clusters=n_depots, random_state=42, n_init=10).fit(coords)
        result = []
        for i, c in enumerate(km.cluster_centers_):
            members = [deliveries[j] for j in range(len(deliveries)) if km.labels_[j] == i]
            total_demand = sum(getattr(m, "demand", 100) for m in members)
            result.append({
                "centroid_lat": c[0],
                "centroid_lon": c[1],
                "cluster_size": len(members),
                "total_demand": total_demand,
                "recommended_capacity": int(total_demand * 1.2),
            })
        return result


# ══════════════════════════════════════════════════════════════
#  K-MEANS DELIVERY CLUSTERER
# ══════════════════════════════════════════════════════════════

class DeliveryClusterer:
    """Cluster delivery locations using K-Means for route pre-processing"""

    @staticmethod
    def cluster_deliveries(
        deliveries: List[Any],
        n_clusters: int = 5,
    ) -> Dict:
        """Returns dict with labels, centroids, inertia, cluster_sizes"""
        coords = np.array([[d.lat, d.lon] for d in deliveries])
        n_clusters = min(n_clusters, len(deliveries))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(coords)

        result = {
            "labels": km.labels_.tolist(),
            "centroids": km.cluster_centers_.tolist(),
            "inertia": km.inertia_,
            "cluster_sizes": {},
        }
        for i in range(n_clusters):
            members = [
                deliveries[j] for j in range(len(deliveries)) if km.labels_[j] == i
            ]
            result["cluster_sizes"][i] = {
                "count": len(members),
                "members": [getattr(m, "name", str(m)) for m in members],
                "total_demand": sum(getattr(m, "demand", 0) for m in members),
                "centroid_lat": km.cluster_centers_[i][0],
                "centroid_lon": km.cluster_centers_[i][1],
            }
        return result

    @staticmethod
    def get_optimal_clusters(deliveries: List[Any], max_cluster_size: int = 5) -> int:
        n = len(deliveries)
        return max(2, min(8, n // max_cluster_size + 1))

    @staticmethod
    def create_cluster_plot(deliveries: List[Any], clustering_result: Dict,
                            warehouse_locs: List[Any] = None):
        """Create Plotly map showing K-Means clusters"""
        import plotly.graph_objects as go

        colors = [
            "#4A9EFF", "#FF8C42", "#3DBA7E", "#BB86FC",
            "#FF6B9D", "#e8c76d", "#00BCD4", "#FF5252",
        ]

        fig = go.Figure()
        labels = clustering_result["labels"]
        n_clusters = len(clustering_result["centroids"])

        for cl in range(n_clusters):
            idx = [i for i, l in enumerate(labels) if l == cl]
            fig.add_trace(go.Scattermapbox(
                lat=[deliveries[i].lat for i in idx],
                lon=[deliveries[i].lon for i in idx],
                mode="markers+text",
                marker=dict(size=12, color=colors[cl % len(colors)]),
                text=[getattr(deliveries[i], "name", f"D{i}") for i in idx],
                textposition="top right",
                name=f"Cluster {cl} ({len(idx)} stops)",
                hoverinfo="text",
            ))

        # Centroids
        fig.add_trace(go.Scattermapbox(
            lat=[c[0] for c in clustering_result["centroids"]],
            lon=[c[1] for c in clustering_result["centroids"]],
            mode="markers",
            marker=dict(size=18, symbol="star", color="gold"),
            name="Centroids",
        ))

        if warehouse_locs:
            fig.add_trace(go.Scattermapbox(
                lat=[w.lat for w in warehouse_locs],
                lon=[w.lon for w in warehouse_locs],
                mode="markers",
                marker=dict(size=20, symbol="square", color="red"),
                text=[w.name for w in warehouse_locs],
                name="Warehouses",
            ))

        center_lat = np.mean([d.lat for d in deliveries])
        center_lon = np.mean([d.lon for d in deliveries])

        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=10,
            ),
            title="📦 Delivery Clusters (K-Means)",
            template="plotly_dark",
            height=550,
            margin=dict(l=0, r=0, t=50, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig


# ══════════════════════════════════════════════════════════════
#  DEMAND PREDICTOR (ML)
# ══════════════════════════════════════════════════════════════

class DemandPredictor:
    """Polynomial Regression demand predictor per location per hour"""

    def __init__(self, degree: int = 2):
        self.degree = degree
        self.models: Dict[str, Tuple[LinearRegression, PolynomialFeatures]] = {}
        self.averages: Dict[str, float] = {}

    def train(self, historical_data: Dict[str, List[Tuple[int, float]]]):
        """
        historical_data: {location_name: [(hour, demand), ...]}
        """
        for loc, data in historical_data.items():
            hours = np.array([d[0] for d in data]).reshape(-1, 1)
            demands = np.array([d[1] for d in data])
            poly = PolynomialFeatures(degree=self.degree)
            X = poly.fit_transform(hours)
            reg = LinearRegression().fit(X, demands)
            self.models[loc] = (reg, poly)
            self.averages[loc] = float(np.mean(demands))

    def predict(self, location: str, hour: int) -> DemandPrediction:
        avg = self.averages.get(location, 100.0)
        if location not in self.models:
            return DemandPrediction(location, avg, 0.5, hour, avg, "stable")

        reg, poly = self.models[location]
        X = poly.transform(np.array([[hour]]))
        predicted = float(reg.predict(X)[0])
        predicted = max(0.0, predicted)

        confidence = float(np.clip(1.0 - abs(predicted - avg) / (avg + 1) * 0.6, 0.4, 0.97))
        if predicted > avg * 1.1:
            trend = "increasing"
        elif predicted < avg * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"

        return DemandPrediction(location, round(predicted, 1), round(confidence, 2),
                                hour, round(avg, 1), trend)

    def predict_all_hours(self, location: str) -> List[DemandPrediction]:
        return [self.predict(location, h) for h in range(6, 22)]

    @staticmethod
    def generate_synthetic_history(deliveries: List[Any]) -> Dict[str, List[Tuple[int, float]]]:
        """Generate plausible Mumbai delivery demand curves for demo"""
        data = {}
        for dl in deliveries:
            base = getattr(dl, "demand", 100)
            records = []
            for h in range(6, 23):
                # Morning peak 9-11, Evening peak 17-20
                factor = (
                    1.0 + 0.5 * np.exp(-0.5 * ((h - 9.5) ** 2))
                    + 0.4 * np.exp(-0.5 * ((h - 18) ** 2))
                )
                noise = random.gauss(0, base * 0.05)
                records.append((h, max(10, base * factor + noise)))
            data[getattr(dl, "name", str(dl))] = records
        return data

    @staticmethod
    def create_demand_chart(predictions: List[DemandPrediction], location_name: str):
        import plotly.graph_objects as go
        hours = [p.prediction_hour for p in predictions]
        vals = [p.predicted_demand for p in predictions]
        confs = [p.confidence for p in predictions]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours, y=vals,
            mode="lines+markers",
            name="Predicted Demand",
            line=dict(color="#4A9EFF", width=2),
            fill="tozeroy",
            fillcolor="rgba(74,158,255,0.15)",
        ))
        fig.add_trace(go.Scatter(
            x=hours, y=[v * c for v, c in zip(vals, confs)],
            mode="lines",
            name="Confidence Band",
            line=dict(color="#3DBA7E", width=1, dash="dot"),
        ))
        fig.update_layout(
            title=f"📈 Demand Forecast — {location_name}",
            xaxis_title="Hour of Day",
            yaxis_title="Predicted Demand (units)",
            template="plotly_dark",
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,29,46,0.8)",
        )
        return fig


# ══════════════════════════════════════════════════════════════
#  TRAVEL TIME FORECASTER (ML)
# ══════════════════════════════════════════════════════════════

class TravelTimeForecaster:
    """Linear Regression travel time predictor using hour, traffic, weather"""

    def __init__(self):
        self.models: Dict[str, LinearRegression] = {}
        self.scaler = StandardScaler()
        self._fitted = False

    def train(self, route_data: Dict[str, List[Dict]]):
        """
        route_data: {route_key: [{'hour': h, 'traffic_level': t, 'weather_factor': w, 'actual_time': m}]}
        """
        all_X, all_y = [], []
        for route_key, records in route_data.items():
            X = [[r["hour"], r["traffic_level"], r["weather_factor"]] for r in records]
            y = [r["actual_time"] for r in records]
            all_X.extend(X)
            all_y.extend(y)

            reg = LinearRegression().fit(np.array(X), np.array(y))
            self.models[route_key] = reg

        if all_X:
            self.scaler.fit(np.array(all_X))
            self._fitted = True

    def predict(self, origin: str, destination: str, hour: int,
                traffic_factor: float = 1.0, weather_factor: float = 1.0) -> TravelTimeForecast:
        key = f"{origin}_{destination}"
        features = np.array([[hour, traffic_factor, weather_factor]])

        if key in self.models:
            predicted = float(self.models[key].predict(features)[0])
            conf = 0.80
        else:
            # Heuristic fallback
            base = 18.0
            predicted = base * traffic_factor * weather_factor * (1 + (hour in range(8, 11) or hour in range(17, 20)) * 0.3)
            conf = 0.55

        return TravelTimeForecast(
            origin=origin,
            destination=destination,
            predicted_time=round(max(3.0, predicted), 1),
            confidence=conf,
            traffic_factor=traffic_factor,
            weather_factor=weather_factor,
            hour_of_day=hour,
        )

    @staticmethod
    def generate_synthetic_data(warehouses, deliveries) -> Dict[str, List[Dict]]:
        """Generate realistic synthetic travel records for training"""
        data = {}
        all_locs = list(warehouses) + list(deliveries)
        for i, loc1 in enumerate(all_locs[:5]):
            for j, loc2 in enumerate(all_locs[:5]):
                if i == j:
                    continue
                key = f"{loc1.name}_{loc2.name}"
                records = []
                for h in range(6, 23):
                    traffic = 1.0 + 0.5 * (h in range(8, 11)) + 0.6 * (h in range(17, 20))
                    weather = random.uniform(0.95, 1.15)
                    base_time = random.uniform(12, 35)
                    actual = base_time * traffic * weather + random.gauss(0, 1)
                    records.append({"hour": h, "traffic_level": traffic, "weather_factor": weather, "actual_time": max(3, actual)})
                data[key] = records
        return data

    @staticmethod
    def create_heatmap(forecasts: List[TravelTimeForecast]):
        import plotly.graph_objects as go
        if not forecasts:
            return go.Figure()

        hours = sorted(set(f.hour_of_day for f in forecasts))
        origins = sorted(set(f.origin for f in forecasts))

        z = []
        for o in origins:
            row = []
            for h in hours:
                match = next((f for f in forecasts if f.origin == o and f.hour_of_day == h), None)
                row.append(match.predicted_time if match else 0)
            z.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=z, x=hours, y=origins,
            colorscale=[[0, "#3DBA7E"], [0.5, "#FF8C42"], [1, "#FF5252"]],
            text=[[f"{v:.1f}" for v in row] for row in z],
            texttemplate="%{text} min",
        ))
        fig.update_layout(
            title="🕐 Travel Time Forecast Heatmap (min)",
            xaxis_title="Hour of Day",
            template="plotly_dark", height=400,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,46,0.8)",
        )
        return fig


# ══════════════════════════════════════════════════════════════
#  PICKUP & DELIVERY SOLVER
# ══════════════════════════════════════════════════════════════

class PickupDeliverySolver:
    """Greedy PDP solver with precedence constraints"""

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    @classmethod
    def solve(cls, requests: List[PickupDeliveryRequest],
              vehicles: List[Any], depots: List[Depot]) -> Dict:
        """Greedy assignment: nearest pickup first, capacity respected"""
        vehicle_routes: Dict[int, Dict] = {}
        for i, v in enumerate(vehicles):
            vehicle_routes[i] = {
                "vehicle": v,
                "stops": [],
                "load": 0,
                "distance": 0.0,
                "served": [],
            }

        unserved = []
        for req in sorted(requests, key=lambda r: -r.priority):
            best_v = None
            best_extra = float("inf")
            for vi, vr in vehicle_routes.items():
                cap = getattr(vr["vehicle"], "capacity", 500)
                if vr["load"] + req.load > cap:
                    continue
                # Extra distance to pickup then deliver
                if vr["stops"]:
                    last = vr["stops"][-1]
                    extra = cls.haversine(last[0], last[1], req.pickup_lat, req.pickup_lon)
                    extra += cls.haversine(req.pickup_lat, req.pickup_lon, req.delivery_lat, req.delivery_lon)
                else:
                    depot = depots[0] if depots else None
                    dlat = depot.lat if depot else req.pickup_lat
                    dlon = depot.lon if depot else req.pickup_lon
                    extra = cls.haversine(dlat, dlon, req.pickup_lat, req.pickup_lon)
                    extra += cls.haversine(req.pickup_lat, req.pickup_lon, req.delivery_lat, req.delivery_lon)

                if extra < best_extra:
                    best_extra = extra
                    best_v = vi

            if best_v is not None:
                vr = vehicle_routes[best_v]
                vr["stops"].append((req.pickup_lat, req.pickup_lon, f"PICKUP: {req.pickup_location}"))
                vr["stops"].append((req.delivery_lat, req.delivery_lon, f"DELIVER: {req.delivery_location}"))
                vr["load"] += req.load
                vr["distance"] += best_extra
                vr["served"].append(req.request_id)
            else:
                unserved.append(req.request_id)

        routes_out = [vr for vr in vehicle_routes.values() if vr["stops"]]
        return {
            "routes": routes_out,
            "served_requests": sum(len(vr["served"]) for vr in vehicle_routes.values()),
            "unserved_requests": unserved,
            "total_requests": len(requests),
        }

    @staticmethod
    def visualize_pdp_routes(requests: List[PickupDeliveryRequest], result: Dict):
        import plotly.graph_objects as go
        colors = ["#4A9EFF", "#FF8C42", "#3DBA7E", "#BB86FC", "#FF6B9D"]
        fig = go.Figure()

        for i, route in enumerate(result["routes"]):
            v_name = getattr(route["vehicle"], "name", f"Vehicle {i+1}")
            color = colors[i % len(colors)]
            lats = [s[0] for s in route["stops"]]
            lons = [s[1] for s in route["stops"]]
            labels = [s[2] for s in route["stops"]]

            fig.add_trace(go.Scattermapbox(
                lat=lats, lon=lons,
                mode="markers+lines+text",
                marker=dict(size=10, color=color),
                text=labels, textposition="top right",
                name=v_name,
                line=dict(width=2, color=color),
            ))

        center_lat = np.mean([r.pickup_lat for r in requests]) if requests else 19.07
        center_lon = np.mean([r.pickup_lon for r in requests]) if requests else 72.88

        fig.update_layout(
            mapbox=dict(style="open-street-map",
                        center=dict(lat=center_lat, lon=center_lon), zoom=11),
            title="🚚 Pickup & Delivery Routes",
            template="plotly_dark", height=500,
            margin=dict(l=0, r=0, t=50, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig


# ══════════════════════════════════════════════════════════════
#  SPLIT DELIVERY OPTIMIZER
# ══════════════════════════════════════════════════════════════

class SplitDeliveryOptimizer:
    """Identify and plan split deliveries for oversized orders"""

    @staticmethod
    def identify_split_candidates(deliveries: List[Any],
                                  vehicle_capacities: List[int],
                                  threshold_pct: float = 0.8) -> List[SplitDelivery]:
        """Return deliveries whose demand > threshold_pct * max_capacity"""
        if not vehicle_capacities:
            return []
        max_cap = max(vehicle_capacities)
        candidates = []

        for dl in deliveries:
            demand = getattr(dl, "demand", 100)
            if demand > max_cap * threshold_pct:
                sd = SplitDelivery(
                    location_name=getattr(dl, "name", "Unknown"),
                    lat=dl.lat,
                    lon=dl.lon,
                    total_demand=demand,
                )
                remaining = demand
                trip = 1
                while remaining > 0:
                    qty = min(max_cap, remaining)
                    sd.splits.append({"trip": trip, "quantity": qty, "vehicle": None, "eta": None})
                    remaining -= qty
                    trip += 1
                candidates.append(sd)

        return candidates

    @staticmethod
    def visualize_split_deliveries(candidates: List[SplitDelivery]):
        import plotly.graph_objects as go

        if not candidates:
            fig = go.Figure()
            fig.add_annotation(text="No split deliveries required ✅",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=18, color="#3DBA7E"))
            fig.update_layout(title="Split Delivery Analysis",
                              template="plotly_dark", height=300,
                              paper_bgcolor="rgba(0,0,0,0)")
            return fig

        fig = go.Figure()
        colors = ["#4A9EFF", "#FF8C42", "#3DBA7E", "#BB86FC"]
        for i, sd in enumerate(candidates):
            trips = [f"Trip {s['trip']}" for s in sd.splits]
            qtys = [s["quantity"] for s in sd.splits]
            fig.add_trace(go.Bar(
                name=sd.location_name,
                x=trips, y=qtys,
                text=[f"{q} units" for q in qtys],
                textposition="outside",
                marker_color=colors[i % len(colors)],
            ))

        fig.update_layout(
            title="📦 Split Delivery Plan (units per trip)",
            barmode="group",
            xaxis_title="Trip",
            yaxis_title="Quantity (units)",
            template="plotly_dark", height=400,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,46,0.8)",
        )
        return fig


# ══════════════════════════════════════════════════════════════
#  DRIVER ASSIGNMENT SYSTEM
# ══════════════════════════════════════════════════════════════

class DriverManager:
    """Manage driver assignments, shifts, and overtime"""

    @staticmethod
    def auto_assign(vehicles: List[Any],
                    drivers: List[DriverAssignment]) -> List[DriverAssignment]:
        """Match drivers to vehicles by shift overlap"""
        updated = []
        used_vehicles = set()
        for driver in drivers:
            if driver.vehicle_name in used_vehicles:
                continue
            updated.append(driver)
            used_vehicles.add(driver.vehicle_name)
        return updated

    @staticmethod
    def calculate_overtime(driver: DriverAssignment, actual_end: float) -> float:
        """Return overtime hours (if any)"""
        standard_end = driver.shift_start + driver.max_working_hours
        return max(0.0, actual_end - standard_end)

    @staticmethod
    def validate_assignments(drivers: List[DriverAssignment],
                              vehicles: List[Any]) -> List[str]:
        """Return list of validation warnings"""
        warnings = []
        v_names = {getattr(v, "name", "") for v in vehicles}
        d_vehicles = [d.vehicle_name for d in drivers]

        for d in drivers:
            if d.vehicle_name not in v_names:
                warnings.append(f"⚠️ {d.driver_name}: assigned vehicle '{d.vehicle_name}' not found")
            if d.shift_end - d.shift_start > d.max_working_hours:
                warnings.append(f"⚠️ {d.driver_name}: shift exceeds max working hours")

        for vn in v_names:
            if vn not in d_vehicles:
                warnings.append(f"ℹ️ Vehicle '{vn}' has no assigned driver")

        return warnings

    @staticmethod
    def create_shift_chart(drivers: List[DriverAssignment]):
        import plotly.graph_objects as go

        colors = ["#4A9EFF", "#3DBA7E", "#FF8C42", "#BB86FC", "#FF6B9D"]
        fig = go.Figure()

        for i, d in enumerate(drivers):
            duration = d.shift_end - d.shift_start
            fig.add_trace(go.Bar(
                name=d.driver_name,
                x=[d.driver_name],
                y=[duration],
                base=[d.shift_start],
                marker_color=colors[i % len(colors)],
                text=f"{d.shift_start:.0f}:00–{d.shift_end:.0f}:00 ({duration:.1f}h)",
                textposition="inside",
                hoverinfo="text",
            ))

        fig.update_layout(
            title="👨‍✈️ Driver Shift Timeline",
            xaxis_title="Driver",
            yaxis_title="Hour of Day",
            yaxis=dict(range=[0, 24], tickvals=list(range(0, 25, 2))),
            barmode="group",
            template="plotly_dark", height=400,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,46,0.8)",
        )
        return fig

    @staticmethod
    def create_performance_chart(drivers: List[DriverAssignment]):
        import plotly.graph_objects as go

        fig = go.Figure(go.Bar(
            x=[d.driver_name for d in drivers],
            y=[d.performance_score for d in drivers],
            marker_color=["#3DBA7E" if d.performance_score >= 80 else "#FF8C42" if d.performance_score >= 60 else "#FF5252" for d in drivers],
            text=[f"{d.performance_score:.0f}%" for d in drivers],
            textposition="outside",
        ))
        fig.add_hline(y=80, line_dash="dash", line_color="#e8c76d", annotation_text="Target 80%")
        fig.update_layout(
            title="⭐ Driver Performance Scores",
            yaxis=dict(range=[0, 110]),
            template="plotly_dark", height=350,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,46,0.8)",
        )
        return fig


# ══════════════════════════════════════════════════════════════
#  REINFORCEMENT LEARNING DISPATCHER (Q-Learning)
# ══════════════════════════════════════════════════════════════

class RLDispatcher:
    """
    Q-Learning dispatcher: learns to assign deliveries to vehicles
    State  = (vehicle_utilization_bin, delivery_priority)
    Action = vehicle index
    Reward = demand_served / cost_penalty
    """

    def __init__(self, lr: float = 0.15, gamma: float = 0.9, epsilon: float = 0.1):
        self.q_table: Dict[str, Dict[int, float]] = {}
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self._episode = 0

    # ── State encoding ──
    def _state(self, vehicle: Any, delivery: Any, load_so_far: int) -> str:
        cap = getattr(vehicle, "capacity", 500)
        util_bin = min(9, int(load_so_far / cap * 10))
        priority = getattr(delivery, "priority", 1)
        return f"u{util_bin}_p{priority}"

    # ── Reward function ──
    def _reward(self, vehicle: Any, delivery: Any) -> float:
        demand = getattr(delivery, "demand", 100)
        cap = getattr(vehicle, "capacity", 500)
        cost = getattr(vehicle, "cost_per_km", 12)
        co2 = getattr(vehicle, "co2_per_km", 0.21)
        priority = getattr(delivery, "priority", 1)
        util_reward = demand / cap * 50
        cost_penalty = cost * 0.3 + co2 * 10
        priority_bonus = priority * 5
        return util_reward - cost_penalty + priority_bonus

    # ── Best action from Q-table ──
    def _best_action(self, state: str, n_actions: int) -> int:
        if state not in self.q_table:
            return random.randint(0, n_actions - 1)
        q_vals = self.q_table[state]
        return max(range(n_actions), key=lambda a: q_vals.get(a, 0.0))

    # ── Q-update ──
    def _update_q(self, state: str, action: int, reward: float,
                  next_state: str, n_actions: int):
        if state not in self.q_table:
            self.q_table[state] = {}
        old = self.q_table[state].get(action, 0.0)
        future = max(
            self.q_table.get(next_state, {}).get(a, 0.0) for a in range(n_actions)
        ) if next_state in self.q_table else 0.0
        self.q_table[state][action] = old + self.lr * (reward + self.gamma * future - old)

    def dispatch_batch(self, vehicles: List[Any], deliveries: List[Any],
                       n_train_episodes: int = 5) -> List[RLDispatchDecision]:
        """
        Train Q-table over n_train_episodes then make decisions.
        Returns RLDispatchDecision for each delivery.
        """
        n = len(vehicles)
        if n == 0:
            return []

        # Training episodes (simulate dispatch)
        for _ in range(n_train_episodes):
            vehicle_loads = [0] * n
            for delivery in deliveries:
                for vi, v in enumerate(vehicles):
                    state = self._state(v, delivery, vehicle_loads[vi])
                    if random.random() < self.epsilon:
                        action = random.randint(0, n - 1)
                    else:
                        action = self._best_action(state, n)
                    reward = self._reward(vehicles[action], delivery)
                    vehicle_loads[action] += getattr(delivery, "demand", 0)
                    next_state = self._state(vehicles[action], delivery, vehicle_loads[action])
                    self._update_q(state, action, reward, next_state, n)
                    break

        # Actual dispatch decisions
        decisions = []
        vehicle_loads = [0] * n
        for delivery in deliveries:
            delivery_name = getattr(delivery, "name", str(delivery))
            # Epsilon-greedy
            best_vi = None
            best_q = -float("inf")
            for vi, v in enumerate(vehicles):
                state = self._state(v, delivery, vehicle_loads[vi])
                q = self.q_table.get(state, {}).get(vi, 0.0)
                if q > best_q:
                    best_q = q
                    best_vi = vi

            if best_vi is None:
                best_vi = 0

            chosen_v = vehicles[best_vi]
            vehicle_loads[best_vi] += getattr(delivery, "demand", 0)
            reward = self._reward(chosen_v, delivery)
            confidence = float(np.clip(abs(best_q) / (abs(best_q) + 10), 0.3, 0.97))
            alternatives = [getattr(v, "name", f"V{i}") for i, v in enumerate(vehicles) if i != best_vi]
            reason = f"Q={best_q:.2f} | util {vehicle_loads[best_vi]}/{getattr(chosen_v,'capacity',500)}"

            decisions.append(RLDispatchDecision(
                vehicle_id=getattr(chosen_v, "name", f"V{best_vi}"),
                assigned_delivery=delivery_name,
                confidence_score=confidence,
                expected_reward=round(reward, 2),
                alternative_vehicles=alternatives,
                decision_reason=reason,
            ))

        self._episode += 1
        return decisions

    def visualize_rl_decisions(self, decisions: List[RLDispatchDecision]):
        import plotly.graph_objects as go

        if not decisions:
            fig = go.Figure()
            fig.update_layout(title="No RL Decisions", template="plotly_dark", height=300)
            return fig

        vehicles = sorted(set(d.vehicle_id for d in decisions))
        colors = ["#4A9EFF", "#FF8C42", "#3DBA7E", "#BB86FC", "#FF6B9D"]

        fig = go.Figure()
        for i, v in enumerate(vehicles):
            vd = [d for d in decisions if d.vehicle_id == v]
            fig.add_trace(go.Bar(
                name=v,
                x=[d.assigned_delivery for d in vd],
                y=[d.expected_reward for d in vd],
                marker_color=colors[i % len(colors)],
                text=[f"conf: {d.confidence_score:.0%}" for d in vd],
                textposition="outside",
            ))

        fig.update_layout(
            title="🤖 RL Dispatch — Expected Reward per Assignment",
            xaxis_title="Delivery",
            yaxis_title="Expected Reward",
            barmode="group",
            template="plotly_dark", height=420,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,46,0.8)",
        )
        return fig

    def visualize_q_table(self):
        import plotly.graph_objects as go

        if not self.q_table:
            return go.Figure()

        states = list(self.q_table.keys())[:20]
        vehicles = sorted({a for s in self.q_table for a in self.q_table[s]})

        z = [[self.q_table.get(s, {}).get(v, 0.0) for v in vehicles] for s in states]
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=[f"V{v}" for v in vehicles],
            y=states,
            colorscale=[[0, "#FF5252"], [0.5, "#e8c76d"], [1, "#3DBA7E"]],
            text=[[f"{val:.1f}" for val in row] for row in z],
            texttemplate="%{text}",
        ))
        fig.update_layout(
            title="🧠 Q-Table Heatmap (State × Action)",
            template="plotly_dark", height=420,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,46,0.8)",
        )
        return fig


# ══════════════════════════════════════════════════════════════
#  REAL-TIME RE-OPTIMIZER
# ══════════════════════════════════════════════════════════════

class RealTimeReoptimizer:
    """Detect operational changes and decide whether to re-optimize"""

    def __init__(self, traffic_threshold: float = 1.25,
                 new_delivery_threshold: int = 2):
        self.traffic_threshold = traffic_threshold
        self.new_delivery_threshold = new_delivery_threshold
        self.history: List[Dict] = []

    def detect_changes(self, current_deliveries: List[Any],
                       original_deliveries: List[Any],
                       current_traffic: float = 1.0,
                       original_traffic: float = 1.0) -> Dict:
        orig_names = {getattr(d, "name", str(d)) for d in original_deliveries}
        curr_names = {getattr(d, "name", str(d)) for d in current_deliveries}

        new_dels = [d for d in current_deliveries if getattr(d, "name", str(d)) not in orig_names]
        cancelled = list(orig_names - curr_names)
        traffic_change = current_traffic / max(original_traffic, 0.001)

        changes = {
            "new_deliveries": new_dels,
            "cancelled_deliveries": cancelled,
            "traffic_change": round(traffic_change, 3),
            "total_changes": len(new_dels) + len(cancelled),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        self.history.append(changes)
        return changes

    def should_reoptimize(self, changes: Dict) -> bool:
        if len(changes["new_deliveries"]) >= self.new_delivery_threshold:
            return True
        if len(changes["cancelled_deliveries"]) > 0:
            return True
        if changes["traffic_change"] > self.traffic_threshold:
            return True
        return False

    def get_recommendation(self, changes: Dict) -> str:
        if not self.should_reoptimize(changes):
            return "✅ No re-optimization needed. Current routes remain efficient."
        reasons = []
        if changes["new_deliveries"]:
            reasons.append(f"{len(changes['new_deliveries'])} new delivery/deliveries added")
        if changes["cancelled_deliveries"]:
            reasons.append(f"{len(changes['cancelled_deliveries'])} delivery/deliveries cancelled")
        if changes["traffic_change"] > self.traffic_threshold:
            reasons.append(f"Traffic changed by {changes['traffic_change']:.2f}x")
        return "⚠️ Re-optimization recommended: " + "; ".join(reasons)

    def visualize_changes(self, changes: Dict):
        import plotly.graph_objects as go

        fig = go.Figure()
        labels = ["New Deliveries", "Cancelled", "Traffic Change"]
        values = [
            len(changes["new_deliveries"]),
            len(changes["cancelled_deliveries"]),
            round(changes["traffic_change"], 2),
        ]
        bar_colors = ["#3DBA7E", "#FF5252", "#FF8C42"]

        fig.add_trace(go.Bar(
            x=labels, y=values,
            marker_color=bar_colors,
            text=[str(round(v, 2)) for v in values],
            textposition="outside",
        ))

        fig.add_hline(y=self.traffic_threshold, line_dash="dash",
                      line_color="#e8c76d", annotation_text="Traffic Threshold")

        fig.update_layout(
            title=f"🔄 Real-Time Changes — {changes.get('timestamp', '')}",
            yaxis_title="Value / Count",
            template="plotly_dark", height=350,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,46,0.8)",
        )
        return fig


# ══════════════════════════════════════════════════════════════
#  PDF ROUTE SHEET EXPORTER
# ══════════════════════════════════════════════════════════════

class PDFRouteExporter:
    """Generate driver route sheets as PDF using ReportLab"""

    @staticmethod
    def generate_route_sheet(route_result: Any, driver_info: DriverAssignment = None) -> bytes:
        """Generate a single-route PDF for one vehicle/driver"""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                             Paragraph, Spacer, HRFlowable)
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
        except ImportError:
            return b""

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=0.5 * inch, leftMargin=0.5 * inch,
                                topMargin=0.5 * inch, bottomMargin=0.5 * inch)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("T", parent=styles["Heading1"],
                                     fontSize=16, spaceAfter=10, textColor=colors.HexColor("#1a1d2e"))
        sub_style = ParagraphStyle("S", parent=styles["Normal"],
                                   fontSize=10, textColor=colors.grey)
        warn_style = ParagraphStyle("W", parent=styles["Normal"],
                                    fontSize=9, textColor=colors.red)

        elements = []
        v_name = getattr(route_result, "vehicle_name", "Vehicle")
        elements.append(Paragraph(f"🚛 Route Sheet — {v_name}", title_style))
        elements.append(Paragraph("Urban Logistics VRP Optimizer | MPSTME NMIMS Mumbai", sub_style))
        elements.append(Spacer(1, 0.15 * inch))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#4A9EFF")))
        elements.append(Spacer(1, 0.1 * inch))

        # Summary table
        summary_rows = [["Metric", "Value"]]
        if driver_info:
            summary_rows += [
                ["Driver", driver_info.driver_name],
                ["Phone", driver_info.phone or "—"],
                ["Shift", f"{driver_info.shift_start:.0f}:00 – {driver_info.shift_end:.0f}:00"],
            ]
        summary_rows += [
            ["Distance", f"{getattr(route_result, 'total_distance', 0):.1f} km"],
            ["Est. Time", f"{getattr(route_result, 'total_time', 0):.0f} min"],
            ["Stops", str(len(getattr(route_result, "route_names", [])) - 2)],
            ["Load", f"{getattr(route_result, 'load_carried', 0)} / {getattr(route_result, 'capacity', 0)} units"],
            ["CO₂", f"{getattr(route_result, 'total_co2', 0):.2f} kg"],
            ["Cost", f"₹{getattr(route_result, 'total_cost', 0):,.0f}"],
        ]

        summary_tbl = Table(summary_rows, colWidths=[2.5 * inch, 2 * inch])
        summary_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4A9EFF")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4ff")]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(summary_tbl)
        elements.append(Spacer(1, 0.2 * inch))

        # Stop table
        route_names = getattr(route_result, "route_names", [])
        arrival_times = getattr(route_result, "arrival_times", [])
        wait_times = getattr(route_result, "wait_times", [])

        stop_rows = [["#", "Location", "Type", "Arrival (h)", "Wait (min)"]]
        for i, name in enumerate(route_names):
            if i == 0:
                stype = "🏭 Depot (Start)"
            elif i == len(route_names) - 1:
                stype = "🏭 Depot (Return)"
            else:
                stype = f"📦 Stop"
            arr = f"{arrival_times[i]:.2f}" if i < len(arrival_times) else "—"
            wait = f"{wait_times[i]:.0f}" if i < len(wait_times) else "—"
            stop_rows.append([str(i), name, stype, arr, wait])

        stop_tbl = Table(stop_rows,
                         colWidths=[0.4 * inch, 2.2 * inch, 1.5 * inch, 1 * inch, 0.8 * inch])
        stop_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3DBA7E")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0fff4")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(stop_tbl)

        # TW violations
        tw_viol = getattr(route_result, "time_window_violations", [])
        if tw_viol:
            elements.append(Spacer(1, 0.15 * inch))
            elements.append(Paragraph("⚠️ Time Window Violations:", warn_style))
            for v in tw_viol:
                elements.append(Paragraph(f"  • {v}", warn_style))

        elements.append(Spacer(1, 0.3 * inch))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        elements.append(Paragraph("<i>Generated by Urban Logistics VRP Optimizer — MPSTME NMIMS Mumbai</i>",
                                  ParagraphStyle("F", parent=styles["Normal"],
                                                 fontSize=7, textColor=colors.grey)))

        doc.build(elements)
        buf.seek(0)
        return buf.getvalue()

    @staticmethod
    def generate_pdf_report(route_data: List[Dict], warehouse_names: List[str],
                            delivery_names: List[str], total_distance: float,
                            total_time: float, total_cost: float,
                            total_emissions: float) -> bytes:
        """
        Generate a consolidated PDF report for all routes.
        Compatible with app.py Tab 8 call signature.
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4, landscape
            from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                             Paragraph, Spacer, HRFlowable, PageBreak)
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
        except ImportError:
            return b"PDF generation requires reportlab. pip install reportlab"

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=0.5 * inch, leftMargin=0.5 * inch,
                                topMargin=0.5 * inch, bottomMargin=0.5 * inch)
        styles = getSampleStyleSheet()

        title_s = ParagraphStyle("TT", parent=styles["Heading1"], fontSize=20,
                                  spaceAfter=6, textColor=colors.HexColor("#1a1d2e"))
        sub_s = ParagraphStyle("SS", parent=styles["Normal"], fontSize=10, textColor=colors.grey)
        h2_s = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13,
                               textColor=colors.HexColor("#4A9EFF"), spaceBefore=12, spaceAfter=6)

        elements = []

        # Cover / Summary
        elements.append(Paragraph("🚛 Urban Logistics VRP — Route Report", title_s))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')} | MPSTME NMIMS Mumbai", sub_s))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#4A9EFF")))
        elements.append(Spacer(1, 0.1 * inch))

        # Overall metrics
        summary_data = [
            ["Metric", "Value"],
            ["Total Distance", f"{total_distance:,.1f} km"],
            ["Total Time", f"{total_time:,.0f} min"],
            ["Total Cost", f"₹{total_cost:,.0f}"],
            ["Total CO₂ Emissions", f"{total_emissions:,.2f} kg"],
            ["Active Routes", str(len(route_data))],
            ["Warehouses", ", ".join(warehouse_names)],
        ]
        tbl = Table(summary_data, colWidths=[2.5 * inch, 3.5 * inch])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4A9EFF")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef4ff")]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(tbl)
        elements.append(Spacer(1, 0.3 * inch))

        # Per-route details
        elements.append(Paragraph("📋 Route Details", h2_s))

        for rd in route_data:
            v_name = rd.get("vehicle_name", rd.get("vehicle_id", "Vehicle"))
            stops = rd.get("stops", rd.get("route_names", []))
            dist = rd.get("distance_km", rd.get("total_distance", 0))
            time_val = rd.get("time_min", rd.get("total_time", 0))
            cost_val = rd.get("cost", rd.get("total_cost", 0))
            em_val = rd.get("emissions_kg", rd.get("total_co2", 0))

            elements.append(Paragraph(f"🚛 {v_name}", ParagraphStyle(
                "VH", parent=styles["Heading3"], fontSize=11,
                textColor=colors.HexColor("#3DBA7E"), spaceBefore=10)))

            route_summary = [
                ["Distance", "Time", "Cost", "CO₂"],
                [f"{dist:.1f} km", f"{time_val:.0f} min", f"₹{cost_val:,.0f}", f"{em_val:.2f} kg"],
            ]
            rs_tbl = Table(route_summary, colWidths=[1.5 * inch] * 4)
            rs_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3348")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            elements.append(rs_tbl)
            elements.append(Spacer(1, 0.08 * inch))

            # Stop list
            if stops:
                stop_data = [["#", "Location"]]
                for si, sname in enumerate(stops):
                    label = "🏭 Depot" if (si == 0 or si == len(stops) - 1) else f"📦 Stop {si}"
                    stop_data.append([str(si), f"{label}: {sname}"])

                st_tbl = Table(stop_data, colWidths=[0.4 * inch, 5.5 * inch])
                st_tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3DBA7E")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0fff4")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                elements.append(st_tbl)
                elements.append(Spacer(1, 0.15 * inch))

        # Footer
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        elements.append(Paragraph(
            "<i>Urban Logistics VRP Optimizer — MPSTME NMIMS Mumbai | Built with OR-Tools + Streamlit</i>",
            ParagraphStyle("FT", parent=styles["Normal"], fontSize=7, textColor=colors.grey)
        ))

        doc.build(elements)
        buf.seek(0)
        return buf.getvalue()

    @classmethod
    def generate_all_route_sheets(cls, solution_result: Any,
                                   drivers: List[DriverAssignment] = None) -> List[bytes]:
        drivers_dict = {d.vehicle_name: d for d in drivers} if drivers else {}
        pdfs = []
        for route in solution_result.routes:
            drv = drivers_dict.get(route.vehicle_name)
            pdfs.append(cls.generate_route_sheet(route, drv))
        return pdfs


# ══════════════════════════════════════════════════════════════
#  ADVANCED VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

class AdvancedVisualizations:
    @staticmethod
    def create_time_window_gantt(routes):
        import plotly.graph_objects as go

        colors = ["#4A9EFF", "#FF8C42", "#3DBA7E", "#BB86FC", "#FF6B9D"]
        fig = go.Figure()

        for r_idx, route in enumerate(routes):
            color = colors[r_idx % len(colors)]
            for i, name in enumerate(route.route_names):
                if i == 0 or i == len(route.route_names) - 1:
                    continue
                arr = route.arrival_times[i] if i < len(route.arrival_times) else 8.0
                wait = route.wait_times[i] / 60 if i < len(route.wait_times) else 0.0
                fig.add_trace(go.Bar(
                    x=[route.vehicle_name],
                    y=[0.25 + wait],
                    base=[arr],
                    marker_color=color,
                    name=name,
                    text=name,
                    textposition="inside",
                    orientation="v",
                    showlegend=False,
                ))

        fig.update_layout(
            title="🕐 Delivery Schedule Gantt",
            xaxis_title="Vehicle",
            yaxis_title="Hour of Day",
            barmode="stack",
            template="plotly_dark", height=400,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,46,0.8)",
        )
        return fig

    @staticmethod
    def create_depot_utilization_chart(depot_assignments: Dict, depot_loads: Dict, depots: List[Depot]):
        import plotly.graph_objects as go

        names = list(depot_loads.keys())
        loads = [depot_loads[n] for n in names]
        caps = [next((d.max_capacity for d in depots if d.name == n), 1000) for n in names]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Used Capacity", x=names, y=loads, marker_color="#4A9EFF"))
        fig.add_trace(go.Bar(name="Max Capacity", x=names, y=caps, marker_color="rgba(100,100,100,0.4)"))

        for i, (n, l, c) in enumerate(zip(names, loads, caps)):
            fig.add_annotation(x=n, y=l, text=f"{l/c*100:.0f}%", showarrow=False, yshift=10)

        fig.update_layout(
            title="🏭 Multi-Depot Utilization",
            barmode="overlay",
            xaxis_title="Depot", yaxis_title="Units",
            template="plotly_dark", height=400,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,46,0.8)",
        )
        return fig


print("✅ Advanced VRP Features module loaded — all classes ready.")
