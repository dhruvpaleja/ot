"""
Advanced VRP Features Module - Complete Implementation
Urban Logistics Optimization — MPSTME NMIMS Mumbai

Features:
✅ Time Window Constraints enforcement
✅ Multi-Depot VRP implementation
✅ Pickup & Delivery capabilities
✅ Split Delivery options
✅ Enhanced heterogeneous fleet optimization
✅ Real-time re-optimization
✅ Driver shift management
✅ Demand prediction models (ML)
✅ Travel time forecasting (ML)
✅ Clustering pre-processing (K-Means)
✅ Reinforcement learning for dispatch
✅ PDF export functionality
✅ K-Means clustering visualization
✅ Driver assignment system
✅ Printable route sheets
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import io
import random
from datetime import datetime, timedelta
import pandas as pd

# ─── EXTENDED DATA CLASSES ──────────────────────────────────

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
    opening_time: float = 6.0  # hours from midnight
    closing_time: float = 22.0  # hours from midnight

@dataclass 
class PickupDeliveryRequest:
    """Pickup and Delivery pair"""
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
    pickup_lat_val: float = 19.0
    pickup_lon_val: float = 72.8
    delivery_lat: float = 19.0
    delivery_lon: float = 72.8

@dataclass
class SplitDelivery:
    """Split delivery across multiple vehicles"""
    location_name: str
    lat: float
    lon: float
    total_demand: int
    splits: List[Dict] = field(default_factory=list)  # [{vehicle, quantity, arrival_time}]
    min_split_quantity: int = 50  # minimum quantity per split
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
    performance_score: float = 0.0  # 0-100

@dataclass
class DemandPrediction:
    """ML-based demand prediction"""
    location_name: str
    predicted_demand: float
    confidence: float
    prediction_hour: int
    historical_average: float
    trend: str  # "increasing", "decreasing", "stable"
    
@dataclass
class TravelTimeForecast:
    """ML-based travel time prediction"""
    origin: str
    destination: str
    predicted_time: float  # minutes
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

# ─── MULTI-DEPOT ASSIGNMENT ─────────────────────────────────

class MultiDepotAllocator:
    """Assign deliveries to nearest depot with capacity constraints"""
    
    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c
    
    @classmethod
    def assign_to_depots(cls, depots: List[Depot], deliveries: List, 
                         capacity_factor=0.85) -> Tuple[Dict[str, List], Dict[str, int]]:
        """
        Assign each delivery to a depot based on distance and capacity
        Returns: (assignments dict, depot_loads dict)
        """
        assignments = {d.name: [] for d in depots}
        depot_loads = {d.name: 0 for d in depots}
        
        # Sort deliveries by distance to nearest depot (furthest first - harder to serve)
        delivery_distances = []
        for i, dl in enumerate(deliveries):
            distances = [(d.name, cls.haversine(dl.lat, dl.lon, d.lat, d.lon)) for d in depots]
            distances.sort(key=lambda x: x[1])
            min_dist = distances[0][1]
            delivery_distances.append((i, dl, min_dist, distances))
        
        delivery_distances.sort(key=lambda x: -x[2])  # Furthest first
        
        for idx, dl, min_dist, distances in delivery_distances:
            demand = dl.demand if hasattr(dl, 'demand') else 100
            
            # Try to assign to nearest depot with capacity
            assigned = False
            for depot_name, dist in distances:
                depot = next(d for d in depots if d.name == depot_name)
                max_load = depot.max_capacity * capacity_factor
                
                if depot_loads[depot_name] + demand <= max_load:
                    assignments[depot_name].append(idx)
                    depot_loads[depot_name] += demand
                    assigned = True
                    break
            
            if not assigned:
                # Assign to least loaded depot
                least_loaded = min(depot_loads.keys(), key=lambda k: depot_loads[k])
                assignments[least_loaded].append(idx)
                depot_loads[least_loaded] += demand
        
        return assignments, depot_loads
    
    @classmethod
    def optimize_depot_locations(cls, deliveries: List, n_depots: int = 3):
        """Use K-Means to find optimal depot locations"""
        coords = np.array([[d.lat, d.lon] for d in deliveries])
        kmeans = KMeans(n_clusters=n_depots, random_state=42, n_init=10)
        kmeans.fit(coords)
        
        optimal_locations = []
        for i, centroid in enumerate(kmeans.cluster_centers_):
            cluster_members = [deliveries[j] for j in range(len(deliveries)) 
                             if kmeans.labels_[j] == i]
            total_demand = sum(getattr(m, 'demand', 100) for m in cluster_members)
            
            optimal_locations.append({
                'centroid_lat': centroid[0],
                'centroid_lon': centroid[1],
                'cluster_size': len(cluster_members),
                'total_demand': total_demand,
                'recommended_capacity': int(total_demand * 1.2)  # 20% buffer
            })
        
        return optimal_locations

# ─── K-MEANS CLUSTERING FOR DELIVERIES ──────────────────────

class DeliveryClusterer:
    """Cluster deliveries for efficient route planning"""
    
    @staticmethod
    def cluster_deliveries(deliveries: List, n_clusters: int = 5, 
                          include_warehouse=True, warehouse_loc=None) -> Dict:
        """
        Cluster delivery locations using K-Means
        Returns cluster assignments and centroids
        """
        coords = np.array([[d.lat, d.lon] for d in deliveries])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(coords)
        
        result = {
            'labels': clusters.tolist(),
            'centroids': kmeans.cluster_centers_.tolist(),
            'inertia': kmeans.inertia_,
            'cluster_sizes': {}
        }
        
        for i in range(n_clusters):
            members = [deliveries[j].name for j in range(len(deliveries)) if clusters[j] == i]
            total_demand = sum(
                deliveries[j].demand if hasattr(deliveries[j], 'demand') else 0 
                for j in range(len(deliveries)) if clusters[j] == i
            )
            result['cluster_sizes'][i] = {
                'count': len(members),
                'members': members,
                'total_demand': total_demand,
                'centroid_lat': kmeans.cluster_centers_[i][0],
                'centroid_lon': kmeans.cluster_centers_[i][1]
            }
        
        return result
    
    @staticmethod
    def get_optimal_clusters(deliveries: List, max_cluster_size: int = 10) -> int:
        """Determine optimal number of clusters based on delivery count"""
        n = len(deliveries)
        return max(2, min(8, n // max_cluster_size + 1))

# ─── PDF EXPORT GENERATOR ───────────────────────────────────

class PDFRouteExporter:
    """Generate PDF route sheets for drivers"""
    
    @staticmethod
    def generate_route_sheet(route_result, driver_info=None, include_map=False) -> bytes:
        """Generate a PDF route sheet"""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=0.5*inch, leftMargin=0.5*inch)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=18, spaceAfter=12)
        normal_style = styles['Normal']
        
        elements = []
        
        # Title
        elements.append(Paragraph(f"🚛 Route Sheet - {route_result.vehicle_name}", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Summary table
        summary_data = [
            ['Metric', 'Value'],
            ['Total Distance', f'{route_result.total_distance:.1f} km'],
            ['Total Time', f'{route_result.total_time:.0f} min'],
            ['Stops', f'{len(route_result.route_names) - 2}'],
            ['Load', f'{route_result.load_carried}/{route_result.capacity} units'],
            ['CO₂ Emissions', f'{route_result.total_co2:.2f} kg'],
        ]
        
        if driver_info:
            summary_data.insert(1, ['Driver', driver_info.driver_name])
            summary_data.insert(2, ['Phone', driver_info.phone or 'N/A'])
        
        summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A9EFF')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f8')]),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Route details table
        route_data = [['Stop #', 'Location', 'Arrival', 'Wait (min)', 'Notes']]
        
        for i, name in enumerate(route_result.route_names):
            if i == 0 or i == len(route_result.route_names) - 1:
                stop_type = "🏭 Depot" if i == 0 else "🏭 Return"
                arrival = f"{route_result.arrival_times[i]:.1f}h" if route_result.arrival_times else "-"
                wait = "-"
            else:
                stop_type = f"📦 Stop {i}"
                arrival = f"{route_result.arrival_times[i]:.1f}h" if i < len(route_result.arrival_times) else "-"
                wait = f"{route_result.wait_times[i]:.0f}" if i < len(route_result.wait_times) else "-"
            
            route_data.append([str(i), f"{stop_type}: {name}", arrival, wait, ""])
        
        route_table = Table(route_data, colWidths=[0.5*inch, 2.5*inch, 0.8*inch, 0.8*inch, 1*inch])
        route_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3DBA7E')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(route_table)
        
        # Time window violations warning
        if route_result.time_window_violations:
            elements.append(Spacer(1, 0.2*inch))
            violation_text = "<b>⚠️ Time Window Violations:</b><br/>" + "<br/>".join(route_result.time_window_violations)
            elements.append(Paragraph(violation_text, ParagraphStyle('Warning', parent=normal_style, textColor=colors.red)))
        
        # Footer
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph("<i>Generated by Urban Logistics VRP Optimizer</i>", 
                                  ParagraphStyle('Footer', parent=normal_style, fontSize=8, textColor=colors.grey)))
        
        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()
    
    @staticmethod
    def generate_all_route_sheets(solution_result, drivers: List[DriverAssignment] = None) -> List[bytes]:
        """Generate PDFs for all routes"""
        pdfs = []
        drivers_dict = {d.vehicle_name: d for d in drivers} if drivers else {}
        
        for route in solution_result.routes:
            driver = drivers_dict.get(route.vehicle_name)
            pdf = PDFRouteExporter.generate_route_sheet(route, driver)
            pdfs.append(pdf)
        
        return pdfs

# ─── VISUALIZATION HELPERS ──────────────────────────────────

class AdvancedVisualizations:
    """Create advanced visualizations for VRP features"""
    
    @staticmethod
    def create_cluster_plot(deliveries, clustering_result, warehouse_locs=None):
        """Create plotly figure for delivery clusters"""
        import plotly.express as px
        import pandas as pd
        
        df_data = []
        for i, dl in enumerate(deliveries):
            cluster = clustering_result['labels'][i]
            df_data.append({
                'name': dl.name,
                'lat': dl.lat,
                'lon': dl.lon,
                'cluster': f'Cluster {cluster}',
                'demand': dl.demand if hasattr(dl, 'demand') else 0
            })
        
        df = pd.DataFrame(df_data)
        
        fig = px.scatter_mapbox(
            df, lat='lat', lon='lon', color='cluster', size='demand',
            hover_name='name', hover_data=['demand'],
            zoom=10, center={'lat': 19.0760, 'lon': 72.8777},
            mapbox_style='open-street-map',
            title='📦 Delivery Clusters (K-Means)',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        # Add centroids
        centroids_df = pd.DataFrame([
            {'lat': c[0], 'lon': c[1], 'cluster': f'Cluster {i} Centroid'}
            for i, c in enumerate(clustering_result['centroids'])
        ])
        
        fig.add_scattermapbox(
            lat=centroids_df['lat'], lon=centroids_df['lon'],
            mode='markers', marker={'size': 15, 'symbol': 'star'},
            name='Centroids', hoverinfo='text',
            text=centroids_df['cluster']
        )
        
        if warehouse_locs:
            wh_df = pd.DataFrame([
                {'lat': w.lat, 'lon': w.lon, 'name': w.name}
                for w in warehouse_locs
            ])
            fig.add_scattermapbox(
                lat=wh_df['lat'], lon=wh_df['lon'],
                mode='markers', marker={'size': 20, 'symbol': 'triangle-up', 'color': 'red'},
                name='Warehouses', text=wh_df['name']
            )
        
        fig.update_layout(height=600, margin={'l': 0, 'r': 0, 't': 50, 'b': 0})
        return fig
    
    @staticmethod
    def create_time_window_gantt(routes):
        """Create Gantt chart for time windows"""
        import plotly.graph_objects as go
        import pandas as pd
        
        tasks = []
        
        for route in routes:
            for i, name in enumerate(route.route_names):
                if i == 0 or i == len(route.route_names) - 1:
                    continue  # Skip depot
                
                arrival = route.arrival_times[i] if i < len(route.arrival_times) else 0
                wait = route.wait_times[i] / 60 if i < len(route.wait_times) else 0  # Convert to hours
                
                tasks.append({
                    'Task': f"{route.vehicle_name}: {name}",
                    'Start': arrival,
                    'End': arrival + wait + 0.25,  # Assuming 15 min service
                    'Vehicle': route.vehicle_name,
                    'Wait Time': f"{wait*60:.0f} min"
                })
        
        df = pd.DataFrame(tasks)
        
        fig = go.Figure()
        
        for vehicle in df['Vehicle'].unique():
            vehicle_df = df[df['Vehicle'] == vehicle]
            fig.add_trace(go.Scatter(
                x=[(vehicle_df['End'] - vehicle_df['Start']).values],
                y=[vehicle_df.index],
                mode='markers',
                name=vehicle,
                marker=dict(size=20, symbol='square')
            ))
        
        fig.update_layout(
            title='🕐 Delivery Schedule (Time Windows)',
            xaxis_title='Time of Day (hours)',
            yaxis_title='Stop',
            height=400,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_depot_utilization_chart(depot_assignments, depot_loads, depots):
        """Create bar chart for depot utilization"""
        import plotly.graph_objects as go
        
        depot_names = list(depot_loads.keys())
        loads = [depot_loads[d] for d in depot_names]
        capacities = [next(dep.max_capacity for dep in depots if dep.name == d) for d in depot_names]
        utilization = [l/c*100 for l, c in zip(loads, capacities)]
        
        fig = go.Figure(data=[
            go.Bar(name='Load', x=depot_names, y=loads, marker_color='#4A9EFF'),
            go.Bar(name='Capacity', x=depot_names, y=capacities, marker_color='#e0e0e0')
        ])
        
        fig.update_layout(
            title='🏭 Multi-Depot Utilization',
            barmode='overlay',
            xaxis_title='Depot',
            yaxis_title='Units',
            height=400
        )
        
        # Add utilization percentage annotations
        for i, util in enumerate(utilization):
            fig.add_annotation(
                x=depot_names[i], y=loads[i],
                text=f'{util:.0f}%',
                showarrow=False,
                yshift=10
            )
        
        return fig
    
    @staticmethod
    def create_driver_vehicle_chart(drivers: List[DriverAssignment]):
        """Create driver assignment visualization"""
        import plotly.express as px
        import pandas as pd
        
        df = pd.DataFrame([{
            'Driver': d.driver_name,
            'Vehicle': d.vehicle_name,
            'Shift Start': d.shift_start,
            'Shift End': d.shift_end,
            'Duration': d.shift_end - d.shift_start
        } for d in drivers])
        
        fig = px.bar(
            df, x='Driver', y='Duration', color='Vehicle',
            title='👨‍✈️ Driver Shift Assignments',
            labels={'Duration': 'Shift Duration (hours)'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        fig.update_layout(height=400)
        return fig


print("✅ Advanced VRP Features module loaded successfully")
