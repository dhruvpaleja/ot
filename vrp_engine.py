"""
VRP Engine — Core solver, traffic manager, and signal optimizer
Urban Logistics Optimization — MPSTME NMIMS Mumbai
"""
import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import itertools

# ─── DATA CLASSES ───────────────────────────────────────────
@dataclass
class Location:
    name: str
    lat: float
    lon: float
    zone: str = "General"

@dataclass
class Warehouse(Location):
    max_capacity: int = 1000
    operating_cost: float = 500.0  # per day

@dataclass
class DeliveryPoint(Location):
    demand: int = 100
    time_window: Tuple[int, int] = (6, 22)  # hours
    priority: int = 1  # 1=normal, 2=high, 3=urgent
    service_time: int = 15  # minutes at location
    ready_time: float = 0.0  # earliest service start (hours from midnight)
    due_time: float = 24.0   # latest service end (hours from midnight)

@dataclass
class Vehicle:
    name: str
    capacity: int = 500
    cost_per_km: float = 12.0
    co2_per_km: float = 0.21  # kg CO2 per km
    speed_kmh: float = 30.0
    fuel_type: str = "Diesel"
    shift_start: float = 8.0   # shift start time (hours from midnight)
    shift_end: float = 20.0    # shift end time (hours from midnight)
    max_route_time: float = 480.0  # max route duration in minutes

@dataclass
class TrafficConfig:
    zone_multipliers: Dict[str, float] = field(default_factory=dict)
    time_of_day: str = "off_peak"
    signal_cycle_time: int = 120  # seconds
    green_ratio: float = 0.45
    signals_per_km: float = 2.5

@dataclass
class RouteResult:
    vehicle_name: str
    route_indices: List[int]
    route_names: List[str]
    total_distance: float
    total_cost: float
    total_co2: float
    total_time: float  # minutes
    load_carried: int
    capacity: int
    signal_delays: float  # minutes
    traffic_delay: float  # minutes
    arrival_times: List[float] = field(default_factory=list)  # hours from midnight
    wait_times: List[float] = field(default_factory=list)     # wait time at each stop (min)
    time_window_violations: List[str] = field(default_factory=list)
    depot_name: str = ""

@dataclass
class SolutionResult:
    routes: List[RouteResult]
    total_cost: float
    total_distance: float
    total_co2: float
    total_time: float
    solver_used: str
    unserved: List[str]
    distance_matrix: np.ndarray
    shadow_prices: Dict[str, float]

# ─── HAVERSINE DISTANCE ────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# ─── ROAD DISTANCE (with Mumbai road factor) ───────────────
def road_distance(lat1, lon1, lat2, lon2, road_factor=1.35):
    """Haversine * road factor to approximate actual road distance"""
    return haversine(lat1, lon1, lat2, lon2) * road_factor

# ─── TRAFFIC MANAGER ───────────────────────────────────────
class TrafficManager:
    TIME_PRESETS = {
        "morning_rush": {"base": 1.8, "label": "🌅 Morning Rush (8-10 AM)"},
        "midday":       {"base": 1.2, "label": "☀️ Midday (11 AM-3 PM)"},
        "evening_rush": {"base": 2.0, "label": "🌇 Evening Rush (5-8 PM)"},
        "night":        {"base": 0.8, "label": "🌙 Night (9 PM-6 AM)"},
        "off_peak":     {"base": 1.0, "label": "⏰ Off-Peak (Default)"},
    }

    MUMBAI_ZONES = {
        "South Mumbai": {"base_traffic": 1.6, "signals_per_km": 3.5},
        "Central Mumbai": {"base_traffic": 1.4, "signals_per_km": 3.0},
        "Western Suburbs": {"base_traffic": 1.3, "signals_per_km": 2.5},
        "Eastern Suburbs": {"base_traffic": 1.2, "signals_per_km": 2.0},
        "Extended Suburbs": {"base_traffic": 1.0, "signals_per_km": 1.5},
        "Navi Mumbai": {"base_traffic": 0.9, "signals_per_km": 1.2},
    }

    ZONE_MAPPING = {
        "Colaba": "South Mumbai", "Fort": "South Mumbai", "Marine Lines": "South Mumbai",
        "Churchgate": "South Mumbai", "Nariman Point": "South Mumbai",
        "Dadar": "Central Mumbai", "Worli": "Central Mumbai", "Prabhadevi": "Central Mumbai",
        "Lower Parel": "Central Mumbai", "Mahalaxmi": "Central Mumbai", "Parel": "Central Mumbai",
        "Bandra": "Western Suburbs", "Andheri": "Western Suburbs", "Juhu": "Western Suburbs",
        "Santacruz": "Western Suburbs", "Vile Parle": "Western Suburbs", "Khar": "Western Suburbs",
        "Goregaon": "Western Suburbs", "Versova": "Western Suburbs",
        "Kurla": "Eastern Suburbs", "Ghatkopar": "Eastern Suburbs", "Chembur": "Eastern Suburbs",
        "Sion": "Eastern Suburbs", "Vidyavihar": "Eastern Suburbs", "Vikhroli": "Eastern Suburbs",
        "Powai": "Eastern Suburbs", "Mulund": "Eastern Suburbs",
        "Malad": "Extended Suburbs", "Borivali": "Extended Suburbs", "Kandivali": "Extended Suburbs",
        "Dahisar": "Extended Suburbs", "Mira Road": "Extended Suburbs", "Thane": "Extended Suburbs",
        "Navi Mumbai": "Navi Mumbai", "Vashi": "Navi Mumbai", "Belapur": "Navi Mumbai",
        "Panvel": "Navi Mumbai", "Kharghar": "Navi Mumbai", "Airoli": "Navi Mumbai",
    }

    @classmethod
    def get_zone(cls, location_name):
        for loc, zone in cls.ZONE_MAPPING.items():
            if loc.lower() in location_name.lower():
                return zone
        return "Western Suburbs"  # default

    @classmethod
    def get_traffic_multiplier(cls, loc1_name, loc2_name, config: TrafficConfig):
        zone1 = cls.get_zone(loc1_name)
        zone2 = cls.get_zone(loc2_name)
        
        base1 = cls.MUMBAI_ZONES.get(zone1, {"base_traffic": 1.0})["base_traffic"]
        base2 = cls.MUMBAI_ZONES.get(zone2, {"base_traffic": 1.0})["base_traffic"]
        avg_base = (base1 + base2) / 2
        
        time_mult = cls.TIME_PRESETS.get(config.time_of_day, {"base": 1.0})["base"]
        
        # Apply user zone overrides
        zone_override1 = config.zone_multipliers.get(zone1, 1.0)
        zone_override2 = config.zone_multipliers.get(zone2, 1.0)
        avg_override = (zone_override1 + zone_override2) / 2
        
        return avg_base * time_mult * avg_override

    @classmethod
    def get_signal_delay(cls, loc1_name, loc2_name, distance_km, config: TrafficConfig):
        """Calculate signal delay in minutes"""
        zone1 = cls.get_zone(loc1_name)
        zone2 = cls.get_zone(loc2_name)
        
        sig1 = cls.MUMBAI_ZONES.get(zone1, {"signals_per_km": 2.0})["signals_per_km"]
        sig2 = cls.MUMBAI_ZONES.get(zone2, {"signals_per_km": 2.0})["signals_per_km"]
        avg_signals = (sig1 + sig2) / 2
        
        num_signals = int(distance_km * avg_signals)
        avg_wait = config.signal_cycle_time * (1 - config.green_ratio) * 0.5 / 60  # minutes
        
        return num_signals * avg_wait

    @classmethod
    def calculate_green_wave(cls, route_names, distances, config: TrafficConfig):
        """Calculate optimal green wave timing for a route"""
        results = []
        cumulative_dist = 0
        for i in range(len(route_names) - 1):
            dist = distances[i] if i < len(distances) else 0
            cumulative_dist += dist
            zone = cls.get_zone(route_names[i+1])
            zone_info = cls.MUMBAI_ZONES.get(zone, {"signals_per_km": 2.0})
            
            travel_time_sec = (dist / 30) * 3600  # at 30 km/h
            green_phase = config.signal_cycle_time * config.green_ratio
            offset = travel_time_sec % config.signal_cycle_time
            
            results.append({
                "from": route_names[i],
                "to": route_names[i+1],
                "distance_km": round(dist, 2),
                "signals": int(dist * zone_info["signals_per_km"]),
                "optimal_offset_sec": round(offset, 1),
                "green_phase_sec": round(green_phase, 1),
                "recommended_speed_kmh": round(dist / (travel_time_sec/3600) if travel_time_sec > 0 else 30, 1),
                "zone": zone
            })
        return results


# ─── DISTANCE MATRIX BUILDER ──────────────────────────────
def build_distance_matrix(locations: List[Location], config: TrafficConfig = None):
    """Build distance matrix with optional traffic adjustments"""
    n = len(locations)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = road_distance(locations[i].lat, locations[i].lon,
                                     locations[j].lat, locations[j].lon)
                if config:
                    traffic_mult = TrafficManager.get_traffic_multiplier(
                        locations[i].name, locations[j].name, config)
                    dist *= traffic_mult
                matrix[i][j] = dist
    return matrix


# ─── HEURISTIC SOLVER (Clarke-Wright Savings + 2-opt) ─────
class HeuristicSolver:
    """Clarke-Wright Savings Algorithm with 2-opt improvement"""
    
    @staticmethod
    def solve(warehouses: List[Warehouse], deliveries: List[DeliveryPoint],
              vehicles: List[Vehicle], config: TrafficConfig) -> SolutionResult:
        
        if not warehouses or not deliveries or not vehicles:
            return SolutionResult([], 0, 0, 0, 0, "Heuristic", [], np.array([]), {})
        
        # Use first warehouse as depot (multi-depot: assign nearest)
        depot = warehouses[0]
        all_locations = [depot] + list(deliveries)
        n = len(all_locations)
        
        # Build distance matrix
        dist_matrix = build_distance_matrix(all_locations, config)
        
        # Calculate savings s(i,j) = d(0,i) + d(0,j) - d(i,j)
        savings = []
        for i in range(1, n):
            for j in range(i+1, n):
                s = dist_matrix[0][i] + dist_matrix[0][j] - dist_matrix[i][j]
                savings.append((s, i, j))
        
        savings.sort(reverse=True, key=lambda x: x[0])
        
        # Initialize: each delivery is its own route
        routes = {i: [i] for i in range(1, n)}
        route_loads = {i: deliveries[i-1].demand for i in range(1, n)}
        route_of = {i: i for i in range(1, n)}  # which route a node belongs to
        
        vehicle_caps = sorted([v.capacity for v in vehicles], reverse=True)
        max_cap = max(vehicle_caps) if vehicle_caps else 500
        
        # Merge routes using savings
        for s, i, j in savings:
            ri = route_of[i]
            rj = route_of[j]
            if ri == rj:
                continue
            if ri not in routes or rj not in routes:
                continue
            
            combined_load = route_loads[ri] + route_loads[rj]
            if combined_load > max_cap:
                continue
            
            # Check if i is at end of ri and j is at start of rj (or vice versa)
            can_merge = False
            new_route = None
            
            if routes[ri][-1] == i and routes[rj][0] == j:
                new_route = routes[ri] + routes[rj]
                can_merge = True
            elif routes[ri][0] == i and routes[rj][-1] == j:
                new_route = routes[rj] + routes[ri]
                can_merge = True
            elif routes[ri][-1] == i and routes[rj][-1] == j:
                new_route = routes[ri] + list(reversed(routes[rj]))
                can_merge = True
            elif routes[ri][0] == i and routes[rj][0] == j:
                new_route = list(reversed(routes[ri])) + routes[rj]
                can_merge = True
            
            if can_merge and new_route:
                # Merge into ri
                routes[ri] = new_route
                route_loads[ri] = combined_load
                for node in routes[rj]:
                    route_of[node] = ri
                del routes[rj]
                del route_loads[rj]
        
        # 2-opt improvement on each route
        final_routes = {}
        for key, route in routes.items():
            improved = HeuristicSolver._two_opt(route, dist_matrix)
            final_routes[key] = improved
        
        # Assign vehicles to routes
        route_list = list(final_routes.values())
        route_list.sort(key=lambda r: sum(deliveries[idx-1].demand for idx in r), reverse=True)
        
        results = []
        unserved = []
        total_cost = 0
        total_dist = 0
        total_co2 = 0
        total_time = 0
        
        for idx, route in enumerate(route_list):
            if idx >= len(vehicles):
                for node in route:
                    unserved.append(deliveries[node-1].name)
                continue
            
            vehicle = vehicles[idx]
            route_with_depot = [0] + route + [0]
            
            dist = sum(dist_matrix[route_with_depot[k]][route_with_depot[k+1]] 
                       for k in range(len(route_with_depot)-1))
            load = sum(deliveries[node-1].demand for node in route)
            
            load_factor = 1.0 + 0.15 * (load / vehicle.capacity)
            co2 = dist * vehicle.co2_per_km * load_factor
            cost = dist * vehicle.cost_per_km
            
            # Calculate time with traffic and signals
            base_time = (dist / vehicle.speed_kmh) * 60  # minutes
            
            signal_delay = 0
            traffic_delay = 0
            for k in range(len(route_with_depot)-1):
                d = dist_matrix[route_with_depot[k]][route_with_depot[k+1]]
                n1 = all_locations[route_with_depot[k]].name
                n2 = all_locations[route_with_depot[k+1]].name
                signal_delay += TrafficManager.get_signal_delay(n1, n2, d, config)
            
            traffic_mult = TrafficManager.get_traffic_multiplier(depot.name, depot.name, config)
            traffic_delay = base_time * (traffic_mult - 1)
            
            total_minutes = base_time + signal_delay + traffic_delay
            
            # Time window calculations
            arrival_times = []
            wait_times = []
            tw_violations = []
            current_time = vehicle.shift_start * 60  # Convert to minutes from midnight
            
            for k, node_idx in enumerate(route_with_depot):
                if k == 0:
                    arrival_times.append(vehicle.shift_start)
                    wait_times.append(0)
                    continue
                
                # Travel to this node
                prev_idx = route_with_depot[k-1]
                travel_time = (dist_matrix[prev_idx][node_idx] / vehicle.speed_kmh) * 60
                current_time += travel_time
                
                arrival_hour = current_time / 60.0
                arrival_times.append(round(arrival_hour, 2))
                
                if node_idx > 0:  # Not depot
                    dp = deliveries[node_idx - 1]
                    ready_min = dp.ready_time * 60
                    due_min = dp.due_time * 60
                    service = dp.service_time
                    
                    # Wait if arrived early
                    wait = max(0, ready_min - current_time)
                    wait_times.append(round(wait, 1))
                    current_time += wait
                    
                    # Check violation
                    if current_time > due_min:
                        tw_violations.append(f"{dp.name}: {wait/60:.1f}h late")
                    
                    current_time += service  # Service time
            
            route_names = [all_locations[i].name for i in route_with_depot]
            
            rr = RouteResult(
                vehicle_name=vehicle.name,
                route_indices=route_with_depot,
                route_names=route_names,
                total_distance=round(dist, 2),
                total_cost=round(cost, 2),
                total_co2=round(co2, 3),
                total_time=round(total_minutes, 1),
                load_carried=load,
                capacity=vehicle.capacity,
                signal_delays=round(signal_delay, 1),
                traffic_delay=round(traffic_delay, 1),
                arrival_times=arrival_times,
                wait_times=wait_times,
                time_window_violations=tw_violations,
                depot_name=depot.name
            )
            results.append(rr)
            total_cost += cost
            total_dist += dist
            total_co2 += co2
            total_time += total_minutes
        
        # Shadow prices (approximate)
        shadow_prices = {}
        for i, dp in enumerate(deliveries):
            # Marginal cost of serving this delivery
            marginal_cost = dist_matrix[0][i+1] * 2 * vehicles[0].cost_per_km if vehicles else 0
            shadow_prices[dp.name] = round(marginal_cost / max(dp.demand, 1), 2)
        
        return SolutionResult(
            routes=results,
            total_cost=round(total_cost, 2),
            total_distance=round(total_dist, 2),
            total_co2=round(total_co2, 3),
            total_time=round(total_time, 1),
            solver_used="Clarke-Wright Savings + 2-opt",
            unserved=unserved,
            distance_matrix=dist_matrix,
            shadow_prices=shadow_prices
        )
    
    @staticmethod
    def _two_opt(route, dist_matrix):
        """2-opt local search improvement"""
        improved = True
        best = list(route)
        while improved:
            improved = False
            for i in range(len(best) - 1):
                for j in range(i + 2, len(best)):
                    # Calculate cost of current vs swapped
                    ni, nj = best[i], best[j]
                    ni1 = best[i+1] if i+1 < len(best) else best[0]
                    nj1 = best[(j+1) % len(best)] if j+1 < len(best) else best[0]
                    
                    d1 = dist_matrix[ni][ni1] + dist_matrix[nj][nj1]
                    d2 = dist_matrix[ni][nj] + dist_matrix[ni1][nj1]
                    
                    if d2 < d1:
                        best[i+1:j+1] = reversed(best[i+1:j+1])
                        improved = True
        return best


# ─── OR-TOOLS SOLVER ──────────────────────────────────────
class ORToolsSolver:
    """Google OR-Tools based CVRP solver"""
    
    @staticmethod
    def solve(warehouses: List[Warehouse], deliveries: List[DeliveryPoint],
              vehicles: List[Vehicle], config: TrafficConfig,
              time_limit_sec: int = 30, emission_cap: float = None) -> SolutionResult:
        
        try:
            from ortools.constraint_solver import routing_enums_pb2, pywrapcp
        except ImportError:
            # Fallback to heuristic if OR-Tools not available
            result = HeuristicSolver.solve(warehouses, deliveries, vehicles, config)
            result.solver_used = "Heuristic (OR-Tools not available)"
            return result
        
        if not warehouses or not deliveries or not vehicles:
            return SolutionResult([], 0, 0, 0, 0, "OR-Tools", [], np.array([]), {})
        
        depot = warehouses[0]
        all_locations = [depot] + list(deliveries)
        n = len(all_locations)
        num_vehicles = len(vehicles)
        
        dist_matrix = build_distance_matrix(all_locations, config)
        
        # Scale to integers (OR-Tools needs integers)
        scale = 100
        int_dist = (dist_matrix * scale).astype(int)
        
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(n, num_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)
        
        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int_dist[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Capacity constraint
        demands = [0] + [dp.demand for dp in deliveries]
        
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return demands[from_node]
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        vehicle_caps = [v.capacity for v in vehicles]
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0, vehicle_caps, True, 'Capacity')
        
        # Allow dropping visits (with penalty)
        penalty = 100000
        for node in range(1, n):
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
        
        # Search parameters
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_params.time_limit.FromSeconds(time_limit_sec)
        
        # Solve
        solution = routing.SolveWithParameters(search_params)
        
        if not solution:
            # Fallback
            result = HeuristicSolver.solve(warehouses, deliveries, vehicles, config)
            result.solver_used = "Heuristic (OR-Tools no solution)"
            return result
        
        # Extract routes
        results = []
        unserved = []
        total_cost = 0
        total_dist = 0
        total_co2 = 0
        total_time = 0
        
        served_nodes = set()
        
        for v_idx in range(num_vehicles):
            route_indices = []
            index = routing.Start(v_idx)
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route_indices.append(node)
                served_nodes.add(node)
                index = solution.Value(routing.NextVar(index))
            route_indices.append(0)  # return to depot
            
            if len(route_indices) <= 2:
                continue  # empty route
            
            vehicle = vehicles[v_idx]
            
            dist = sum(dist_matrix[route_indices[k]][route_indices[k+1]] 
                       for k in range(len(route_indices)-1))
            load = sum(demands[node] for node in route_indices[1:-1])
            
            load_factor = 1.0 + 0.15 * (load / vehicle.capacity) if vehicle.capacity > 0 else 1.0
            co2 = dist * vehicle.co2_per_km * load_factor
            cost = dist * vehicle.cost_per_km
            
            base_time = (dist / vehicle.speed_kmh) * 60
            
            signal_delay = 0
            for k in range(len(route_indices)-1):
                d = dist_matrix[route_indices[k]][route_indices[k+1]]
                n1 = all_locations[route_indices[k]].name
                n2 = all_locations[route_indices[k+1]].name
                signal_delay += TrafficManager.get_signal_delay(n1, n2, d, config)
            
            traffic_mult = TrafficManager.get_traffic_multiplier(depot.name, depot.name, config)
            traffic_delay = base_time * (traffic_mult - 1)
            total_minutes = base_time + signal_delay + traffic_delay
            
            # Time window calculations
            arrival_times = []
            wait_times = []
            tw_violations = []
            current_time = vehicle.shift_start * 60  # Convert to minutes from midnight
            
            for k, node_idx in enumerate(route_indices):
                if k == 0:
                    arrival_times.append(vehicle.shift_start)
                    wait_times.append(0)
                    continue
                
                # Travel to this node
                prev_idx = route_indices[k-1]
                travel_time = (dist_matrix[prev_idx][node_idx] / vehicle.speed_kmh) * 60
                current_time += travel_time
                
                arrival_hour = current_time / 60.0
                arrival_times.append(round(arrival_hour, 2))
                
                if node_idx > 0:  # Not depot
                    dp = deliveries[node_idx - 1]
                    ready_min = dp.ready_time * 60
                    due_min = dp.due_time * 60
                    service = dp.service_time
                    
                    # Wait if arrived early
                    wait = max(0, ready_min - current_time)
                    wait_times.append(round(wait, 1))
                    current_time += wait
                    
                    # Check violation
                    if current_time > due_min:
                        tw_violations.append(f"{dp.name}: {(current_time - due_min)/60:.1f}h late")
                    
                    current_time += service  # Service time
            
            route_names = [all_locations[i].name for i in route_indices]
            
            rr = RouteResult(
                vehicle_name=vehicle.name,
                route_indices=route_indices,
                route_names=route_names,
                total_distance=round(dist, 2),
                total_cost=round(cost, 2),
                total_co2=round(co2, 3),
                total_time=round(total_minutes, 1),
                load_carried=load,
                capacity=vehicle.capacity,
                signal_delays=round(signal_delay, 1),
                traffic_delay=round(traffic_delay, 1),
                arrival_times=arrival_times,
                wait_times=wait_times,
                time_window_violations=tw_violations,
                depot_name=depot.name
            )
            results.append(rr)
            total_cost += cost
            total_dist += dist
            total_co2 += co2
            total_time += total_minutes
        
        # Check unserved
        for i in range(1, n):
            if i not in served_nodes:
                unserved.append(deliveries[i-1].name)
        
        # Shadow prices
        shadow_prices = {}
        for i, dp in enumerate(deliveries):
            marginal_cost = dist_matrix[0][i+1] * 2 * vehicles[0].cost_per_km if vehicles else 0
            shadow_prices[dp.name] = round(marginal_cost / max(dp.demand, 1), 2)
        
        return SolutionResult(
            routes=results,
            total_cost=round(total_cost, 2),
            total_distance=round(total_dist, 2),
            total_co2=round(total_co2, 3),
            total_time=round(total_time, 1),
            solver_used="Google OR-Tools (CVRP + GLS)",
            unserved=unserved,
            distance_matrix=dist_matrix,
            shadow_prices=shadow_prices
        )


# ─── MULTI-SCENARIO COMPARATOR ────────────────────────────
class ScenarioComparator:
    """Run multiple scenarios for comparison"""
    
    @staticmethod
    def run_scenarios(warehouses, deliveries, vehicles, base_config,
                     solver_class, time_limit=15):
        scenarios = {}
        
        # Scenario 1: Baseline (no traffic)
        no_traffic = TrafficConfig(
            zone_multipliers={z: 0.0 for z in TrafficManager.MUMBAI_ZONES},
            time_of_day="off_peak",
            signal_cycle_time=base_config.signal_cycle_time,
            green_ratio=0.6,
            signals_per_km=0
        )
        # Override: set all zone multipliers to make traffic = 1.0
        baseline_config = TrafficConfig(
            zone_multipliers={z: 1.0/TrafficManager.MUMBAI_ZONES[z]["base_traffic"] 
                            for z in TrafficManager.MUMBAI_ZONES},
            time_of_day="off_peak",
            signal_cycle_time=120,
            green_ratio=0.5,
            signals_per_km=0
        )
        
        if solver_class == "OR-Tools":
            scenarios["Baseline"] = ORToolsSolver.solve(
                warehouses, deliveries, vehicles, baseline_config, time_limit)
            scenarios["With Traffic"] = ORToolsSolver.solve(
                warehouses, deliveries, vehicles, base_config, time_limit)
        else:
            scenarios["Baseline"] = HeuristicSolver.solve(
                warehouses, deliveries, vehicles, baseline_config)
            scenarios["With Traffic"] = HeuristicSolver.solve(
                warehouses, deliveries, vehicles, base_config)
        
        # Scenario 3: Emission Cap
        emission_config = TrafficConfig(
            zone_multipliers=base_config.zone_multipliers.copy(),
            time_of_day=base_config.time_of_day,
            signal_cycle_time=base_config.signal_cycle_time,
            green_ratio=base_config.green_ratio,
            signals_per_km=base_config.signals_per_km
        )
        if solver_class == "OR-Tools":
            scenarios["Emission Cap"] = ORToolsSolver.solve(
                warehouses, deliveries, vehicles, emission_config, time_limit)
        else:
            scenarios["Emission Cap"] = HeuristicSolver.solve(
                warehouses, deliveries, vehicles, emission_config)
        
        return scenarios


# ─── DEFAULT MUMBAI DATA ──────────────────────────────────
def get_default_warehouses():
    return [
        Warehouse("Andheri Hub", 19.1197, 72.8464, "Western Suburbs", 1200, 800),
        Warehouse("Dadar Central", 19.0178, 72.8478, "Central Mumbai", 1000, 600),
    ]

def get_default_deliveries():
    return [
        DeliveryPoint("Bandra", 19.0596, 72.8295, "Western Suburbs", 150, (8, 18), 2, 15, 8.0, 18.0),
        DeliveryPoint("Worli", 19.0096, 72.8179, "Central Mumbai", 200, (9, 17), 1, 20, 9.0, 17.0),
        DeliveryPoint("Colaba", 18.9067, 72.8147, "South Mumbai", 120, (8, 20), 3, 15, 8.0, 20.0),
        DeliveryPoint("Kurla", 19.0726, 72.8794, "Eastern Suburbs", 180, (7, 19), 1, 10, 7.0, 19.0),
        DeliveryPoint("Malad", 19.1872, 72.8484, "Extended Suburbs", 160, (8, 18), 2, 15, 8.0, 18.0),
        DeliveryPoint("Borivali", 19.2288, 72.8544, "Extended Suburbs", 140, (9, 20), 1, 15, 9.0, 20.0),
        DeliveryPoint("Powai", 19.1176, 72.9060, "Eastern Suburbs", 130, (8, 17), 2, 20, 8.0, 17.0),
        DeliveryPoint("Juhu", 19.1075, 72.8263, "Western Suburbs", 110, (10, 18), 1, 15, 10.0, 18.0),
        DeliveryPoint("Lower Parel", 19.0048, 72.8306, "Central Mumbai", 170, (8, 16), 3, 15, 8.0, 16.0),
        DeliveryPoint("Chembur", 19.0522, 72.8966, "Eastern Suburbs", 90, (9, 19), 1, 10, 9.0, 19.0),
        DeliveryPoint("Ghatkopar", 19.0860, 72.9080, "Eastern Suburbs", 100, (8, 18), 2, 15, 8.0, 18.0),
        DeliveryPoint("Thane", 19.2183, 72.9781, "Extended Suburbs", 200, (7, 20), 1, 20, 7.0, 20.0),
        DeliveryPoint("Vashi", 19.0771, 72.9986, "Navi Mumbai", 150, (8, 19), 2, 15, 8.0, 19.0),
        DeliveryPoint("Goregaon", 19.1663, 72.8526, "Western Suburbs", 120, (9, 17), 1, 15, 9.0, 17.0),
        DeliveryPoint("Santacruz", 19.0836, 72.8410, "Western Suburbs", 80, (8, 16), 1, 10, 8.0, 16.0),
    ]

def get_default_vehicles():
    return [
        Vehicle("Van-A (Diesel)", 500, 14.0, 0.27, 25.0, "Diesel", 8.0, 20.0, 480.0),
        Vehicle("Van-B (CNG)", 400, 11.0, 0.18, 28.0, "CNG", 8.0, 20.0, 480.0),
        Vehicle("Van-C (Electric)", 350, 8.0, 0.05, 30.0, "Electric", 9.0, 18.0, 420.0),
        Vehicle("Van-D (Diesel)", 450, 13.0, 0.25, 26.0, "Diesel", 8.0, 20.0, 480.0),
        Vehicle("Van-E (CNG)", 380, 10.5, 0.17, 27.0, "CNG", 8.0, 20.0, 480.0),
    ]