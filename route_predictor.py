#route_predictor.py
import osmnx as ox
import networkx as nx
import folium
import joblib
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable
from scipy.spatial import cKDTree
import time

# -------- User customizable params --------
RAW_CRIME_CSV = "cleaned_chicago_crime_data.csv"
MODEL_PATH = "random_forest_model.pkl"
LAT_COL = "Latitude"
LON_COL = "Longitude"
# -------------------------------------------

model = joblib.load(MODEL_PATH)
crime_data = pd.read_csv(RAW_CRIME_CSV)
crime_data = crime_data[[LAT_COL, LON_COL]]

def build_crime_density_map(crime_data):
    coords = crime_data[[LAT_COL, LON_COL]].values
    return cKDTree(coords)

geolocator = Nominatim(user_agent="safe_route_app")

def get_coordinates(address, max_retries=5):
    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(address, timeout=15)
            if location:
                return (location.latitude, location.longitude)
            else:
                print(f"Address not found: {address}")
                return None
        except GeocoderUnavailable as e:
            print(f"Geocoder service unavailable (attempt {attempt+1}/{max_retries}). Retrying...")
            time.sleep(2)
        except Exception as e:
            print(f"Error geocoding address: {address}: {e}. Retrying...")
            time.sleep(2)
    raise SystemExit(f"ERROR: Could not geocode address '{address}' after {max_retries} retries.")

def get_graph(lat, lon, dist=3000):
    # Fetch walking network for pedestrians with smaller search radius
    return ox.graph_from_point((lat, lon), dist=dist, network_type="walk", simplify=False)

def predict_safety(G, model, crime_tree, radius=0.003):
    edge_features = []
    edges = list(G.edges(data=True))
    for u, v, data in edges:
        lat = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
        lon = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
        hour_of_day = 12
        day_of_week = 2
        nearby_crimes = len(crime_tree.query_ball_point([lat, lon], r=radius))
        edge_features.append({
            'Latitude': lat,
            'Longitude': lon,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'primary_type_encoded': 0,
            'time_of_day_encoded': 0,
            'time_slot_morning': 0,
            'time_slot_afternoon': 0,
            'time_slot_evening': 0,
            'time_slot_night': 0,
            'nearby_crimes': nearby_crimes
        })
    df = pd.DataFrame(edge_features)

    feature_cols = [
        'Latitude', 'Longitude', 'hour_of_day', 'day_of_week',
        'time_slot_morning', 'time_slot_afternoon', 'time_slot_evening', 'time_slot_night',
        'primary_type_encoded', 'time_of_day_encoded'
    ]

    preds = model.predict_proba(df[feature_cols])[:, 1]

    # Calculate safety scores for all edges
    all_safety_scores = []
    for i, (u, v, data) in enumerate(edges):
        # Combine nearby crimes and ML prediction into safety score
        # Weight nearby crimes more heavily for pedestrians
        safety_score = 1 / (1 + df.loc[i, 'nearby_crimes'] * 0.3 + preds[i])
        all_safety_scores.append(safety_score)
        data['safety_score'] = safety_score
        data['nearby_crimes'] = df.loc[i, 'nearby_crimes']
        data['ml_prediction'] = preds[i]

    # Use percentile-based threshold for marking unsafe edges
    # Mark bottom 35% as unsafe for better route differentiation
    threshold = np.percentile(all_safety_scores, 35)
    
    n_unsafe = 0
    n_safe = 0
    for i, (u, v, data) in enumerate(edges):
        data['unsafe'] = int(data['safety_score'] < threshold)
        if data['unsafe']:
            n_unsafe += 1
        else:
            n_safe += 1

    print(f"Unsafe edges: {n_unsafe} / {len(edges)} ({round(n_unsafe/len(edges)*100, 1)}%)")
    print(f"Safe edges: {n_safe} / {len(edges)}")
    print(f"Safety threshold: {threshold:.4f}")
    print("Safety score min/max:", f"{min(all_safety_scores):.4f}", f"{max(all_safety_scores):.4f}")
    print("Score percentiles:", np.percentile(all_safety_scores, [0, 10, 25, 50, 75, 90, 100]).round(4))

    return G

def get_routes(G, origin_point, destination_point):
    orig_node = ox.distance.nearest_nodes(G, origin_point[1], origin_point[0])
    dest_node = ox.distance.nearest_nodes(G, destination_point[1], destination_point[0])

    # Check if origin and destination are the same or too close
    if orig_node == dest_node:
        raise ValueError("Start and end locations are at the same point! Please choose addresses that are further apart (at least 1-2 miles).")

    orig_coords = (G.nodes[orig_node]['y'], G.nodes[orig_node]['x'])
    dest_coords = (G.nodes[dest_node]['y'], G.nodes[dest_node]['x'])

    # Calculate fastest route (shortest distance)
    shortest_route = nx.shortest_path(G, orig_node, dest_node, weight="length")

    # Calculate safest route with strong penalty for unsafe roads
    for u, v, data in G.edges(data=True):
        base_length = data.get("length", 1)
        safety_score = data.get("safety_score", 0.5)
        
        # Use a very strong multiplier to force route to avoid unsafe edges
        if data.get("unsafe", 0) == 1:
            # Make unsafe roads EXTREMELY costly - force major detours
            data["safety_weight"] = base_length * 500000
        else:
            # For safe roads, still penalize based on safety score
            # This creates gradients even among "safe" roads
            safety_penalty = (1.0 - safety_score) * 5 + 1.0  # Range: 1.0 to 6.0
            data["safety_weight"] = base_length * safety_penalty

    try:
        safest_route = nx.shortest_path(G, orig_node, dest_node, weight="safety_weight")
    except nx.NetworkXNoPath:
        # If no path found with strict safety, relax constraints
        print("Warning: No safe path found, using less strict safety criteria")
        for u, v, data in G.edges(data=True):
            base_length = data.get("length", 1)
            if data.get("unsafe", 0) == 1:
                data["safety_weight"] = base_length * 500  # Much weaker penalty
            else:
                data["safety_weight"] = base_length
        safest_route = nx.shortest_path(G, orig_node, dest_node, weight="safety_weight")

    def route_length(route):
        return sum(
            G.get_edge_data(route[i], route[i+1])[0]["length"]
            if isinstance(G.get_edge_data(route[i], route[i+1]), dict)
            else G.get_edge_data(route[i], route[i+1])["length"]
            for i in range(len(route)-1)
        )
    
    length_shortest = route_length(shortest_route)
    length_safest = route_length(safest_route)
    # Convert meters to kilometers for display
    distance_shortest_km = round(length_shortest / 1000, 2)
    distance_safest_km = round(length_safest / 1000, 2)
    # Walking speed: ~5 km/h = 83 meters/minute
    est_time_shortest = round(length_shortest / 83)
    est_time_safest = round(length_safest / 83)

    # Calculate route-specific risk percentages
    def calculate_route_risk(route):
        unsafe_count = 0
        total_count = 0
        for i in range(len(route)-1):
            edge_data = G.get_edge_data(route[i], route[i+1])
            if isinstance(edge_data, dict):
                edge_data = edge_data[0]
            if edge_data.get('unsafe', 0) == 1:
                unsafe_count += 1
            total_count += 1
        return round((unsafe_count / total_count) * 100, 2) if total_count > 0 else 0

    fastest_route_risk = calculate_route_risk(shortest_route)
    safest_route_risk = calculate_route_risk(safest_route)
    
    # Return all values as a tuple
    return (shortest_route, safest_route, orig_coords, dest_coords, 
            est_time_shortest, est_time_safest, 
            fastest_route_risk, safest_route_risk, 
            distance_shortest_km, distance_safest_km)

def visualize_routes(G, shortest_route, safest_route):
    m = ox.plot_route_folium(G, safest_route, color="green", weight=5, opacity=0.7)
    ox.plot_route_folium(G, shortest_route, color="red", weight=3, opacity=0.7, route_map=m)
    return m

if __name__ == "_main_":
    from_address = input("Enter FROM address: ")
    to_address = input("Enter TO address: ")
    from_coords = get_coordinates(from_address)
    to_coords = get_coordinates(to_address)
    
    # Check if geocoding was successful
    if from_coords is None or to_coords is None:
        print("\n❌ Error: Could not find one or both addresses.")
        print("Please use complete street addresses with city and state.")
        print("\nExample format:")
        print("  233 S Wacker Dr, Chicago, IL")
        print("  130 E Randolph St, Chicago, IL")
        exit(1)
    
    print("Fetching map data...")
    G = get_graph(from_coords[0], from_coords[1])
    print("Building crime density map...")
    crime_tree = build_crime_density_map(crime_data)
    print("Predicting safety for all roads...")
    G = predict_safety(G, model, crime_tree)
    
    # Calculate overall area risk percentage
    edges = list(G.edges(data=True))
    total_edges = len(edges)
    unsafe_edges = sum(data.get('unsafe', 0) for _, _, data in edges)
    overall_risk_percent = round((unsafe_edges / total_edges) * 100, 2)
    
    print("Finding routes...")
    result = get_routes(G, from_coords, to_coords)
    shortest_route, safest_route, orig_coords, dest_coords, est_time_shortest, est_time_safest, fastest_route_risk, safest_route_risk, distance_shortest_km, distance_safest_km = result
    
    print(f"\n{'='*50}")
    print(f"FASTEST ROUTE: {est_time_shortest} min | {distance_shortest_km} km | Risk: {fastest_route_risk}%")
    print(f"SAFEST ROUTE:  {est_time_safest} min | {distance_safest_km} km | Risk: {safest_route_risk}%")
    print(f"OVERALL AREA RISK: {overall_risk_percent}%")
    print(f"{'='*50}\n")
    
    print("Visualizing routes...")
    route_map = visualize_routes(G, shortest_route, safest_route)
    folium.Marker(orig_coords, popup="Start Location", icon=folium.Icon(color="green")).add_to(route_map)
    folium.Marker(dest_coords, popup="End Location", icon=folium.Icon(color="red")).add_to(route_map)

    # Overlay box showing route details
    box_html = f"""
    <div style="
        position: absolute;
        top: 20px; left: 20px;
        background: white;
        border-radius: 10px;
        border: 2px solid #333;
        padding: 12px 32px;
        font-size: 18px;
        font-weight: bold;
        color: #204080;
        z-index:9999;
        box-shadow: 2px 2px 10px #aaa;">
        <div style="color: #28a745; margin-bottom: 8px;">
            🟢 Safest Route: {est_time_safest} min | {distance_safest_km} km | Risk: {safest_route_risk}%
        </div>
        <div style="color: #dc3545; margin-bottom: 8px;">
            🔴 Fastest Route: {est_time_shortest} min | {distance_shortest_km} km | Risk: {fastest_route_risk}%
        </div>
        <div style="color: #6c757d; font-size: 16px; border-top: 2px solid #ddd; padding-top: 8px; margin-top: 8px;">
            Overall Area Risk: {overall_risk_percent}%
        </div>
    </div>
    """
    route_map.get_root().html.add_child(folium.Element(box_html))

    # SOS button (top right)
    sos_button_html = """
    <div onclick="alert('Emergency SOS Activated!')" style="
        position: absolute;
        top: 20px; right: 20px;
        background: #e3342f;
        border-radius: 15px;
        border: 2px solid #b71c1c;
        color: white;
        font-weight: bold;
        font-size: 24px;
        padding: 12px 36px;
        box-shadow: 1px 1px 7px #900;
        cursor: pointer;
        text-align:center;
        z-index: 9999;">
        SOS
    </div>
    """
    route_map.get_root().html.add_child(folium.Element(sos_button_html))

    route_map.save("routes_map.html")
    print("✅ Map saved as routes_map.html (open it in your browser!)")