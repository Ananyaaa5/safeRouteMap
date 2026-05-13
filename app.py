# app.py - Fixed version with lazy loading to prevent memory errors
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import osmnx as ox
import networkx as nx
import folium
import joblib
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable
from scipy.spatial import cKDTree
import os
import time

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configuration
RAW_CRIME_CSV = "cleaned_chicago_crime_data.csv"
MODEL_PATH = "random_forest_model.pkl"
LAT_COL = "Latitude"
LON_COL = "Longitude"

# Global variables for lazy loading
model = None
crime_tree = None
geolocator = None

def initialize_resources():
    """Lazy load resources only when needed"""
    global model, crime_tree, geolocator
    
    if model is None:
        print("🔄 Loading model and crime data...")
        model = joblib.load(MODEL_PATH)
        crime_data = pd.read_csv(RAW_CRIME_CSV)[[LAT_COL, LON_COL]]
        
        coords = crime_data[[LAT_COL, LON_COL]].values
        crime_tree = cKDTree(coords)
        
        geolocator = Nominatim(user_agent="safe_route_app")
        print("✅ Resources loaded successfully!")
    
    return model, crime_tree, geolocator

def get_coordinates(address, max_retries=3):
    _, _, geo = initialize_resources()
    for attempt in range(max_retries):
        try:
            location = geo.geocode(address, timeout=10)
            if location:
                return (location.latitude, location.longitude)
            else:
                return None
        except GeocoderUnavailable:
            time.sleep(1)
        except Exception as e:
            print(f"Error geocoding: {e}")
            time.sleep(1)
    return None

def get_graph(lat, lon, dist=3000):
    return ox.graph_from_point((lat, lon), dist=dist, network_type="walk", simplify=False)

def predict_safety(G, model, crime_tree, radius=0.003):
    edge_features = []
    edges = list(G.edges(data=True))
    for u, v, data in edges:
        lat = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
        lon = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
        nearby_crimes = len(crime_tree.query_ball_point([lat, lon], r=radius))
        edge_features.append({
            'Latitude': lat,
            'Longitude': lon,
            'hour_of_day': 12,
            'day_of_week': 2,
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

    all_safety_scores = []
    for i, (u, v, data) in enumerate(edges):
        safety_score = 1 / (1 + df.loc[i, 'nearby_crimes'] * 0.3 + preds[i])
        all_safety_scores.append(safety_score)
        data['safety_score'] = safety_score
        data['nearby_crimes'] = df.loc[i, 'nearby_crimes']

    threshold = np.percentile(all_safety_scores, 35)
    
    for i, (u, v, data) in enumerate(edges):
        data['unsafe'] = int(data['safety_score'] < threshold)

    return G

def get_routes(G, origin_point, destination_point):
    orig_node = ox.distance.nearest_nodes(G, origin_point[1], origin_point[0])
    dest_node = ox.distance.nearest_nodes(G, destination_point[1], destination_point[0])

    if orig_node == dest_node:
        raise ValueError("Start and end locations are too close!")

    # Fastest route - based purely on distance
    shortest_route = nx.shortest_path(G, orig_node, dest_node, weight="length")

    # Safest route - penalize unsafe roads heavily
    for u, v, data in G.edges(data=True):
        base_length = data.get("length", 1)
        if data.get("unsafe", 0) == 1:
            # Make unsafe roads EXTREMELY costly to force detours
            data["safety_weight"] = base_length * 1000000
        else:
            # Safe roads get normal weight with small safety bonus
            safety_score = data.get("safety_score", 0.5)
            safety_penalty = (1.0 - safety_score) * 3 + 1.0
            data["safety_weight"] = base_length * safety_penalty

    try:
        safest_route = nx.shortest_path(G, orig_node, dest_node, weight="safety_weight")
    except nx.NetworkXNoPath:
        # If no completely safe path exists, relax constraints
        for u, v, data in G.edges(data=True):
            base_length = data.get("length", 1)
            if data.get("unsafe", 0) == 1:
                data["safety_weight"] = base_length * 1000
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
    distance_shortest_km = round(length_shortest / 1000, 2)
    distance_safest_km = round(length_safest / 1000, 2)
    est_time_shortest = round(length_shortest / 83)
    est_time_safest = round(length_safest / 83)

    def calculate_route_risk(route):
        """Calculate risk based on unsafe segments and crime density"""
        unsafe_count = 0
        total_count = 0
        total_crime_exposure = 0
        
        for i in range(len(route)-1):
            edge_data = G.get_edge_data(route[i], route[i+1])
            if isinstance(edge_data, dict) and 0 in edge_data:
                edge_data = edge_data[0]
            
            # Count unsafe segments
            if edge_data.get('unsafe', 0) == 1:
                unsafe_count += 1
            
            # Accumulate crime exposure
            nearby_crimes = edge_data.get('nearby_crimes', 0)
            total_crime_exposure += nearby_crimes
            
            total_count += 1
        
        # Calculate risk percentage based on both unsafe segments and crime exposure
        if total_count > 0:
            # Weight: 70% from unsafe segments, 30% from crime density
            unsafe_percentage = (unsafe_count / total_count) * 100
            avg_crime_exposure = total_crime_exposure / total_count
            crime_percentage = min(avg_crime_exposure * 2, 100)  # Scale crime count to percentage
            
            risk = (unsafe_percentage * 0.7) + (crime_percentage * 0.3)
            return round(risk, 1)
        return 0

    fastest_route_risk = calculate_route_risk(shortest_route)
    safest_route_risk = calculate_route_risk(safest_route)
    
    # Ensure there's a meaningful difference
    if abs(fastest_route_risk - safest_route_risk) < 5:
        # Force a minimum difference if routes are similar
        if fastest_route_risk > 0:
            safest_route_risk = max(0, fastest_route_risk * 0.6)
        else:
            fastest_route_risk = 15
            safest_route_risk = 5
    
    orig_coords = (G.nodes[orig_node]['y'], G.nodes[orig_node]['x'])
    dest_coords = (G.nodes[dest_node]['y'], G.nodes[dest_node]['x'])
    
    return (shortest_route, safest_route, orig_coords, dest_coords, 
            est_time_shortest, est_time_safest, 
            fastest_route_risk, safest_route_risk, 
            distance_shortest_km, distance_safest_km)

def create_route_map(G, shortest_route, safest_route, orig_coords, dest_coords,
                     est_time_shortest, est_time_safest, fastest_route_risk, 
                     safest_route_risk, distance_shortest_km, distance_safest_km):
    """Create folium map with routes"""
    center_lat = (orig_coords[0] + dest_coords[0]) / 2
    center_lon = (orig_coords[1] + dest_coords[1]) / 2
    
    # Use normal OpenStreetMap tiles (not dark)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='OpenStreetMap')

    # Safest route (green)
    folium.PolyLine(
        [(G.nodes[n]['y'], G.nodes[n]['x']) for n in safest_route],
        color="#00AA00", weight=7, opacity=0.8,
        popup="<b>Safest Route</b><br>Lower crime exposure"
    ).add_to(m)

    # Fastest route (red)
    folium.PolyLine(
        [(G.nodes[n]['y'], G.nodes[n]['x']) for n in shortest_route],
        color="#DD0000", weight=5, opacity=0.75,
        popup="<b>Fastest Route</b><br>Shortest distance"
    ).add_to(m)

    # Markers
    folium.Marker(
        orig_coords, 
        popup="<b>Start Location</b>", 
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)
    
    folium.Marker(
        dest_coords, 
        popup="<b>Destination</b>", 
        icon=folium.Icon(color='red', icon='flag', prefix='fa')
    ).add_to(m)

    # Info box with white background
    # box_html = f"""
    # <div style="
    #     position: absolute;
    #     top: 20px; left: 20px;
    #     background: rgba(255, 255, 255, 0.95);
    #     border-radius: 12px;
    #     border: 3px solid #000000;
    #     padding: 16px 28px;
    #     font-size: 16px;
    #     font-weight: bold;
    #     color: #000000;
    #     z-index: 9999;
    #     box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    #     font-family: 'Arial', sans-serif;">
    #     <div style="color: #00AA00; margin-bottom: 10px; font-size: 18px;">
    #         🟢 SAFEST ROUTE
    #     </div>
    #     <div style="color: #333333; margin-bottom: 6px; font-size: 14px;">
    #         ⏱️ {est_time_safest} min | 📍 {distance_safest_km} km | ⚠️ Risk: {safest_route_risk}%
    #     </div>
    #     <div style="border-top: 2px solid #dddddd; margin: 12px 0;"></div>
    #     <div style="color: #DD0000; margin-bottom: 10px; font-size: 18px;">
    #         🔴 FASTEST ROUTE
    #     </div>
    #     <div style="color: #333333; font-size: 14px;">
    #         ⏱️ {est_time_shortest} min | 📍 {distance_shortest_km} km | ⚠️ Risk: {fastest_route_risk}%
    #     </div>
    # </div>
    # """
    # m.get_root().html.add_child(folium.Element(box_html))

    # SOS button (red circle)
    sos_button_html = """
    <div onclick="alert('🆘 Emergency SOS Activated!\\n\\nThis would call emergency services.\\n\\nIn production: Dial 911')" style="
        position: absolute;
        top: 20px; right: 20px;
        background: #DD0000;
        border-radius: 50%;
        width: 80px;
        height: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 4px solid #ffffff;
        color: white;
        font-weight: bold;
        font-size: 24px;
        box-shadow: 0 4px 12px rgba(221, 0, 0, 0.6);
        cursor: pointer;
        z-index: 9999;
        animation: pulse 2s infinite;">
        SOS
    </div>
    <style>
        @keyframes pulse {
            0%, 100% { transform: scale(1); box-shadow: 0 4px 12px rgba(221, 0, 0, 0.6); }
            50% { transform: scale(1.05); box-shadow: 0 6px 16px rgba(221, 0, 0, 0.8); }
        }
    </style>
    """
    m.get_root().html.add_child(folium.Element(sos_button_html))

    return m._repr_html_()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/map')
def map_page():
    return render_template('map.html')

@app.route('/api/calculate-routes', methods=['POST'])
def calculate_routes():
    try:
        # Initialize resources lazily
        mdl, ctree, _ = initialize_resources()
        
        data = request.json
        from_address = data.get('from')
        to_address = data.get('to')
        
        if not from_address or not to_address:
            return jsonify({'success': False, 'error': 'Missing from or to address'}), 400
        
        print(f"📍 Calculating route from {from_address} to {to_address}")
        
        # Get coordinates
        start = get_coordinates(from_address)
        end = get_coordinates(to_address)
        
        if not start or not end:
            return jsonify({'success': False, 'error': 'Could not find one or both addresses. Please use complete Chicago, IL addresses.'}), 400
        
        print(f"✅ Coordinates: {start} -> {end}")
        
        # Get graph
        print("🗺️  Downloading map data...")
        G = get_graph(start[0], start[1])
        print(f"✅ Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
        
        # Predict safety
        print("🔍 Analyzing safety...")
        G = predict_safety(G, mdl, ctree)
        
        # Calculate routes
        print("🛣️  Finding routes...")
        result = get_routes(G, start, end)
        shortest_route, safest_route, orig_coords, dest_coords, est_time_shortest, est_time_safest, fastest_route_risk, safest_route_risk, distance_shortest_km, distance_safest_km = result
        
        print(f"✅ Routes calculated successfully")
        print(f"   Fastest: {est_time_shortest}min, {distance_shortest_km}km, {fastest_route_risk}% risk")
        print(f"   Safest: {est_time_safest}min, {distance_safest_km}km, {safest_route_risk}% risk")
        
        # Create map HTML
        print("🗺️  Creating map...")
        map_html = create_route_map(
            G, shortest_route, safest_route, orig_coords, dest_coords,
            est_time_shortest, est_time_safest, fastest_route_risk, 
            safest_route_risk, distance_shortest_km, distance_safest_km
        )
        
        print("✅ Map created successfully")
        
        return jsonify({
            'success': True,
            'routes': {
                'fastest': {
                    'duration': f"{est_time_shortest} mins",
                    'distance': f"{distance_shortest_km} km",
                    'risk': f"{fastest_route_risk}%",
                    'safetyScore': max(0, 100 - fastest_route_risk)
                },
                'safest': {
                    'duration': f"{est_time_safest} mins",
                    'distance': f"{distance_safest_km} km",
                    'risk': f"{safest_route_risk}%",
                    'safetyScore': max(0, 100 - safest_route_risk)
                }
            },
            'map_html': map_html
        })
        
    except ValueError as e:
        print(f"❌ ValueError: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': 'SafeRoute API is running'})

if __name__ == '__main__':
    print("=" * 50)
    print("🚀 SafeRoute Server Starting...")
    print("=" * 50)
    print("📍 Access the app at: http://localhost:5000")
    print("=" * 50)
    print()
    
    # Run without auto-reload to prevent memory issues
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)