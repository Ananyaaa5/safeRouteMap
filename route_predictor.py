import osmnx as ox
import networkx as nx
import folium
import joblib
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from scipy.spatial import cKDTree

RAW_CRIME_CSV = "cleaned_chicago_crime_data.csv"
MODEL_PATH = "random_forest_model.pkl"
LAT_COL = "Latitude"
LON_COL = "Longitude"

model = joblib.load(MODEL_PATH)
crime_data = pd.read_csv(RAW_CRIME_CSV)[[LAT_COL, LON_COL]]

def build_crime_density_map(crime_data):
    coords = crime_data[[LAT_COL, LON_COL]].values
    return cKDTree(coords)

geolocator = Nominatim(user_agent="safe_route_app")

def get_coordinates(address):
    loc = geolocator.geocode(address)
    if not loc:
        raise ValueError(f"Address not found: {address}")
    return (loc.latitude, loc.longitude)

def get_graph(lat, lon, dist=2000):
    return ox.graph_from_point((lat, lon), dist=dist, network_type="drive")

def predict_safety(G, model, crime_tree, radius=0.003):
    edges = list(G.edges(data=True))
    df = pd.DataFrame([
        {
            'Latitude': (G.nodes[u]['y'] + G.nodes[v]['y']) / 2,
            'Longitude': (G.nodes[u]['x'] + G.nodes[v]['x']) / 2,
            'hour_of_day': 12,
            'day_of_week': 2,
            'nearby_crimes': len(crime_tree.query_ball_point(
                [(G.nodes[u]['y'] + G.nodes[v]['y']) / 2,
                 (G.nodes[u]['x'] + G.nodes[v]['x']) / 2], r=radius))
        }
        for u, v, data in edges
    ])
    for c in [
        'time_slot_morning', 'time_slot_afternoon', 'time_slot_evening', 'time_slot_night',
        'primary_type_encoded', 'time_of_day_encoded'
    ]:
        df[c] = 0
    preds = model.predict_proba(df[['Latitude', 'Longitude', 'hour_of_day', 'day_of_week',
                                    'time_slot_morning', 'time_slot_afternoon', 'time_slot_evening',
                                    'time_slot_night', 'primary_type_encoded',
                                    'time_of_day_encoded']])[:, 1]

    for i, (u, v, data) in enumerate(edges):
        data['nearby_crimes'] = df.loc[i, 'nearby_crimes']
        data['safety_score'] = 1 / (1 + preds[i] + df.loc[i, 'nearby_crimes'])
        data['unsafe'] = 1 if data['safety_score'] < 0.4 else 0
    return G

def get_routes(G, origin_point, destination_point):
    orig = ox.distance.nearest_nodes(G, origin_point[1], origin_point[0])
    dest = ox.distance.nearest_nodes(G, destination_point[1], destination_point[0])

    fastest = nx.shortest_path(G, orig, dest, weight='length')

    for u, v, d in G.edges(data=True):
        penalty = 200 if d.get('unsafe', 0) else 1
        d['safety_weight'] = d.get('length', 1) * penalty

    safest = nx.shortest_path(G, orig, dest, weight='safety_weight')
    return fastest, safest

def visualize_routes(G, fastest, safest, start, end):
    m = folium.Map(location=start, zoom_start=15)

    # Fastest route (red)
    folium.PolyLine(
        [(G.nodes[n]['y'], G.nodes[n]['x']) for n in fastest],
        color="red", weight=6, opacity=0.8,
        tooltip="Fastest Route"
    ).add_to(m)

    # Safest route (green)
    folium.PolyLine(
        [(G.nodes[n]['y'], G.nodes[n]['x']) for n in safest],
        color="green", weight=6, opacity=0.8,
        tooltip="Safest Route"
    ).add_to(m)

    # Start & end markers (aligned with node points)
    start_node = ox.distance.nearest_nodes(G, start[1], start[0])
    end_node = ox.distance.nearest_nodes(G, end[1], end[0])
    folium.Marker(
        [G.nodes[start_node]['y'], G.nodes[start_node]['x']],
        popup="Start Point", icon=folium.Icon(color='grey', icon='play')
    ).add_to(m)
    folium.Marker(
        [G.nodes[end_node]['y'], G.nodes[end_node]['x']],
        popup="Destination", icon=folium.Icon(color='red', icon='flag')
    ).add_to(m)

    return m

if __name__ == "__main__":
    f = input("Enter FROM address: ")
    t = input("Enter TO address: ")
    start = get_coordinates(f)
    end = get_coordinates(t)
    print("Fetching map...")
    G = get_graph(start[0], start[1])
    print("Building crime tree...")
    tree = build_crime_density_map(crime_data)
    print("Predicting road safety...")
    G = predict_safety(G, model, tree)
    print("Calculating routes...")
    fastest, safest = get_routes(G, start, end)
    print("Visualizing...")
    m = visualize_routes(G, fastest, safest, start, end)
    m.save("routes_map.html")
    print("✅ routes_map.html generated!")
