import osmnx as ox
import networkx as nx
import folium
import joblib
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from scipy.spatial import cKDTree

# -------- User customizable params --------
RAW_CRIME_CSV = "cleaned_chicago_crime_data.csv"  # Raw crime points dataset
MODEL_PATH = "random_forest_saferoute.pkl"
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

def get_coordinates(address):
    location = geolocator.geocode(address)
    if location:
        return (location.latitude, location.longitude)
    else:
        raise ValueError(f"Address not found: {address}")

def get_graph(lat, lon, dist=10000):
    return ox.graph_from_point((lat, lon), dist=dist, network_type="drive")

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
            'latitude': lat,
            'longitude': lon,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'nearby_crimes': nearby_crimes
        })
    df = pd.DataFrame(edge_features)
    preds = model.predict_proba(df[['latitude', 'longitude', 'hour_of_day', 'day_of_week']])[:, 1]

    n_unsafe = 0
    n_safe = 0
    for i, (u, v, data) in enumerate(edges):
        data['safety_score'] = 1 / (1 + df.loc[i, 'nearby_crimes'] + preds[i])
        data['nearby_crimes'] = df.loc[i, 'nearby_crimes']
        # Original unsafe based on threshold
        data['unsafe'] = int(data['safety_score'] < 0.3)
        if data['unsafe']:
            n_unsafe += 1
        else:
            n_safe += 1

    print(f"Unsafe edges: {n_unsafe} / {len(edges)}")
    print(f"Safe edges: {n_safe} / {len(edges)}")

    all_scores = [data['safety_score'] for _, _, data in edges]
    print("Safety score min/max:", min(all_scores), max(all_scores))
    print("Score percentiles:", np.percentile(all_scores, [0, 10, 25, 50, 75, 90, 100]))

    # DEBUG: Force every 10th edge unsafe
    print("Forced unsafe patch: marking every 10th edge as unsafe")
    for i, (u, v, data) in enumerate(edges):
        if i % 10 == 0:
            data['unsafe'] = 1
        else:
            data['unsafe'] = 0

    return G

def get_routes(G, origin_point, destination_point):
    orig_node = ox.distance.nearest_nodes(G, origin_point[1], origin_point[0])
    dest_node = ox.distance.nearest_nodes(G, destination_point[1], destination_point[0])

    shortest_route = nx.shortest_path(G, orig_node, dest_node, weight="length")

    for u, v, data in G.edges(data=True):
        data["safety_weight"] = data.get("length", 1) * (100 if data.get("unsafe", 0) == 1 else 1)

    safest_route = nx.shortest_path(G, orig_node, dest_node, weight="safety_weight")

    def count_crimes(route):
        total = 0
        for i in range(len(route) - 1):
            edge_data = G.get_edge_data(route[i], route[i + 1])
            if isinstance(edge_data, dict):
                edge_data = edge_data[next(iter(edge_data))]
            total += edge_data.get('nearby_crimes', 0)
        return total

    crimes_on_shortest = count_crimes(shortest_route)
    crimes_on_safest = count_crimes(safest_route)
    print(f"Total crimes on fastest route: {crimes_on_shortest}")
    print(f"Total crimes on safest route: {crimes_on_safest}")

    return shortest_route, safest_route

def visualize_routes(G, shortest_route, safest_route):
    m = ox.plot_route_folium(G, safest_route, color="green", weight=5, opacity=0.7)
    ox.plot_route_folium(G, shortest_route, color="red", weight=3, opacity=0.7, route_map=m)
    return m

if __name__ == "__main__":
    from_address = input("Enter FROM address: ")
    to_address = input("Enter TO address: ")
    from_coords = get_coordinates(from_address)
    to_coords = get_coordinates(to_address)
    print("Fetching map data...")
    G = get_graph(from_coords[0], from_coords[1])
    print("Building crime density map...")
    crime_tree = build_crime_density_map(crime_data)
    print("Predicting safety for all roads...")
    G = predict_safety(G, model, crime_tree)
    print("Finding routes...")
    shortest_route, safest_route = get_routes(G, from_coords, to_coords)
    print("Visualizing routes...")
    route_map = visualize_routes(G, shortest_route, safest_route)
    route_map.save("routes_map.html")
    print("✅ Map saved as routes_map.html (open it in your browser!)")
