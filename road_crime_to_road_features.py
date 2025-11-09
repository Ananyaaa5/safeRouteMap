import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point

# Load cleaned crime data CSV
csv_path = "cleaned_chicago_crime_data.csv"  # Your cleaned raw crime dataset path

crime_df = pd.read_csv(csv_path)

# Ensure these columns exist
required_cols = {"Latitude", "Longitude", "Time_of_Day"}
if not required_cols.issubset(crime_df.columns):
    raise ValueError(f"CSV must contain columns: {required_cols}")

# Drop rows with missing coordinates
crime_df = crime_df.dropna(subset=["Latitude", "Longitude"])

# Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(crime_df["Longitude"], crime_df["Latitude"])]
crime_gdf = gpd.GeoDataFrame(crime_df, geometry=geometry, crs="EPSG:4326")

# Download Chicago road network graph
print("Downloading Chicago road network...")
G = ox.graph_from_place("Chicago, Illinois, USA", network_type="drive")

# Convert road network to GeoDataFrame (edges)
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

# Add unique road IDs
if "osmid" in edges.columns:
    edges["road_id"] = edges["osmid"].apply(lambda x: x if isinstance(x, int) else x[0])
else:
    edges["road_id"] = edges.index.astype(str)

edges = edges[["road_id", "geometry"]].reset_index(drop=True)

# Reproject to a projected CRS (UTM zone applicable to Chicago)
projected_crs = "EPSG:26971"  # NAD83 / Illinois East (example)
crime_proj = crime_gdf.to_crs(projected_crs)
edges_proj = edges.to_crs(projected_crs)

# Spatial join: find nearest road segment for each crime
print("Mapping crimes to nearest road segments...")
joined = gpd.sjoin_nearest(crime_proj, edges_proj, how="left", distance_col="dist_to_road")

# Aggregate crime counts per road
print("Aggregating crime counts per road segment...")
agg = joined.groupby("road_id").agg(
    total_crimes=("Time_of_Day", "count"),
    day_crimes=("Time_of_Day", lambda x: (x.str.lower() == "day").sum()),
    night_crimes=("Time_of_Day", lambda x: (x.str.lower() == "night").sum())
).reset_index()

# Merge aggregation with edges geometry
road_crime_gdf = edges.merge(agg, on="road_id", how="left").fillna(0)

# Save output files
road_crime_gdf.to_file("road_crime_features.geojson", driver="GeoJSON")
road_crime_gdf.drop(columns="geometry").to_csv("road_crime_features.csv", index=False)

print("✅ Done!")
print(f"Total roads: {len(road_crime_gdf)}")
print(f"Roads with crimes: {(road_crime_gdf['total_crimes']>0).sum()}")
print("Outputs saved as:")
print(" - road_crime_features.geojson")
print(" - road_crime_features.csv")
