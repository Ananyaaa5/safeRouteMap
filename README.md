# 🗺️ SafeRouteMap

A crime-aware navigation system that uses machine learning and geospatial data 
to recommend the **safest**, **fastest**, and **balanced** walking routes 
in Chicago, IL.

---

## 📌 What It Does

Most navigation apps optimize only for speed. SafeRouteMap overlays 
real Chicago crime data onto OpenStreetMap road networks to score 
every road segment for safety, then offers users two route options:

- 🟢 **Safest Route** — avoids high-crime road segments
- 🔴 **Fastest Route** — shortest distance regardless of safety

Each route displays estimated time, distance, and a crime risk percentage.

---

## 🛠️ Tech Stack

| Layer        | Technology                          |
|--------------|-------------------------------------|
| Backend      | Python, Flask, Flask-CORS           |
| ML Models    | scikit-learn (Random Forest, Logistic Regression) |
| Geospatial   | OSMnx, NetworkX, Folium/Leaflet.js  |
| Spatial Index| SciPy cKDTree                       |
| Geocoding    | Geopy (Nominatim)                   |
| Data         | Chicago Crime Dataset (CSV)         |
| Frontend     | HTML, CSS, JavaScript               |

---

## 🧠 How It Works

1. **Crime Data → Features** (`feature.py`)  
   Loads Chicago crime CSV, engineers time-based features (hour, day, 
   time slots), encodes crime types, calculates crime density per location.

2. **Model Training** (`train_random_forest.py`, `train.py`)  
   Trains a Random Forest and Logistic Regression classifier to predict 
   whether a road segment is safe or unsafe based on nearby crime history.

3. **Safety Scoring at Route Time** (`app.py`)  
   For every road edge in the graph:
   - Uses cKDTree to count crimes within a 0.003° radius
   - Feeds features into the Random Forest model
   - Combines both signals into a safety score per edge

4. **Route Generation**  
   Uses Dijkstra's algorithm (via NetworkX) with custom edge weights:
   - Fastest: weighted by road length
   - Safest: unsafe roads penalized with 1,000,000× multiplier
   - Balanced: 50/50 blend of length and safety weight

5. **Visualization** (`folium`)  
   Interactive map rendered with color-coded routes, start/end markers, 
   and an SOS emergency button.

---

## 📊 Model Results

| Model               | Accuracy | ROC-AUC | Safe F1 | Unsafe F1 |
|---------------------|----------|---------|---------|-----------|
| Logistic Regression | 63%      | 0.628   | 0.73    | 0.41      |
| **Random Forest**   | **72%**  | **0.776**| **0.78**| **0.64** |

**Random Forest was selected** for deployment due to:
- 9% higher accuracy than Logistic Regression
- ROC-AUC improved from 0.628 → 0.776
- Unsafe class F1 improved from 0.41 → 0.64 (critical for safety routing)
- Better at capturing non-linear spatial crime patterns

Full evaluation: see [`model_results.md`](model_results.md)

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- Chicago crime dataset CSV (`cleaned_chicago_crime_data.csv`)

### 1. Clone the repository
```bash
git clone https://github.com/Ananyaaa5/safeRouteMap.git
cd safeRouteMap
```

### 2. Install dependencies
```bash
pip install -r requirements_web.txt
```

### 3. Prepare the data and train the model
```bash
# Step 1: Engineer features from crime data
python feature.py

# Step 2: Train Random Forest (used in app)
python train_random_forest.py

# Step 3: (Optional) Train Logistic Regression for comparison
python train.py
```

### 4. Run the app
```bash
python app.py
```

Visit **http://localhost:5000** in your browser.

---

## 🚀 Usage

1. Enter a **From** address (Chicago, IL)
2. Enter a **To** address (Chicago, IL)
3. Click **Get Routes**
4. View the three color-coded routes on the map with risk scores

**Example addresses to test:**
From: 233 S Wacker Dr, Chicago, IL
To:   130 E Randolph St, Chicago, IL

---

## 📌 Key Design Decisions

**Why Random Forest over Logistic Regression?**  
Crime risk is spatially non-linear — high-crime zones cluster in complex 
patterns that logistic regression's linear boundary can't capture well. 
Random Forest handles this naturally and requires no feature scaling at 
inference time.

**Why cKDTree for spatial queries?**  
Brute-force crime proximity search across 34,000+ records per road edge 
would be O(n). cKDTree reduces this to O(log n), making real-time 
per-request graph scoring feasible.

**Why the 1,000,000× penalty for unsafe roads?**  
Dijkstra's algorithm finds the minimum cost path. To force genuine 
detours around unsafe segments (not just slight re-routing), unsafe 
roads need to be orders of magnitude more expensive than their length 
alone — otherwise the algorithm still picks them if they're short enough.

---

## ⚠️ Limitations

- Crime data is static (not real-time) — reflects historical patterns
- Currently scoped to Chicago, IL only
- Route calculation takes 20–40 seconds due to live OSM graph download
- Walking network only (no driving/cycling modes)

---

## 🔮 Future Improvements

- [ ] Real-time crime data integration via Chicago Data Portal API
- [ ] OSM graph caching to reduce response time from ~30s to ~3s
- [ ] Time-of-day input so safety scores reflect hour-specific crime patterns
- [ ] Extend to other cities with open crime datasets
- [ ] Mobile-responsive UI

---

## 📄 License

MIT License — free to use and modify.
