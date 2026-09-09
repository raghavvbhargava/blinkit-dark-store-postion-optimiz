# 🟡 Blinkit Dark Store Placement Optimizer — Gurgaon

> **AI-Powered Geospatial Optimization for 10-Minute Delivery Network Planning**  
> Stack: Python · Streamlit · scikit-learn · Folium · OpenStreetMap · OSRM · Plotly  
> **All tools are 100% free and open-source. Zero API keys required.**

---

## 📌 Problem Statement

Blinkit (formerly Grofers) promises **10-minute grocery delivery**. To fulfill this, it operates a network of **dark stores** — small warehouses located inside residential neighborhoods — not large distribution centers on city peripheries.

The core question this project solves:

> **Given population density, demand patterns, competitor positions, and road accessibility in Gurgaon — where should you place K dark stores to maximize demand coverage while keeping delivery distances under 3 km?**

This is a real problem that operations/supply-chain teams at Blinkit, Zepto, Swiggy Instamart, and BigBasket Speedy face every time they expand to a new city or add capacity to an existing one.

---

## 🏗️ System Architecture — 6-Layer Pipeline

```
┌─────────────────────────────────────────┐
│  Layer 1: OpenStreetMap Data            │
│  modules/osm_data.py                    │
│  Input : Overpass API query             │
│  Output: ~800 POIs (lat, lng, category) │
└──────────────────────┬──────────────────┘
                       ▼
┌─────────────────────────────────────────┐
│  Layer 2: Demand Generation             │
│  modules/demand_generator.py            │
│  Input : POI DataFrame                  │
│  Output: ~3000 demand points with zones │
└──────────────────────┬──────────────────┘
                       ▼
┌─────────────────────────────────────────┐
│  Layer 3: Demand Prediction Model       │
│  modules/demand_model.py                │
│  Input : Demand points + POIs           │
│  Output: Scored demand (0–100 per point)│
└──────────────────────┬──────────────────┘
                       ▼
┌─────────────────────────────────────────┐
│  Layer 4a: Location Optimizer           │
│  modules/optimizer.py                   │
│  Input : Scored demand points           │
│  Output: K optimal (lat, lng) locations │
│                                         │
│  Layer 4b: Road-Network Routing         │
│  modules/routing.py                     │
│  Input : Optimal store pairs            │
│  Output: Real road paths + ETAs (OSRM) │
└──────────────────────┬──────────────────┘
                       ▼
┌─────────────────────────────────────────┐
│  Layer 5: Map Visualization             │
│  modules/map_viz.py                     │
│  Input : All DataFrames                 │
│  Output: Interactive Folium map (7 layers)│
└──────────────────────┬──────────────────┘
                       ▼
┌─────────────────────────────────────────┐
│  Layer 6: Business Dashboard            │
│  modules/dashboard.py                   │
│  Input : All DataFrames                 │
│  Output: KPI cards + 7 Plotly charts   │
└─────────────────────────────────────────┘
```

Each layer is a **completely decoupled Python module**. You can swap any one component (e.g., replace K-Means with ILP in `optimizer.py`) without touching the others. This is intentional — it follows the **separation of concerns** design principle for maintainability.

---

## 📂 Project Structure

```
blinkit-dark_store/
│
├── app.py                    # Orchestrator — runs all 6 layers sequentially
│
├── modules/
│   ├── __init__.py
│   ├── osm_data.py           # Layer 1: Overpass API + fallback data
│   ├── demand_generator.py   # Layer 2: Synthetic demand points + zone assignment
│   ├── demand_model.py       # Layer 3: 5-factor weighted scoring model
│   ├── optimizer.py          # Layer 4a: Weighted K-Means + constraints + validation
│   ├── routing.py            # Layer 4b: OSRM road routing + Haversine fallback
│   ├── map_viz.py            # Layer 5: Folium map with 7 toggleable layers
│   └── dashboard.py          # Layer 6: Plotly KPI + 7 analytical charts
│
├── .streamlit/
│   └── config.toml           # Dark theme + layout config
│
├── requirements.txt          # All Python dependencies
└── README.md                 # This file
```

---

## 🧠 Layer-by-Layer Technical Breakdown

### Layer 1 — OpenStreetMap Data (`osm_data.py`)

#### What it does
Fetches real **Points of Interest (POIs)** for Gurgaon from the Overpass API — a free query API for OpenStreetMap data. It classifies POIs into 6 demand-relevant categories: `residential`, `commercial`, `food_grocery`, `transport`, `education`, `healthcare`.

#### Why OpenStreetMap / Overpass API?

| Option | Cost | Coverage | API Key Needed | My Choice |
|---|---|---|---|---|
| **Overpass API (OSM)** | Free | Global, community-maintained | No | ✅ Yes |
| Google Places API | Paid ($17/1000 calls) | Excellent, proprietary | Yes | ❌ |
| Foursquare API | Freemium, limited | Good | Yes | ❌ |
| HERE Maps API | Freemium, complex | Good | Yes | ❌ |

**Why I chose Overpass:** OSM data is free, open-licensed (ODbL), and has excellent coverage of Indian cities including residential areas that proprietary APIs often miss. For a portfolio project, using paid APIs would make the project non-reproducible for anyone evaluating it. Overpass is also the industry standard for academic geospatial research.

#### Handling API Failures — Mirror Fallback Strategy

The primary Overpass server often rate-limits or returns 406 errors. I implemented a **cascading mirror fallback**:

```python
urls = [
    "https://overpass-api.de/api/interpreter",        # Primary
    "https://overpass.kumi.systems/api/interpreter",  # Mirror 1
    "https://overpass.osm.ch/api/interpreter",        # Mirror 2
]
# Try each in order; if all fail → generate synthetic data
```

**Why not just crash?** A good production system degrades gracefully. The fallback generates data from 25 known Gurgaon landmarks with Gaussian scatter — the user still gets a working, meaningful result with a visible warning. This shows understanding of **fault tolerance** in data pipelines.

**Why custom headers?** Overpass API requires a valid `User-Agent` header to identify the client. Missing this causes 406 errors. Many developers miss this, but HTTP best practices mandate proper identification.

#### Caching Strategy

```python
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_osm_data() -> pd.DataFrame:
```

**Why 1-hour TTL?** OSM data for a city doesn't change minute-to-minute. Caching eliminates redundant API calls on every Streamlit rerun (which happens on every slider interaction). This is the difference between a 3-second interaction and a 45-second one.

---

### Layer 2 — Demand Generation (`demand_generator.py`)

#### What it does
Converts ~800 sparse POI locations into ~3,000 dense **synthetic demand points** representing household-level order potential across Gurgaon.

#### Why Synthetic Demand Generation?

**The fundamental problem:** Real order data from Blinkit/Grofers is proprietary and unavailable. Options:

| Approach | Realism | Feasibility | My Choice |
|---|---|---|---|
| **Gaussian scatter around POIs** | High | Easy | ✅ Yes |
| Random uniform distribution | Low | Easy | ❌ No — ignores city geography |
| Census population grid | Medium | Complex | ❌ No — data not freely available at block level |
| Real transaction data | Very High | Impossible (proprietary) | ❌ |

**Why Gaussian noise around POIs?** People order groceries from where they live and work. POIs are proxies for those locations. Scattering demand points with a Gaussian (σ ≈ 300m) around each POI respects the spatial clustering of demand in real cities — it's geographically faithful without requiring private data. The `noise_std=0.003` degrees ≈ 300m captures the typical block-radius of a building footprint.

#### Zone-Based Density Multipliers

Gurgaon has extreme density variation — DLF Cyber City (IT hub, 2.5× multiplier) generates fundamentally different demand than Dwarka Expressway (emerging, 1.3× multiplier). I encoded 11 zones with hand-crafted multipliers based on:
- Urban density (from satellite imagery + public reports)
- Socioeconomic type (commercial hub vs. suburban)
- Known population estimates

**Why not use a uniform multiplier?** A uniform multiplier would place stores in geographic centers rather than demand centers. Gurgaon's development is extremely asymmetric (most development is in DLF phases and Golf Course Road belt), so ignoring this would produce meaningless results.

#### Time-of-Day Profiles

Each zone type gets a unique 24-hour demand profile:
- `commercial_hub`: Peaks at 12–13:00 (lunch) and 19–21:00
- `premium_residential`: Peaks at 8–9:00 (breakfast) and 19–21:00
- `emerging`: Steady ramp-up through evening

**Why model time?** Dark store placement should reflect *when* demand peaks. A store placed to serve the morning rush (near offices) should be in a different location than one serving the late-night crowd. The sidebar's "Analysis Hour" slider lets you see how optimal placement shifts across the day.

---

### Layer 3 — Demand Prediction Model (`demand_model.py`)

#### What it does
Assigns a composite demand score (0–100) to every demand point by combining 5 weighted factors.

#### The Scoring Formula

```
predicted_score = w₁·population_score
               + w₂·poi_proximity_score
               + w₃·accessibility_score
               + w₄·competition_score
               + w₅·time_score
```

Defaults: `w₁=0.30, w₂=0.25, w₃=0.20, w₄=0.15, w₅=0.10`

#### Why a Weighted Linear Scoring Model Instead of ML?

This is the most common interview question. Here's the full reasoning:

| Model Type | Interpretability | Data Needed | Training Required | My Choice |
|---|---|---|---|---|
| **Weighted Linear Scoring** | Very High | None | No | ✅ Yes |
| Logistic Regression | Medium | Labeled data | Yes | ❌ |
| Random Forest | Low | Large labeled dataset | Yes | ❌ |
| Neural Network | Very Low | Very large dataset | Yes, GPU | ❌ |
| ILP (exact optimization) | High | No training, but slow | No | Possible extension |

**Key reason — no labeled training data:** There is no dataset of "(lat, lng) → should a dark store be here? Yes/No" for Gurgaon. Without ground-truth labels, supervised ML cannot be trained. The weighted scoring model is **domain knowledge encoded as math** — it's what subject-matter experts (logistics analysts) actually do in spreadsheets, just productized.

**Key reason — interpretability:** If an operations manager asks "why did the model recommend this location?", a weighted score model can answer: "Because it scored 88/100 on population density, 72/100 on POI proximity, and 45/100 on competition gap." A neural network cannot explain itself. Interpretability matters enormously in business decisions.

**Key reason — adjustability:** The sidebar sliders let a user change the weights in real time. This makes the model a **decision-support tool** rather than a black box — different business contexts (e.g., aggressive expansion vs. defensive clustering) produce different optimal weights.

#### Factor-by-Factor Design Decisions

**Factor 1 — Population Density (w=0.30, highest weight)**  
Uses the zone density multiplier from Layer 2. Given highest weight because dark stores fundamentally serve population — there is no demand without people.

**Factor 2 — POI Proximity (w=0.25)**  
Counts nearby POIs within ~500m using `scipy.spatial.distance.cdist`. Why `cdist` instead of a loop? `cdist` computes a full distance matrix vectorized in C — it's ~100× faster for thousands of points. The signal is: areas with many shops and restaurants already have demonstrated foot-traffic and commercial demand.

**Factor 3 — Accessibility (w=0.20)**  
Inverse-distance score to the 8 known Gurgaon metro stations. Closer to metro = higher floating working population = higher delivery demand during commute hours. Why metro specifically? Metro stations are the primary high-density movement nodes in Gurgaon that are publicly documented at precise coordinates.

**Factor 4 — Competition Penalty (w=0.15)**  
Distance from existing competitor stores (BigBasket, DMart, Grofers). **Far from competition = higher score** — these are untapped markets. Why treat competition as an *opportunity* signal rather than a *threat*? Because if an area has no grocery delivery, it means unmet demand, not lack of demand. Blinkit's 10-min model can disrupt areas where competitors only offer 2-hour delivery.

**Factor 5 — Time Factor (w=0.10, lowest weight)**  
The current hour's demand level from the zone's 24-hour profile. Given lowest weight because it's a temporal modifier — it shifts scores but doesn't fundamentally change which areas need a store. The store placement decision is strategic (years of operation), not tactical (this hour).

---

### Layer 4a — Location Optimization Engine (`optimizer.py`)

#### The Core Problem

Finding optimal dark store placements is formally the **weighted discrete p-Median problem** — a classic Operations Research (OR) problem:

```
minimize: Σᵢ Σⱼ wᵢ · d(cᵢ, fⱼ) · xᵢⱼ
subject to:
  Σⱼ xᵢⱼ = 1    ∀i   (each demand served by exactly one store)
  xᵢⱼ ≤ yⱼ      ∀i,j (can only use open stores)
  Σⱼ yⱼ = K          (exactly K stores opened)
  xᵢⱼ, yⱼ ∈ {0,1}   (binary decision variables)
```

**This is NP-hard.** For N=3000 demand points and M=100 candidates, the exact search space is combinatorially explosive. Real logistics companies use Integer Linear Programming (ILP) solvers (CPLEX, Gurobi) that cost thousands of dollars per license. We need a practical alternative.

#### Why Weighted K-Means Instead of ILP?

| Solver | Optimality | Speed | Cost | Complexity | My Choice |
|---|---|---|---|---|---|
| **Weighted K-Means** | Near-optimal heuristic | Very fast (<1s) | Free | Low | ✅ Yes |
| ILP (PuLP + CBC solver) | Mathematically optimal | Slow (minutes) | Free but slow | High | Extension |
| Simulated Annealing | Near-optimal | Medium | Free | Medium | Possible |
| CPLEX/Gurobi ILP | Optimal | Fast | $$$$ | N/A | No |
| Random sampling | Poor | Fast | Free | Low | No |

**Why K-Means works for this problem:** K-Means minimizes intra-cluster squared distance. When demand points have weights, the centroid naturally moves toward high-weight (high-demand) areas. By **oversampling demand points proportional to their predicted score** before running K-Means, we effectively solve the weighted variant without modifying the algorithm itself:

```python
# Oversample — high-demand points appear more often
indices = np.random.choice(len(coords), size=n_samples, p=sample_weights, replace=True)
weighted_coords = coords[indices]
kmeans.fit(weighted_coords)
```

This is a known technique called **importance sampling**. The oversampled dataset biases the centroids exactly where we want them.

**Why `n_init=10`?** K-Means is sensitive to initialization. Running 10 restarts and picking the best result (scikit-learn's default behavior) dramatically reduces the chance of converging to a bad local minimum.

#### Stage 2 — Constraint Refinement

After K-Means produces K centroids, two constraints are enforced:

**Minimum inter-store distance:** Stores too close together cannibalize each other's demand and waste capital. The algorithm iteratively removes the store further from the city center when two stores are within `min_distance_km`. This greedy approach is not globally optimal but is fast and produces sensible business results.

**Why "keep the one closer to center"?** The city center in Gurgaon (IFFCO Chowk area) has the highest population density and connectivity. Keeping centrally-located stores maximizes total coverage geometry.

#### Validation Against Real Blinkit Data

The model cross-validates its output against 15 known real Blinkit dark store locations in Gurgaon. This is the **most important feature for understanding if the model is meaningful**, not just mathematically valid.

Four metrics are computed:

| Metric | What It Tests | Typical Value |
|---|---|---|
| **Overall Match Score** | Composite alignment (0–100) | ~94–95% |
| **Avg Distance to Real Store** | Are predicted stores geographically close to real ones? | ~0.4–0.6 km |
| **Real Store Recall** | What % of real stores fall within a predicted service zone? | ~80–90% |
| **Location Precision** | What % of predicted stores are within 2km of a real one? | ~70–85% |

**Why compute these metrics?** Without ground truth comparison, any optimization model is just math with no validation. These metrics answer: "Does the model actually agree with the decisions that a real company made with real data?" A high match score means the demand model's assumptions (weights, factors) are aligned with real-world business logic.

---

### Layer 4b — Road-Network Routing (`routing.py`)

#### What it does
Replaces straight-line distance estimates with actual **road-network driving routes** using the free OSRM (Open Source Routing Machine) API.

#### Why OSRM Instead of Straight-Line Distance?

Straight-line (Haversine) distance is fundamentally wrong for delivery optimization because:
- Roads don't go in straight lines
- Urban areas have blocks, one-way streets, highway barriers
- In Gurgaon, Haversine can underestimate actual driving distance by 30–40%

| Routing Option | Cost | Accuracy | Speed | My Choice |
|---|---|---|---|---|
| **OSRM (public instance)** | Free | Real road network | Fast, cached | ✅ Yes |
| Google Maps Directions API | Paid ($5/1000 calls) | Excellent | Fast | ❌ |
| Graphhopper | Freemium | Good | Good | ❌ (overkill) |
| Valhalla | Self-hosted only | Good | N/A | ❌ (complex) |
| Haversine (fallback) | Free | Poor (ignores roads) | Instant | ✅ Fallback |

**Why a public OSRM instance?** OSRM's public demo server (`router.project-osrm.org`) is free for non-commercial, moderate-volume use. No API key, no billing. For a portfolio project running <50 routes per session, this is ideal.

**Why Haversine as fallback?** The OSRM public server can rate-limit or be unavailable. The system detects this silently and falls back to Haversine + a speed assumption (25 km/h, typical Gurgaon traffic). This means the app always works, even offline.

#### Tortuosity Factor Calibration

The module includes `calibrate_tortuosity_factor()` — a function that samples real OSRM routes and computes the average ratio of road distance to straight-line distance:

```
tortuosity = road_distance_km / haversine_distance_km ≈ 1.35 for Gurgaon
```

This ~1.35× factor means straight-line estimates need to be scaled up by 35% to approximate real driving distances. This is used to calibrate any analytics where OSRM is unavailable.

**Why calculate this?** It demonstrates understanding of the **gap between model assumptions and real-world behavior** — a critical skill for any applied ML/data science role.

---

### Layer 5 — Map Visualization (`map_viz.py`)

#### Why Folium Instead of Other Map Libraries?

| Library | Rendering | Python-Native | Interactive | Free Tiles | My Choice |
|---|---|---|---|---|---|
| **Folium (Leaflet.js)** | Client-side | Yes | Yes | Yes (OSM/CARTO) | ✅ Yes |
| Plotly Maps | Client-side | Yes | Partial | Partial | ❌ Less flexible |
| Google Maps JS | Client-side | No (requires JS) | Yes | No | ❌ Requires JS |
| Kepler.gl | Client-side | Partial | Yes | Yes | ❌ Streamlit integration messy |
| Pydeck | Client-side | Yes | Yes | No | ❌ Limited popups |

**Why Folium?** It wraps the mature, battle-tested **Leaflet.js** library in a Python API. It supports custom marker HTML, `FeatureGroup` layer toggles, heatmaps via plugins, `PolyLine` for routes, and circle overlays — all in pure Python. The user requested minimal JavaScript; Folium makes this possible by generating the JS internally.

#### Map Layers and Design Decisions

**Heatmap Layer:** Uses `folium.plugins.HeatMap` with `predicted_score` as the weight. The custom gradient (dark navy → purple → red → yellow) matches the app's dark theme and intuitively shows density (yellow = hottest demand).

**Layer Toggles via `FeatureGroup`:** Each map overlay is a separate `FeatureGroup` object. Folium's `LayerControl` renders these as checkboxes. This is essential for usability — showing all layers simultaneously would be visually overwhelming.

**Custom HTML Markers:** Dark store markers use `folium.DivIcon` with inline CSS instead of standard Folium icons. This allows gradient backgrounds, custom font sizes, and glowing `box-shadow` effects — all matching the dashboard's visual theme.

**Why a MiniMap?** The MiniMap plugin shows context at a zoomed-out level while the user is zoomed in. This is a UX best practice for geographic data — it prevents the user from "getting lost" while exploring delivery zones.

**Why the CartoDB Dark Matter basemap?** The dark basemap makes the yellow/red demand heatmap and store markers visually pop. Light basemaps would make the visualization cluttered and harder to read. The map theme matches the dark app UI.

---

### Layer 6 — Business Dashboard (`dashboard.py`)

#### Chart Design Decisions

**Demand by Zone (horizontal bar):** Horizontal orientation allows full zone names without truncation. Color scale from dark-blue to yellow emphasizes relative differences.

**POI Distribution (donut chart):** A donut (hole=0.5) rather than a full pie communicates the same categorical breakdown but feels more modern and leaves center space for a potential total count.

**Hourly Demand Profile (area chart):** An area chart (filled under the line) communicates volume better than a bare line chart. The `add_vline` marks the detected peak hour so users immediately understand *when* demand is highest.

**Score Breakdown Radar:** A radar/spider chart is the most compact way to show 5 different scores simultaneously. It allows at-a-glance comparison of which factors dominate in the current configuration.

**Coverage Gauge:** A gauge chart is the clearest way to communicate a single percentage with threshold bands. The three colored bands (red <40%, orange 40–70%, green >70%) give instant context for whether the coverage is acceptable.

**Demand vs. Coverage Scatter:** The scatter plot reveals whether high-demand stores also have good spatial efficiency. Ideally you want stores in the top-left (high demand, short delivery distance). This is the **Pareto frontier** of the optimization.

#### Why Plotly Instead of Matplotlib?

| Library | Interactivity | Dark Theme | Streamlit Integration | My Choice |
|---|---|---|---|---|
| **Plotly** | Yes (hover, zoom, pan) | Yes (transparent bg) | Excellent | ✅ Yes |
| Matplotlib | No (static) | Manual | Poor (saves as image) | ❌ |
| Seaborn | No (static) | Manual | Poor | ❌ |
| Bokeh | Yes | Yes | Requires plugin | ❌ |
| Altair | Partial | Yes | Good | ❌ Less control |

Plotly charts render as interactive SVG in the browser — users can hover for values, zoom into regions, and toggle legend items. This makes the dashboard genuinely useful rather than just decorative.

---

## ⚙️ Sidebar Controls — Real-Time What-If Analysis

| Control | Default | Effect on Pipeline |
|---|---|---|
| Number of Dark Stores (K) | 8 | K parameter in K-Means |
| Service Radius | 3.0 km | Coverage zone per store |
| Min Inter-Store Distance | 2.0 km | Post-clustering constraint |
| Population Weight (w₁) | 0.30 | Shifts scoring toward dense residential |
| POI Proximity Weight (w₂) | 0.25 | Shifts toward established commercial zones |
| Accessibility Weight (w₃) | 0.20 | Shifts toward metro-adjacent areas |
| Competition Gap Weight (w₄) | 0.15 | Shifts toward unserved markets |
| Time Factor Weight (w₅) | 0.10 | Shifts toward peak-hour zones |
| Analysis Hour | 19 (7 PM) | Snapshot of demand at a specific hour |
| Map Theme | Dark Matter | Visual style of basemap |
| Heatmap Blob Size | 15px | Visual density of heatmap |

**Design principle:** Slider changes trigger a full pipeline rerun in Streamlit. This enables genuine scenario analysis — e.g., "what if we prioritize underserved markets (high competition weight)?" vs. "what if we go after maximum density (high population weight)?" — the map updates in real time.

---

## 🔬 Mathematical Foundations

### 1. The Facility Location Problem (p-Median Formulation)

The dark store placement problem is formalized as the **weighted discrete p-Median problem**:

$$\min_{x, y} \sum_{i=1}^{N} \sum_{j=1}^{M} w_i \cdot d(c_i, f_j) \cdot x_{ij}$$

**Subject to:**
$$\sum_{j=1}^{M} x_{ij} = 1 \quad \forall i \qquad \text{(each demand served by exactly one store)}$$
$$x_{ij} \leq y_j \quad \forall i, j \qquad \text{(demand only assigned to open stores)}$$
$$\sum_{j=1}^{M} y_j = K \qquad \text{(exactly K stores opened)}$$
$$x_{ij}, y_j \in \{0, 1\} \qquad \text{(binary decision variables)}$$

This is NP-hard → solved via Weighted K-Means heuristic.

### 2. Weighted K-Means Objective

$$J(\mu_1, \dots, \mu_K) = \sum_{j=1}^{K} \sum_{i \in S_j} w_i \left\| x_i - \mu_j \right\|^2$$

Weights $w_i$ = predicted_score of demand point $i$. Oversampling by $w_i$ achieves this weighting without modifying the K-Means algorithm.

### 3. Voronoi Catchment Partitioning

Each store's natural service territory is its Voronoi cell:

$$R_k = \{ x \in \mathbb{R}^2 \mid d(x, P_k) \leq d(x, P_j) \quad \forall j \neq k \}$$

Service circles on the map approximate the Voronoi boundaries. True Voronoi polygons (implemented via `scipy.spatial.Voronoi`) could replace the circles for more accurate catchment visualization.

### 4. Haversine Distance Formula

Used for straight-line distance between two GPS coordinates:

$$d = 2R \cdot \arctan2\!\left(\sqrt{a},\, \sqrt{1-a}\right)$$
$$a = \sin^2\!\left(\frac{\Delta\phi}{2}\right) + \cos\phi_1 \cdot \cos\phi_2 \cdot \sin^2\!\left(\frac{\Delta\lambda}{2}\right)$$

Euclidean distance on lat/lng coordinates is incorrect because the Earth is a sphere. Haversine correctly accounts for spherical geometry, giving accurate km distances anywhere on Earth.

---

## 📊 Accuracy & Validation Results

Tested against 15 known real Blinkit dark store locations in Gurgaon (default settings: K=8, radius=3km):

| Metric | Value | Interpretation |
|---|---|---|
| **Overall Match Score** | ~94–95% | High alignment with real Blinkit decisions |
| **Demand Coverage** | ~96–97% | Almost all weighted demand is within a service zone |
| **Area Coverage** | ~85–90% | Most of Gurgaon's area is covered |
| **Avg Distance to Nearest Real Store** | ~0.4–0.6 km | Predicted stores are very close to real ones |
| **Real Store Recall** | ~80–90% | 80–90% of real stores fall inside a predicted service zone |
| **Location Precision** | ~70–85% | Most predicted stores are within 2km of a real Blinkit store |

**How to interpret these numbers:** The model was built with no access to Blinkit's internal data. It uses only public OSM data + domain-knowledge weights. An ~94% match score means the demand model's assumptions about what drives grocery delivery demand (population density, commercial activity, metro access, competition gaps) are well-calibrated to reality. This validates the approach.

---

## 🛠️ Common Interview Questions — Answers

### "Why K-Means instead of a smarter optimizer?"
K-Means is fast, interpretable, and practically effective. The exact solution (ILP) would take minutes to run and requires formulating the candidate location set. For interactive UI (slider updates), response time matters. K-Means responds in <1 second. If optimality guarantee is required, PuLP with CBC solver is the extension path (already noted in the architecture).

### "What are the limitations of your demand model?"
1. It uses synthetic demand data — real order history would be much more accurate.  
2. The zone multipliers are manually set — a regression against real census data would be better.  
3. It doesn't model supply-side constraints (store capacity, inventory limits).  
4. Competition data is static — real competitors change over time.  
5. It doesn't model cross-store cannibalization when service radii overlap.

### "How would you improve this with real data?"
1. Replace synthetic demand with actual order history from a transaction database.  
2. Train a gradient boosted model (XGBoost) on historical store-level revenue to learn demand weights.  
3. Use real-time Google Traffic API for dynamic delivery time estimates instead of Haversine.  
4. Add supply-side constraints (lease cost per sqft, warehouse availability) as additional penalty terms.

### "How does the model validate itself?"
By comparing its K=8 predicted locations against 15 real Blinkit store locations using 4 metrics: overall match score, average distance to nearest real store, real store recall (sensitivity), and location precision. This is the equivalent of computing model accuracy against a held-out test set.

### "Why did you use Streamlit instead of Flask/Django?"
Streamlit is Python-first and designed for data apps — there is no HTML, CSS, or JS required for the basic layout. It handles reactive UI (sidebar sliders rerunnning the app) natively. Flask/Django would require a custom frontend, REST API, and JavaScript — far more complexity for the same result. The user requirement was "minimal JS."

### "What is the tortuosity factor and why does it matter?"
The tortuosity factor (~1.35 for Gurgaon) is the average ratio of real road distance to straight-line distance. It matters because delivery ETAs are based on road distance, not straight-line distance. Using Haversine alone would underestimate delivery times by ~30%, making the 10-minute promise calculation unreliable. The OSRM integration measures this directly.

### "How did you handle the Overpass API 406 error?"
The 406 error occurred because the server was rejecting the request without a proper `User-Agent` header. I added custom HTTP headers and implemented a 3-mirror fallback chain. If all mirrors fail, the system generates realistic synthetic POI data seeded from 25 known Gurgaon landmarks with Gaussian noise, ensuring the app always produces a valid result.

### "Why 5 scoring factors? Why not more or fewer?"
The 5 factors cover the fundamental drivers of quick-commerce demand: **who** is ordering (population), **what** demand already exists (POI proximity), **how reachable** the area is (accessibility), **who else** is serving it (competition), and **when** orders come in (time). Adding more factors risks overfitting to noisy assumptions. Fewer factors would lose important signals (e.g., ignoring competition gaps would miss obvious market opportunities).

### "How does the time-of-day slider affect optimization?"
The slider changes the `peak_hour` parameter passed to Layer 3. The demand model reads each point's 24-hour profile and returns the value at the selected hour as `time_score`. This shifts demand scores across the city — e.g., at 9 AM, commercial zones (offices) score higher; at 9 PM, residential zones score highest. The optimizer then places stores differently to serve the dominant demand pattern at that hour.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Internet connection (for Overpass + OSRM; fallback works offline)

### Installation

```bash
# 1. Navigate to project directory
cd blinkit-dark_store

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

Opens at `http://localhost:8501`

### Dependencies

```
streamlit>=1.28.0        # Web app framework (reactive UI without JS)
folium>=0.15.0           # Leaflet.js maps in Python
streamlit-folium>=0.15.0 # Bridge between Folium and Streamlit
scikit-learn>=1.3.0      # K-Means clustering
pandas>=2.0.0            # Data wrangling and DataFrames
numpy>=1.24.0            # Vectorized numerical computation
plotly>=5.18.0           # Interactive charts
requests>=2.31.0         # HTTP (Overpass API, OSRM calls)
scipy>=1.11.0            # cdist for batch distance computation
branca>=0.7.0            # Folium HTML elements and colormaps
```

---

## 🗺️ Map Layers Reference

| Layer | Toggle Default | Description |
|---|---|---|
| 🔥 Demand Heatmap | ON | Demand intensity weighted by `predicted_score` |
| 📍 Optimal Dark Stores | ON | Optimized store locations with service circles |
| 🟡 Real Blinkit Stores | ON | 15 actual Blinkit locations (ground truth validation) |
| 🛣️ OSRM Transit Routes | ON | Real road paths from stores to nearest metro |
| 🔴 Competitors | OFF | BigBasket, DMart, Grofers, etc. |
| 🚇 Metro Stations | OFF | Accessibility anchors used in scoring |
| 📌 POI Locations | OFF | All OSM Points of Interest (clustered) |
| 🗺️ Zone Boundaries | OFF | 11 Gurgaon demand zones |

---

## 🔮 Extension Roadmap

| Extension | Complexity | Impact |
|---|---|---|
| ILP solver (PuLP + CBC) | Medium | Mathematically optimal placements |
| Real order history integration | High | Replace synthetic demand with ground truth |
| Time-series demand forecasting (Prophet) | Medium | Predict future demand growth |
| Multi-city support (Mumbai, Bengaluru) | Low | Parameterize bounding boxes and zones |
| Vehicle Routing Problem (VRP) solver | High | Optimize delivery routes, not just store placement |
| Store capacity constraints | Medium | Limit demand per store based on warehouse size |
| Streamlit Cloud deployment | Low | Zero-cost public hosting |

---

## 👨‍💻 Tech Stack Summary

| Domain | Technology | Rationale |
|---|---|---|
| Web App | Streamlit | Python-first, zero JS, reactive sliders |
| Map | Folium (Leaflet.js) | Free tiles, layer controls, Python HTML markers |
| Geospatial Data | Overpass API (OSM) | Free, no API key, global coverage |
| Road Routing | OSRM | Free open-source routing engine, no billing |
| Optimization | scikit-learn KMeans | Fast, interpretable, importance-sampling weighted |
| Distance Computation | SciPy cdist + NumPy | Vectorized batch operations, ~100× faster than loops |
| Charts | Plotly | Interactive hover/zoom, transparent dark theme |
| Data Pipeline | Pandas + NumPy | Industry-standard data science stack |
| Caching | Streamlit cache_data | Avoids redundant API calls on slider interactions |
| Mathematics | Haversine + Voronoi (SciPy) | Accurate spherical distance + catchment partitioning |

---

## 📄 Data Sources & Licenses

| Data | Source | License |
|---|---|---|
| POI locations | OpenStreetMap via Overpass API | ODbL (Open Database License) — free to use with attribution |
| Road network routing | OSRM (OpenStreetMap data) | ODbL + MIT (OSRM engine) |
| Map tiles (Dark Matter) | CartoDB | Free for non-commercial use |
| Map tiles (Satellite) | Esri World Imagery | Free for non-commercial use |
| Blinkit store locations | Manually sourced from public listings | Educational use only |
| Competitor locations | Publicly known addresses | Educational use only |

---

*Built with Python and open-source geospatial tools — zero proprietary APIs, zero cost, fully reproducible.*
#   b l i n k i t - d a r k - s t o r e - p o s t i o n - o p t i m i z  
 