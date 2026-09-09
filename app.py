"""
Blinkit Dark Store Placement Optimization — Gurgaon
=====================================================
Main Streamlit application that orchestrates all 6 layers:

  1. OpenStreetMap Data    → Fetch real Gurgaon POIs
  2. Demand Generation     → Synthesize demand points
  3. Demand Prediction     → Score demand with weighted model
  4. Location Optimization → K-Means + constraint solver
  5. Map Visualization     → Interactive Folium map
  6. Business Dashboard    → KPI cards + Plotly charts

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium

# ── Page Config (must be first Streamlit call) ────────────────────
st.set_page_config(
    page_title="Blinkit Dark Store Optimizer — Gurgaon",
    page_icon="🟡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import modules ────────────────────────────────────────────────
from modules.osm_data import fetch_osm_data, get_poi_summary, GURGAON_BOUNDS
from modules.demand_generator import generate_demand_points, get_demand_summary
from modules.demand_model import predict_demand, get_score_breakdown, DEFAULT_WEIGHTS
from modules.optimizer import optimize_locations, get_total_coverage, calculate_validation_metrics
from modules.map_viz import build_full_map
from modules import dashboard as dash


# ── Custom CSS for premium dark theme ─────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header styling ── */
    .main-header {
        background: linear-gradient(135deg, #0D1117 0%, #161B22 50%, #1a1a2e 100%);
        border: 1px solid rgba(247, 201, 72, 0.2);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #F7C948, #e94560, #9C27B0, #F7C948);
        background-size: 200% 100%;
        animation: shimmer 3s ease-in-out infinite;
    }
    @keyframes shimmer {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    .main-header h1 {
        color: #F7C948;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #8B949E;
        font-size: 0.95rem;
        margin: 4px 0 0;
    }

    /* ── KPI Cards ── */
    .kpi-card {
        background: linear-gradient(135deg, #161B22, #1a1a2e);
        border: 1px solid rgba(247, 201, 72, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .kpi-card:hover {
        border-color: rgba(247, 201, 72, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(247, 201, 72, 0.1);
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #F7C948;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* ── Pipeline step indicators ── */
    .pipeline-step {
        background: #161B22;
        border: 1px solid rgba(139, 148, 158, 0.2);
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.85rem;
    }
    .pipeline-step.active {
        border-color: #F7C948;
        background: rgba(247, 201, 72, 0.05);
    }
    .pipeline-step.done {
        border-color: #4CAF50;
        background: rgba(76, 175, 80, 0.05);
    }

    /* ── Section headers ── */
    .section-header {
        color: #E6EDF3;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 24px 0 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(247, 201, 72, 0.3);
    }

    /* ── Store table styling ── */
    .store-table {
        background: #161B22;
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Sidebar styling ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1117, #161B22);
    }

    /* ── Expander styling ── */
    .streamlit-expanderHeader {
        background: #161B22;
        border-radius: 8px;
    }

    /* ── Hide default streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Map container ── */
    .map-container {
        border: 1px solid rgba(247, 201, 72, 0.2);
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Slider labels ── */
    .stSlider label {
        color: #E6EDF3 !important;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🟡 Blinkit Dark Store Optimizer</h1>
    <p>AI-Powered Placement Optimization for Gurgaon — Powered by OpenStreetMap</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR — Controls
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Optimization Controls")
    st.markdown("---")

    # ── Store count slider ──
    st.markdown("### 🏪 Dark Store Settings")
    n_stores = st.slider(
        "Number of Dark Stores",
        min_value=3,
        max_value=15,
        value=8,
        step=1,
        help="How many dark stores to optimally place in Gurgaon",
    )

    service_radius = st.slider(
        "Service Radius (km)",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.5,
        help="Maximum delivery radius per store (Blinkit: ~3km for 10-min delivery)",
    )

    min_distance = st.slider(
        "Min Inter-Store Distance (km)",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="Minimum distance between any two dark stores",
    )

    st.markdown("---")

    # ── Demand model weights ──
    st.markdown("### 🧠 Demand Model Weights")
    st.caption("Adjust how each factor influences demand scoring")

    w_pop = st.slider("📊 Population Density", 0.0, 1.0, 0.30, 0.05)
    w_poi = st.slider("🏪 POI Proximity", 0.0, 1.0, 0.25, 0.05)
    w_acc = st.slider("🚇 Accessibility", 0.0, 1.0, 0.20, 0.05)
    w_comp = st.slider("🔴 Competition Gap", 0.0, 1.0, 0.15, 0.05)
    w_time = st.slider("⏰ Time Factor", 0.0, 1.0, 0.10, 0.05)

    weights = {
        "population_density": w_pop,
        "poi_proximity": w_poi,
        "accessibility": w_acc,
        "competition_penalty": w_comp,
        "time_factor": w_time,
    }

    st.markdown("---")

    # ── Time of day ──
    st.markdown("### ⏰ Time of Day")
    peak_hour = st.slider(
        "Analysis Hour",
        min_value=0,
        max_value=23,
        value=19,
        help="Hour of day to analyze (19 = 7 PM evening peak)",
        format="%d:00",
    )

    st.markdown("---")

    # ── Map settings ──
    st.markdown("### 🗺️ Map Settings")
    tile_style = st.selectbox(
        "Map Theme",
        ["Dark Matter", "OpenStreetMap", "Satellite"],
        index=0,
    )

    heatmap_radius = st.slider(
        "Heatmap Blob Size",
        min_value=8,
        max_value=30,
        value=15,
        help="Visual radius of heatmap blobs (pixels)",
    )


# ══════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════

# ── Layer 1: Fetch OSM Data ──
with st.spinner("🗺️ Fetching OpenStreetMap data for Gurgaon..."):
    poi_df = fetch_osm_data()
    poi_summary = get_poi_summary(poi_df)

# ── Layer 2: Generate Demand ──
with st.spinner("📊 Generating demand points..."):
    demand_df = generate_demand_points(poi_df)
    demand_summary = get_demand_summary(demand_df)

# ── Layer 3: Predict Demand ──
with st.spinner("🧠 Running demand prediction model..."):
    scored_df = predict_demand(demand_df, poi_df, weights=weights, peak_hour=peak_hour)
    score_breakdown = get_score_breakdown(scored_df)

# ── Layer 4: Optimize Locations ──
with st.spinner("📍 Optimizing dark store locations..."):
    optimal_df = optimize_locations(
        scored_df,
        n_stores=n_stores,
        min_distance_km=min_distance,
        service_radius_km=service_radius,
    )
    total_coverage = get_total_coverage(optimal_df, scored_df, service_radius)
    validation_metrics = calculate_validation_metrics(optimal_df, service_radius)

# ── Layer 5: Build Map ──
with st.spinner("🗺️ Building interactive map..."):
    folium_map = build_full_map(
        poi_df=poi_df,
        scored_df=scored_df,
        optimal_df=optimal_df,
        service_radius_km=service_radius,
        heatmap_radius=heatmap_radius,
        tile_style=tile_style,
    )


# ══════════════════════════════════════════════════════════════════
# KPI CARDS ROW
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📈 Key Metrics</div>', unsafe_allow_html=True)

kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)

with kpi_col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{total_coverage['total_stores']}</div>
        <div class="kpi-label">Dark Stores</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{total_coverage['coverage_pct']}%</div>
        <div class="kpi-label">Area Coverage</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color: #4CAF50;">{total_coverage['demand_coverage_pct']}%</div>
        <div class="kpi-label">Demand Coverage</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color: #2196F3;">{poi_summary['total_pois']}</div>
        <div class="kpi-label">POIs Analyzed</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color: #e94560;">{demand_summary['total_points']}</div>
        <div class="kpi-label">Demand Points</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col6:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color: #FF9800;">{total_coverage['avg_delivery_radius']} km</div>
        <div class="kpi-label">Avg Delivery Radius</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ACCURACY & VALIDATION METRICS
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🎯 Accuracy & Validation Metrics (vs. Real Blinkit Locations)</div>', unsafe_allow_html=True)

val_col1, val_col2, val_col3, val_col4 = st.columns(4)

with val_col1:
    st.markdown(f"""
    <div class="kpi-card" style="border: 1px solid rgba(76, 175, 80, 0.4); background: rgba(76, 175, 80, 0.03);">
        <div class="kpi-value" style="color: #4CAF50;">{validation_metrics['overall_match_score']}%</div>
        <div class="kpi-label">Overall Match Score</div>
    </div>
    """, unsafe_allow_html=True)

with val_col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color: #E6EDF3;">{validation_metrics['avg_dist_to_real_km']} km</div>
        <div class="kpi-label">Avg Distance to Real Store</div>
    </div>
    """, unsafe_allow_html=True)

with val_col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color: #2196F3;">{validation_metrics['real_stores_covered_pct']}%</div>
        <div class="kpi-label">Real Stores Covered (Recall)</div>
    </div>
    """, unsafe_allow_html=True)

with val_col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color: #FF9800;">{validation_metrics['optimized_precision_pct']}%</div>
        <div class="kpi-label">Location Precision (≤2km)</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MAP + STORE TABLE (side by side)
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🗺️ Optimization Map</div>', unsafe_allow_html=True)

map_col, table_col = st.columns([3, 1])

with map_col:
    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    st_folium(folium_map, width=None, height=550, returned_objects=[])
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("💡 Toggle layers using the control panel on the map. Click markers for detailed score breakdowns.")

with table_col:
    st.markdown("#### 📍 Optimal Locations")

    for _, store in optimal_df.iterrows():
        # Color based on rank
        rank = store["score_rank"]
        if rank <= 3:
            rank_color = "#F7C948"
            rank_emoji = "🥇" if rank == 1 else ("🥈" if rank == 2 else "🥉")
        else:
            rank_color = "#8B949E"
            rank_emoji = f"#{rank}"

        st.markdown(f"""
        <div style="
            background: #161B22;
            border: 1px solid rgba(247,201,72,0.15);
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 8px;
            border-left: 3px solid {rank_color};
        ">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#F7C948; font-weight:700;">{store['store_id']}</span>
                <span style="font-size:0.8rem;">{rank_emoji}</span>
            </div>
            <div style="font-size:0.75rem; color:#8B949E; margin-top:4px;">
                Coverage: <span style="color:#4CAF50;">{store['coverage_pct']}%</span> ·
                Demand: <span style="color:#2196F3;">{store['demand_covered']:.0f}</span>
            </div>
            <div style="font-size:0.65rem; color:#6E7681; margin-top:2px;">
                {store['lat']:.4f}°N, {store['lng']:.4f}°E
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)

# ── Row 1: Zone demand + POI distribution ──
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.plotly_chart(dash.demand_by_zone_chart(scored_df), use_container_width=True)

with chart_col2:
    st.plotly_chart(dash.poi_distribution_chart(poi_df), use_container_width=True)

# ── Row 2: Hourly profile + Radar ──
chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.plotly_chart(dash.hourly_demand_chart(scored_df), use_container_width=True)

with chart_col4:
    st.plotly_chart(dash.score_breakdown_radar(scored_df), use_container_width=True)

# ── Row 3: Store ranking + Coverage gauge ──
chart_col5, chart_col6 = st.columns(2)

with chart_col5:
    st.plotly_chart(dash.store_ranking_chart(optimal_df), use_container_width=True)

with chart_col6:
    st.plotly_chart(dash.coverage_gauge(total_coverage["demand_coverage_pct"]), use_container_width=True)

# ── Row 4: Scatter plot ──
st.plotly_chart(dash.demand_vs_coverage_scatter(optimal_df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# DATA EXPLORER
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔍 Data Explorer</div>', unsafe_allow_html=True)

with st.expander("📋 Optimal Store Locations — Full Data Table"):
    st.dataframe(
        optimal_df.style.format({
            "lat": "{:.6f}",
            "lng": "{:.6f}",
            "demand_covered": "{:.1f}",
            "coverage_pct": "{:.1f}%",
            "avg_distance_km": "{:.2f} km",
        }),
        use_container_width=True,
        height=300,
    )

with st.expander("📊 POI Data Summary"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**POI Count by Category**")
        st.dataframe(
            poi_df["category"].value_counts().reset_index().rename(
                columns={"index": "Category", "category": "Category", "count": "Count"}
            ),
            use_container_width=True,
        )
    with col_b:
        st.markdown("**Demand by Zone**")
        zone_data = scored_df.groupby("zone").agg(
            total_demand=("predicted_score", "sum"),
            avg_demand=("predicted_score", "mean"),
            count=("predicted_score", "count"),
        ).round(1).sort_values("total_demand", ascending=False)
        st.dataframe(zone_data, use_container_width=True)

with st.expander("🔧 Pipeline Architecture"):
    st.markdown("""
    ```
    ┌──────────────────────────┐
    │  1. OpenStreetMap Data   │  ← Overpass API fetches real Gurgaon POIs
    │     (osm_data.py)        │     Residential, Commercial, Transport, etc.
    └────────────┬─────────────┘
                 ▼
    ┌──────────────────────────┐
    │  2. Demand Generation    │  ← Synthetic demand points from POI locations
    │     (demand_generator.py)│     Zone-aware density, time profiles
    └────────────┬─────────────┘
                 ▼
    ┌──────────────────────────┐
    │  3. Demand Prediction    │  ← Weighted scoring model (5 factors)
    │     (demand_model.py)    │     Population, POI, Access, Competition, Time
    └────────────┬─────────────┘
                 ▼
    ┌──────────────────────────┐
    │  4. Location Optimizer   │  ← Weighted K-Means + constraint solver
    │     (optimizer.py)       │     Min distance, max radius, coverage
    └────────────┬─────────────┘
                 ▼
    ┌──────────────────────────┐
    │  5. Map Visualization    │  ← Folium map with 7 toggleable layers
    │     (map_viz.py)         │     Heatmap, markers, zones, competitors
    └────────────┬─────────────┘
                 ▼
    ┌──────────────────────────┐
    │  6. Business Dashboard   │  ← KPI cards, 7 Plotly charts
    │     (dashboard.py)       │     Real-time updates via sidebar sliders
    └──────────────────────────┘
    ```
    """)

with st.expander("🧮 Theoretical Framework & Mathematical Formulations"):
    st.markdown("### 1. The Facility Location Problem (p-Median Formulation)")
    st.markdown("""
    The placement of dark stores is modeled as a weighted variant of the classical **discrete p-Median problem**. The goal is to open exactly $p$ facilities (dark stores) from a set of candidate locations such that the sum of weighted distances from all demand points to their nearest facility is minimized.
    """)
    st.latex(r"""
    \min_{x, y} \sum_{i=1}^{N} \sum_{j=1}^{M} w_i \cdot d(c_i, f_j) \cdot x_{ij}
    """)
    st.markdown("""
    **Subject to the following constraints:**
    1. Each demand point is served by exactly one facility:
    """)
    st.latex(r"""
    \sum_{j=1}^{M} x_{ij} = 1 \quad \forall i \in \{1, \dots, N\}
    """)
    st.markdown("""
    2. A demand point can only be assigned to an open facility:
    """)
    st.latex(r"""
    x_{ij} \leq y_j \quad \forall i, j
    """)
    st.markdown("""
    3. Exactly $K$ facilities are opened (where $K$ is the slider input):
    """)
    st.latex(r"""
    \sum_{j=1}^{M} y_j = K
    """)
    st.markdown("""
    Where:
    - $N$ is the number of demand points ($N = {total_demand_points}$).
    - $M$ is the set of candidate locations.
    - $w_i$ is the demand score of point $i$ computed by the scoring model.
    - $d(c_i, f_j)$ is the distance between demand point $i$ and store $j$.
    - $x_{ij} \in \{0, 1\}$ is a binary variable indicating if point $i$ is served by store $j$.
    - $y_j \in \{0, 1\}$ is a binary variable indicating if store $j$ is opened.
    """.format(total_demand_points=total_coverage['total_demand_points']))

    st.markdown("### 2. Weighted K-Means Objective Function")
    st.markdown("""
    To solve the p-median optimization efficiently on the client side, we use a **Weighted K-Means Clustering** heuristic. The objective function minimizes the sum of weighted squared Euclidean distances from demand points to their closest cluster centroid (dark store):
    """)
    st.latex(r"""
    J(\mu_1, \dots, \mu_K) = \sum_{j=1}^{K} \sum_{i \in S_j} w_i \left\| x_i - \mu_j \right\|^2
    """)
    st.markdown("""
    Where:
    - $S_j$ is the set of demand points assigned to cluster $j$.
    - $\mu_j$ is the coordinate centroid of cluster $j$ (representing the dark store position).
    - $w_i$ is the weight of demand point $i$ (adjusting the centroid position toward high density).
    """)

    st.markdown("### 3. Catchment Partitioning via Voronoi Tessellation")
    st.markdown("""
    To determine the delivery boundaries (catchment areas) for each dark store on the map, we construct a **Voronoi Diagram** over the optimal coordinates. A Voronoi cell $R_k$ for a dark store $P_k$ is defined as the set of all coordinates in the plane closer to $P_k$ than to any other store:
    """)
    st.latex(r"""
    R_k = \{ x \in \mathbb{R}^2 \mid d(x, P_k) \leq d(x, P_j) \quad \forall j \neq k \}
    """)
    st.markdown("""
    Under Euclidean metrics, boundaries between adjacent cells form perpendicular bisectors of the segments connecting the store locations.
    """)


# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#6E7681; font-size:0.8rem; padding:10px;">
    🟡 <b>Blinkit Dark Store Optimizer</b> · Built with Python, Streamlit, Folium, scikit-learn & Plotly<br>
    Data Source: OpenStreetMap (Overpass API) · All tools are free & open-source<br>
    Gurgaon, Haryana, India · {n_stores} stores optimized · {total_coverage['coverage_pct']}% coverage
</div>
""".format(n_stores=total_coverage["total_stores"], total_coverage=total_coverage), unsafe_allow_html=True)
