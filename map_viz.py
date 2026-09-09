"""
Layer 5: Map Visualization
============================
Creates interactive Folium maps with multiple layers:
  - Dark-themed base map (CartoDB Dark Matter)
  - Demand heatmap overlay
  - Optimal store locations (pulsing markers)
  - Service radius circles
  - Voronoi-style catchment areas
  - Competitor markers
  - POI cluster markers

All rendered in pure Python with Folium (no JavaScript knowledge needed).
"""

import folium
from folium import plugins
import numpy as np
import pandas as pd
import branca.colormap as cm


# ── Map tile options ──────────────────────────────────────────────
TILE_LAYERS = {
    "Dark Matter": {
        "tiles": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        "attr": '&copy; <a href="https://carto.com/">CARTO</a>',
    },
    "OpenStreetMap": {
        "tiles": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attr": '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>',
    },
    "Satellite": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "&copy; Esri",
    },
}


def create_base_map(
    center_lat: float = 28.4595,
    center_lng: float = 77.0266,
    zoom: int = 12,
    tile_style: str = "Dark Matter",
) -> folium.Map:
    """
    Create the base Folium map centered on Gurgaon.

    Args:
        center_lat, center_lng: Map center coordinates
        zoom: Initial zoom level
        tile_style: One of 'Dark Matter', 'OpenStreetMap', 'Satellite'

    Returns:
        folium.Map object
    """
    tile_config = TILE_LAYERS.get(tile_style, TILE_LAYERS["Dark Matter"])

    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=zoom,
        tiles=tile_config["tiles"],
        attr=tile_config["attr"],
        control_scale=True,
    )

    # Add other tile layers as switchable options
    for name, config in TILE_LAYERS.items():
        if name != tile_style:
            folium.TileLayer(
                tiles=config["tiles"],
                attr=config["attr"],
                name=name,
            ).add_to(m)

    return m


def add_demand_heatmap(
    m: folium.Map,
    scored_df: pd.DataFrame,
    radius: int = 15,
    blur: int = 20,
    max_opacity: float = 0.8,
) -> folium.Map:
    """
    Add a heatmap layer showing demand intensity.

    Uses predicted_score as the heat weight.
    """
    heat_data = [
        [row["lat"], row["lng"], row["predicted_score"]]
        for _, row in scored_df.iterrows()
    ]

    gradient = {
        0.2: "#0D1117",
        0.4: "#1a1a6c",
        0.6: "#6a3093",
        0.8: "#e94560",
        1.0: "#F7C948",
    }

    heatmap_layer = plugins.HeatMap(
        heat_data,
        name="🔥 Demand Heatmap",
        radius=radius,
        blur=blur,
        max_zoom=15,
        gradient=gradient,
        max_val=scored_df["predicted_score"].max(),
    )

    heatmap_layer.add_to(m)
    return m


def add_optimal_locations(
    m: folium.Map,
    optimal_df: pd.DataFrame,
    service_radius_km: float = 3.0,
    show_radius: bool = True,
) -> folium.Map:
    """
    Add optimal dark store locations as prominent markers
    with service radius circles.
    """
    # Create a feature group for stores
    store_group = folium.FeatureGroup(name="📍 Optimal Dark Stores")

    for _, store in optimal_df.iterrows():
        # ── Custom HTML for marker icon ──
        icon_html = f"""
        <div style="
            background: linear-gradient(135deg, #F7C948, #e94560);
            border-radius: 50%;
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #0D1117;
            font-weight: bold;
            font-size: 12px;
            border: 3px solid #fff;
            box-shadow: 0 0 15px rgba(247,201,72,0.6);
        ">{store['store_id'][-2:]}</div>
        """

        icon = folium.DivIcon(
            html=icon_html,
            icon_size=(32, 32),
            icon_anchor=(16, 16),
        )

        # ── Popup with score breakdown ──
        popup_html = f"""
        <div style="
            font-family: 'Segoe UI', sans-serif;
            background: #161B22;
            color: #E6EDF3;
            padding: 12px;
            border-radius: 8px;
            min-width: 220px;
            border: 1px solid #F7C948;
        ">
            <h4 style="margin:0 0 8px; color:#F7C948;">
                📍 {store['store_id']}
            </h4>
            <table style="width:100%; font-size:12px; border-collapse: collapse;">
                <tr><td>📊 Demand Score</td><td style="text-align:right; color:#4CAF50;"><b>{store['demand_covered']}</b></td></tr>
                <tr><td>👥 Points Covered</td><td style="text-align:right;">{store['points_covered']}</td></tr>
                <tr><td>📈 Coverage</td><td style="text-align:right; color:#2196F3;">{store['coverage_pct']}%</td></tr>
                <tr><td>📏 Avg Radius</td><td style="text-align:right;">{store['avg_distance_km']} km</td></tr>
                <tr><td>🏆 Rank</td><td style="text-align:right; color:#F7C948;">#{store['score_rank']}</td></tr>
            </table>
            <div style="margin-top:8px; font-size:10px; color:#8B949E;">
                Lat: {store['lat']:.4f}, Lng: {store['lng']:.4f}
            </div>
        </div>
        """

        folium.Marker(
            location=[store["lat"], store["lng"]],
            icon=icon,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{store['store_id']} — Coverage: {store['coverage_pct']}%",
        ).add_to(store_group)

        # ── Service radius circle ──
        if show_radius:
            folium.Circle(
                location=[store["lat"], store["lng"]],
                radius=service_radius_km * 1000,  # meters
                color="#F7C948",
                fill=True,
                fill_color="#F7C948",
                fill_opacity=0.08,
                weight=1.5,
                dash_array="5 5",
            ).add_to(store_group)

    store_group.add_to(m)
    return m


def add_competitor_markers(m: folium.Map) -> folium.Map:
    """Add markers for existing competitor locations."""
    from modules.demand_model import COMPETITORS

    comp_group = folium.FeatureGroup(name="🔴 Competitors", show=False)

    for lat, lng, name in COMPETITORS:
        folium.CircleMarker(
            location=[lat, lng],
            radius=6,
            color="#F44336",
            fill=True,
            fill_color="#F44336",
            fill_opacity=0.7,
            popup=f"<b>{name}</b><br>Existing competitor",
            tooltip=name,
        ).add_to(comp_group)

    comp_group.add_to(m)
    return m


def add_metro_markers(m: folium.Map) -> folium.Map:
    """Add markers for metro stations (accessibility reference)."""
    from modules.demand_model import METRO_STATIONS

    metro_group = folium.FeatureGroup(name="🚇 Metro Stations", show=False)

    for lat, lng, name in METRO_STATIONS:
        folium.CircleMarker(
            location=[lat, lng],
            radius=5,
            color="#9C27B0",
            fill=True,
            fill_color="#9C27B0",
            fill_opacity=0.7,
            popup=f"<b>🚇 {name}</b>",
            tooltip=f"Metro: {name}",
        ).add_to(metro_group)

    metro_group.add_to(m)
    return m


def add_poi_markers(m: folium.Map, poi_df: pd.DataFrame, max_markers: int = 300) -> folium.Map:
    """
    Add POI markers with clustering (to avoid clutter).
    Shows a sample of POIs colored by category.
    """
    poi_group = plugins.MarkerCluster(name="📌 POI Locations", show=False)

    # Sample if too many
    if len(poi_df) > max_markers:
        sample_df = poi_df.sample(n=max_markers, random_state=42)
    else:
        sample_df = poi_df

    for _, poi in sample_df.iterrows():
        folium.CircleMarker(
            location=[poi["lat"], poi["lng"]],
            radius=3,
            color=poi["color"],
            fill=True,
            fill_color=poi["color"],
            fill_opacity=0.6,
            popup=f"{poi['icon']} {poi['name']}<br>Category: {poi['category']}",
            tooltip=f"{poi['icon']} {poi['category']}",
        ).add_to(poi_group)

    poi_group.add_to(m)
    return m


def add_zone_boundaries(m: folium.Map) -> folium.Map:
    """Add approximate zone boundaries as circles."""
    from modules.demand_generator import GURGAON_ZONES

    zone_group = folium.FeatureGroup(name="🗺️ Zone Boundaries", show=False)

    zone_colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
        "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
        "#BB8FCE", "#85C1E9", "#F0B27A",
    ]

    for i, (zone_name, zone_info) in enumerate(GURGAON_ZONES.items()):
        clat, clng = zone_info["center"]
        radius_m = zone_info["radius_deg"] * 111000  # Approx degrees to meters
        color = zone_colors[i % len(zone_colors)]

        folium.Circle(
            location=[clat, clng],
            radius=radius_m,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.05,
            weight=1,
            popup=f"""
                <b>{zone_name}</b><br>
                Type: {zone_info['zone_type']}<br>
                Density: {zone_info['density_multiplier']}x<br>
                {zone_info['description']}
            """,
            tooltip=zone_name,
        ).add_to(zone_group)

    zone_group.add_to(m)
    return m


def add_real_blinkit_markers(m: folium.Map) -> folium.Map:
    """Add markers for real Blinkit dark store locations in Gurgaon."""
    from modules.demand_model import REAL_BLINKIT_STORES

    blinkit_group = folium.FeatureGroup(name="🟡 Real Blinkit Stores", show=True)

    for lat, lng, name in REAL_BLINKIT_STORES:
        # Create a small yellow circle marker representing Blinkit brand
        folium.CircleMarker(
            location=[lat, lng],
            radius=6,
            color="#0D1117",
            weight=1,
            fill=True,
            fill_color="#F7C948",
            fill_opacity=0.9,
            popup=f"<b>{name}</b><br>Actual store hub",
            tooltip=name,
        ).add_to(blinkit_group)

    blinkit_group.add_to(m)
    return m


def add_real_blinkit_markers(m: folium.Map) -> folium.Map:
    """Add markers for real Blinkit dark store locations in Gurgaon."""
    from modules.demand_model import REAL_BLINKIT_STORES

    blinkit_group = folium.FeatureGroup(name="🟡 Real Blinkit Stores", show=True)

    for lat, lng, name in REAL_BLINKIT_STORES:
        # Create a small yellow circle marker representing Blinkit brand
        folium.CircleMarker(
            location=[lat, lng],
            radius=6,
            color="#0D1117",
            weight=1,
            fill=True,
            fill_color="#F7C948",
            fill_opacity=0.9,
            popup=f"<b>{name}</b><br>Actual store hub",
            tooltip=name,
        ).add_to(blinkit_group)

    blinkit_group.add_to(m)
    return m


def add_routing_layers(m: folium.Map, optimal_df: pd.DataFrame) -> folium.Map:
    """
    Fetch and draw OSRM road route lines from each optimized dark store
    to its closest metro station (representing the primary supply/logistics corridor).
    """
    from modules.demand_model import METRO_STATIONS
    from modules.routing import get_road_route

    route_group = folium.FeatureGroup(name="🛣️ OSRM Transit Routes", show=True)

    for _, store in optimal_df.iterrows():
        # Find closest metro station
        dists = [
            np.sqrt((store["lat"] - m_lat)**2 + (store["lng"] - m_lng)**2)
            for m_lat, m_lng, _ in METRO_STATIONS
        ]
        closest_idx = np.argmin(dists)
        target_lat, target_lng, target_name = METRO_STATIONS[closest_idx]

        # Get OSRM road route
        route_details = get_road_route(store["lat"], store["lng"], target_lat, target_lng)
        path_coords = route_details["coordinates"]

        # Determine tooltip label
        road_label = f"OSRM Road: {route_details['distance_km']} km ({route_details['duration_mins']} mins)" if route_details["is_real_road"] else f"Straight line: {route_details['distance_km']} km"

        # Draw line on map
        folium.PolyLine(
            locations=path_coords,
            color="#00BCD4",
            weight=3,
            opacity=0.6,
            dash_array="5, 10" if not route_details["is_real_road"] else None,
            tooltip=f"DS Route to {target_name} Metro<br>{road_label}",
        ).add_to(route_group)

    route_group.add_to(m)
    return m


def build_full_map(
    poi_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    optimal_df: pd.DataFrame,
    service_radius_km: float = 3.0,
    heatmap_radius: int = 15,
    tile_style: str = "Dark Matter",
) -> folium.Map:
    """
    Build the complete map with all layers.

    This is the main function called from the Streamlit app.

    Args:
        poi_df: POI data from Layer 1
        scored_df: Scored demand data from Layer 3
        optimal_df: Optimal locations from Layer 4
        service_radius_km: Service radius for stores
        heatmap_radius: Pixel radius for heatmap blobs
        tile_style: Base map tile style

    Returns:
        Complete folium.Map with all layers
    """
    m = create_base_map(tile_style=tile_style)

    # Add layers in order (bottom to top)
    m = add_zone_boundaries(m)
    m = add_demand_heatmap(m, scored_df, radius=heatmap_radius)
    m = add_poi_markers(m, poi_df)
    m = add_metro_markers(m)
    m = add_competitor_markers(m)
    m = add_real_blinkit_markers(m)
    m = add_routing_layers(m, optimal_df)
    m = add_optimal_locations(m, optimal_df, service_radius_km)

    # Add layer control toggle
    folium.LayerControl(collapsed=False).add_to(m)

    # Add fullscreen button
    plugins.Fullscreen(
        position="topright",
        title="Fullscreen",
        title_cancel="Exit Fullscreen",
    ).add_to(m)

    # Add minimap
    plugins.MiniMap(
        toggle_display=True,
        tile_layer=folium.TileLayer(
            tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
            attr="CARTO",
        ),
    ).add_to(m)

    return m

