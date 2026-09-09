"""
Layer 1: OpenStreetMap Data Fetcher
===================================
Fetches real Point of Interest (POI) data for Gurgaon/Gurugram
from the Overpass API (free, no API key needed).

POI Categories fetched:
  - Residential: apartments, residential areas
  - Commercial: offices, malls, coworking spaces
  - Food/Grocery: restaurants, supermarkets, convenience stores
  - Transport: metro stations, bus stops
  - Education: schools, colleges, universities
  - Healthcare: hospitals, clinics, pharmacies
"""

import requests
import pandas as pd
import numpy as np
import streamlit as st
import time


# ── Gurgaon Bounding Box ──────────────────────────────────────────
GURGAON_BOUNDS = {
    "south": 28.38,
    "west": 76.94,
    "north": 28.52,
    "east": 77.12,
    "center_lat": 28.4595,
    "center_lng": 77.0266,
}


# ── POI category definitions with Overpass tags ────────────────────
POI_CATEGORIES = {
    "residential": {
        "tags": [
            '["building"="apartments"]',
            '["building"="residential"]',
            '["landuse"="residential"]',
        ],
        "color": "#4CAF50",
        "icon": "🏠",
        "demand_weight": 1.0,
    },
    "commercial": {
        "tags": [
            '["building"="commercial"]',
            '["building"="office"]',
            '["shop"="mall"]',
            '["amenity"="coworking_space"]',
        ],
        "color": "#2196F3",
        "icon": "🏢",
        "demand_weight": 0.8,
    },
    "food_grocery": {
        "tags": [
            '["amenity"="restaurant"]',
            '["shop"="supermarket"]',
            '["shop"="convenience"]',
            '["shop"="grocery"]',
            '["amenity"="fast_food"]',
        ],
        "color": "#FF9800",
        "icon": "🛒",
        "demand_weight": 0.6,
    },
    "transport": {
        "tags": [
            '["railway"="station"]',
            '["station"="subway"]',
            '["amenity"="bus_station"]',
            '["highway"="bus_stop"]',
        ],
        "color": "#9C27B0",
        "icon": "🚇",
        "demand_weight": 0.7,
    },
    "education": {
        "tags": [
            '["amenity"="school"]',
            '["amenity"="college"]',
            '["amenity"="university"]',
        ],
        "color": "#00BCD4",
        "icon": "🎓",
        "demand_weight": 0.5,
    },
    "healthcare": {
        "tags": [
            '["amenity"="hospital"]',
            '["amenity"="clinic"]',
            '["amenity"="pharmacy"]',
        ],
        "color": "#F44336",
        "icon": "🏥",
        "demand_weight": 0.4,
    },
}


def _build_overpass_query() -> str:
    """
    Build a single Overpass QL query that fetches all POI categories
    within the Gurgaon bounding box.
    """
    bbox = f"{GURGAON_BOUNDS['south']},{GURGAON_BOUNDS['west']},{GURGAON_BOUNDS['north']},{GURGAON_BOUNDS['east']}"

    # Build union of all tag queries
    tag_queries = []
    for category, config in POI_CATEGORIES.items():
        for tag in config["tags"]:
            tag_queries.append(f'  node{tag}({bbox});')
            tag_queries.append(f'  way{tag}({bbox});')

    union_body = "\n".join(tag_queries)

    query = f"""
[out:json][timeout:60];
(
{union_body}
);
out center body;
"""
    return query


def _classify_element(element: dict) -> str:
    """Classify an OSM element into one of our POI categories."""
    tags = element.get("tags", {})

    # Check each category's tags
    if tags.get("building") in ("apartments", "residential") or tags.get("landuse") == "residential":
        return "residential"
    elif tags.get("building") in ("commercial", "office") or tags.get("shop") == "mall" or tags.get("amenity") == "coworking_space":
        return "commercial"
    elif tags.get("amenity") in ("restaurant", "fast_food") or tags.get("shop") in ("supermarket", "convenience", "grocery"):
        return "food_grocery"
    elif tags.get("railway") == "station" or tags.get("station") == "subway" or tags.get("amenity") == "bus_station" or tags.get("highway") == "bus_stop":
        return "transport"
    elif tags.get("amenity") in ("school", "college", "university"):
        return "education"
    elif tags.get("amenity") in ("hospital", "clinic", "pharmacy"):
        return "healthcare"
    return "other"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_osm_data() -> pd.DataFrame:
    """
    Fetch POI data from Overpass API for Gurgaon.
    Results are cached for 1 hour to avoid re-fetching.

    Returns:
        DataFrame with columns: lat, lng, name, category, demand_weight, color, icon
    """
    query = _build_overpass_query()
    urls = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.osm.ch/api/interpreter",
    ]
    
    headers = {
        "User-Agent": "BlinkitDarkStoreOptimizer/1.0 (contact_dev@example.com)",
        "Accept": "application/json",
    }

    data = None
    last_error = ""
    for url in urls:
        try:
            response = requests.post(url, data={"data": query}, headers=headers, timeout=45)
            response.raise_for_status()
            data = response.json()
            break  # Success!
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            continue  # Try next mirror

    if data is None:
        st.warning(f"⚠️ Overpass API unavailable ({last_error}). Using fallback data.")
        return _generate_fallback_data()

    elements = data.get("elements", [])
    if len(elements) < 20:
        st.warning("⚠️ Limited OSM data received. Supplementing with synthetic points.")
        return _generate_fallback_data()

    records = []
    for el in elements:
        # Get coordinates (center for ways, direct for nodes)
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lng = el.get("lon") or el.get("center", {}).get("lon")

        if lat is None or lng is None:
            continue

        category = _classify_element(el)
        if category == "other":
            continue

        cat_config = POI_CATEGORIES[category]
        name = el.get("tags", {}).get("name", f"{cat_config['icon']} {category.replace('_', ' ').title()}")

        records.append({
            "lat": float(lat),
            "lng": float(lng),
            "name": name,
            "category": category,
            "demand_weight": cat_config["demand_weight"],
            "color": cat_config["color"],
            "icon": cat_config["icon"],
        })

    df = pd.DataFrame(records)

    # Filter to bounding box (safety check)
    df = df[
        (df["lat"] >= GURGAON_BOUNDS["south"])
        & (df["lat"] <= GURGAON_BOUNDS["north"])
        & (df["lng"] >= GURGAON_BOUNDS["west"])
        & (df["lng"] <= GURGAON_BOUNDS["east"])
    ].reset_index(drop=True)

    return df


def _generate_fallback_data() -> pd.DataFrame:
    """
    Generate realistic synthetic POI data for Gurgaon when
    the Overpass API is unavailable.

    Uses known Gurgaon landmarks and sector coordinates to
    create a believable POI distribution.
    """
    np.random.seed(42)

    # ── Known Gurgaon hotspots with approximate coordinates ──
    hotspots = [
        # (name, lat, lng, primary_category, spread_radius)
        ("DLF Cyber City", 28.4949, 77.0880, "commercial", 0.008),
        ("DLF Phase 1", 28.4725, 77.0920, "residential", 0.010),
        ("DLF Phase 2", 28.4780, 77.0850, "residential", 0.008),
        ("DLF Phase 3", 28.4820, 77.0780, "residential", 0.008),
        ("DLF Phase 5", 28.4620, 77.1010, "residential", 0.007),
        ("Sector 14 Market", 28.4690, 77.0380, "food_grocery", 0.005),
        ("Sector 29 Market", 28.4590, 77.0570, "food_grocery", 0.006),
        ("MG Road", 28.4790, 77.0720, "commercial", 0.009),
        ("Golf Course Road", 28.4480, 77.0680, "commercial", 0.012),
        ("Sohna Road", 28.4220, 77.0500, "commercial", 0.015),
        ("Huda City Centre Metro", 28.4594, 77.0724, "transport", 0.003),
        ("Guru Dronacharya Metro", 28.4822, 77.0956, "transport", 0.003),
        ("IFFCO Chowk Metro", 28.4728, 77.0568, "transport", 0.003),
        ("Sector 40-44", 28.4500, 77.0440, "residential", 0.010),
        ("Sector 45-50", 28.4430, 77.0580, "residential", 0.010),
        ("Sector 54-57", 28.4380, 77.0750, "residential", 0.009),
        ("Palam Vihar", 28.4890, 77.0200, "residential", 0.012),
        ("South City 1", 28.4450, 77.0650, "residential", 0.006),
        ("Nirvana Country", 28.4350, 77.0530, "residential", 0.006),
        ("Sector 82-85 Dwarka Exp", 28.4200, 76.9800, "residential", 0.015),
        ("Ambience Mall", 28.5040, 77.0960, "commercial", 0.005),
        ("Medanta Hospital", 28.4396, 77.0420, "healthcare", 0.004),
        ("Fortis Hospital", 28.4560, 77.0690, "healthcare", 0.004),
        ("Amity University", 28.4510, 77.0820, "education", 0.005),
        ("GD Goenka School", 28.4670, 77.0460, "education", 0.004),
    ]

    records = []

    for name, lat, lng, primary_cat, spread in hotspots:
        # Number of POIs around each hotspot
        n_pois = np.random.randint(15, 50)

        for i in range(n_pois):
            # Randomly pick category (70% primary, 30% other)
            if np.random.random() < 0.7:
                cat = primary_cat
            else:
                cat = np.random.choice(list(POI_CATEGORIES.keys()))

            cat_config = POI_CATEGORIES[cat]

            jitter_lat = np.random.normal(0, spread)
            jitter_lng = np.random.normal(0, spread)

            records.append({
                "lat": lat + jitter_lat,
                "lng": lng + jitter_lng,
                "name": f"{cat_config['icon']} {name} - {cat.replace('_', ' ').title()} #{i+1}",
                "category": cat,
                "demand_weight": cat_config["demand_weight"],
                "color": cat_config["color"],
                "icon": cat_config["icon"],
            })

    df = pd.DataFrame(records)

    # Clip to bounding box
    df = df[
        (df["lat"] >= GURGAON_BOUNDS["south"])
        & (df["lat"] <= GURGAON_BOUNDS["north"])
        & (df["lng"] >= GURGAON_BOUNDS["west"])
        & (df["lng"] <= GURGAON_BOUNDS["east"])
    ].reset_index(drop=True)

    return df


def get_poi_summary(df: pd.DataFrame) -> dict:
    """Return summary statistics about the POI data."""
    return {
        "total_pois": len(df),
        "by_category": df["category"].value_counts().to_dict(),
        "bounds": {
            "lat_min": df["lat"].min(),
            "lat_max": df["lat"].max(),
            "lng_min": df["lng"].min(),
            "lng_max": df["lng"].max(),
        },
    }
