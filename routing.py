"""
Layer 4b: Real Road-Network Routing (OSRM Integration)
======================================================
Queries the free public Open Source Routing Machine (OSRM) API
to fetch actual road paths and driving distances.

Includes:
  - Cache-backed query requests to avoid rate limits
  - Automatic fallback to straight-line coordinates
  - Road vs. straight-line distance calibration (Tortuosity Factor)
"""

import requests
import streamlit as st
import numpy as np


@st.cache_data(ttl=3600, show_spinner=False)
def get_road_route(lat1: float, lng1: float, lat2: float, lng2: float) -> dict:
    """
    Fetch road route details between two points using OSRM.

    Args:
        lat1, lng1: Starting point coordinates
        lat2, lng2: Destination coordinates

    Returns:
        Dict containing:
            - coordinates: List of [lat, lng] points representing the path
            - distance_km: Road distance in kilometers
            - duration_mins: Driving duration in minutes
            - is_real_road: True if successfully fetched from OSRM, False if fallback
    """
    # Public OSRM API expects: {lng1},{lat1};{lng2},{lat2}
    url = f"https://router.project-osrm.org/route/v1/driving/{lng1},{lat1};{lng2},{lat2}?overview=full&geometries=geojson"
    headers = {
        "User-Agent": "BlinkitDarkStoreOptimizer/1.0 (contact_dev@example.com)",
        "Accept": "application/json",
    }

    try:
        response = requests.get(url, headers=headers, timeout=6)
        if response.status_code == 200:
            data = response.json()
            routes = data.get("routes", [])
            if routes:
                route = routes[0]
                geometry = route.get("geometry", {})
                coordinates = geometry.get("coordinates", [])
                
                # OSRM returns coordinates as [lng, lat], convert to [lat, lng]
                path = [[coord[1], coord[0]] for coord in coordinates]
                distance_m = route.get("distance", 0)
                duration_s = route.get("duration", 0)

                return {
                    "coordinates": path,
                    "distance_km": round(distance_m / 1000.0, 2),
                    "duration_mins": round(duration_s / 60.0, 1),
                    "is_real_road": True,
                }
    except Exception:
        pass

    # Fallback to straight-line calculation if OSRM fails/rate-limits
    # Haversine distance
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist_km = R * c

    # Estimate driving duration (approx 25 km/h average speed in Gurgaon traffic)
    duration_mins = (dist_km / 25.0) * 60.0

    return {
        "coordinates": [[lat1, lng1], [lat2, lng2]],
        "distance_km": round(dist_km, 2),
        "duration_mins": round(duration_mins, 1),
        "is_real_road": False,
    }


def calibrate_tortuosity_factor(sample_pairs: list) -> float:
    """
    Calculate the average ratio of actual road distance to straight-line distance
    (Tortuosity Factor) for Gurgaon based on sample pairs.
    """
    ratios = []
    for lat1, lng1, lat2, lng2 in sample_pairs:
        route = get_road_route(lat1, lng1, lat2, lng2)
        
        # Calculate straight line distance
        R = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlng = np.radians(lng2 - lng1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        straight_km = R * c

        if straight_km > 0.1 and route["is_real_road"]:
            ratios.append(route["distance_km"] / straight_km)

    return float(np.mean(ratios)) if ratios else 1.35
