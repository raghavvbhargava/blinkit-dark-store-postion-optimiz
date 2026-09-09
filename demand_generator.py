"""
Layer 2: Demand Generation
===========================
Synthesizes realistic household-level demand points from POI data.

How it works:
  1. Takes POI locations from Layer 1
  2. Applies population density multipliers for known Gurgaon zones
  3. Scatters synthetic demand points around each POI with Gaussian noise
  4. Assigns demand intensity based on POI type and zone
  5. Adds time-of-day demand profiles

Output: ~2000-5000 demand points with lat, lng, weight, zone, time_profile
"""

import pandas as pd
import numpy as np
import streamlit as st


# ── Gurgaon Zone Definitions ──────────────────────────────────────
# Each zone has a center, radius, and population density multiplier
GURGAON_ZONES = {
    "DLF Cyber City": {
        "center": (28.4949, 77.0880),
        "radius_deg": 0.012,
        "density_multiplier": 2.5,
        "zone_type": "commercial_hub",
        "description": "IT/Business hub with massive floating population",
    },
    "DLF Phase 1-3": {
        "center": (28.4775, 77.0850),
        "radius_deg": 0.015,
        "density_multiplier": 2.0,
        "zone_type": "premium_residential",
        "description": "Established premium residential area",
    },
    "Golf Course Road": {
        "center": (28.4480, 77.0680),
        "radius_deg": 0.018,
        "density_multiplier": 2.2,
        "zone_type": "premium_residential",
        "description": "Luxury apartments and high-income households",
    },
    "Sector 40-50": {
        "center": (28.4480, 77.0500),
        "radius_deg": 0.015,
        "density_multiplier": 1.8,
        "zone_type": "mid_residential",
        "description": "Dense middle-class residential sectors",
    },
    "Sector 54-57": {
        "center": (28.4380, 77.0750),
        "radius_deg": 0.012,
        "density_multiplier": 1.9,
        "zone_type": "premium_residential",
        "description": "Golf Course Extension Road premium area",
    },
    "Sohna Road": {
        "center": (28.4220, 77.0500),
        "radius_deg": 0.020,
        "density_multiplier": 1.5,
        "zone_type": "emerging",
        "description": "Rapidly developing corridor with new projects",
    },
    "Palam Vihar": {
        "center": (28.4890, 77.0200),
        "radius_deg": 0.015,
        "density_multiplier": 1.6,
        "zone_type": "suburban_residential",
        "description": "Established suburban residential area",
    },
    "MG Road": {
        "center": (28.4790, 77.0720),
        "radius_deg": 0.008,
        "density_multiplier": 2.0,
        "zone_type": "commercial_hub",
        "description": "Major commercial and entertainment strip",
    },
    "Dwarka Expressway": {
        "center": (28.4200, 76.9800),
        "radius_deg": 0.025,
        "density_multiplier": 1.3,
        "zone_type": "emerging",
        "description": "New development corridor with upcoming projects",
    },
    "Sector 14-17": {
        "center": (28.4690, 77.0380),
        "radius_deg": 0.010,
        "density_multiplier": 1.7,
        "zone_type": "old_city",
        "description": "Old Gurgaon with dense population",
    },
    "Sector 29": {
        "center": (28.4590, 77.0570),
        "radius_deg": 0.006,
        "density_multiplier": 1.8,
        "zone_type": "commercial_hub",
        "description": "Leisure and dining hub",
    },
}

# ── Time-of-day demand profiles ──────────────────────────────────
TIME_PROFILES = {
    "commercial_hub": {
        "description": "Peak during work hours + lunch",
        "profile": [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.9, 1.0,
                     1.0, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.7, 0.8, 0.9,
                     0.7, 0.5, 0.3, 0.2],
    },
    "premium_residential": {
        "description": "Morning + evening peaks",
        "profile": [0.1, 0.05, 0.05, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 0.7,
                     0.5, 0.6, 0.5, 0.4, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0,
                     0.9, 0.7, 0.4, 0.2],
    },
    "mid_residential": {
        "description": "Evening dominant",
        "profile": [0.1, 0.05, 0.05, 0.05, 0.1, 0.2, 0.4, 0.7, 0.9, 0.6,
                     0.5, 0.5, 0.5, 0.4, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0,
                     0.9, 0.7, 0.4, 0.2],
    },
    "suburban_residential": {
        "description": "Evening peak, steady daytime",
        "profile": [0.1, 0.05, 0.05, 0.05, 0.1, 0.2, 0.3, 0.6, 0.8, 0.5,
                     0.4, 0.5, 0.5, 0.4, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,
                     0.9, 0.6, 0.3, 0.2],
    },
    "emerging": {
        "description": "Growing demand, evening focus",
        "profile": [0.05, 0.05, 0.05, 0.05, 0.1, 0.15, 0.3, 0.5, 0.6, 0.4,
                     0.3, 0.4, 0.4, 0.3, 0.3, 0.3, 0.4, 0.5, 0.7, 0.8,
                     0.7, 0.5, 0.3, 0.1],
    },
    "old_city": {
        "description": "Steady throughout day",
        "profile": [0.1, 0.1, 0.05, 0.05, 0.1, 0.2, 0.4, 0.7, 0.8, 0.7,
                     0.7, 0.7, 0.6, 0.6, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9,
                     0.8, 0.6, 0.3, 0.2],
    },
}


def _get_zone_for_point(lat: float, lng: float) -> tuple:
    """
    Determine which Gurgaon zone a point belongs to.
    Returns (zone_name, density_multiplier, zone_type).
    Falls back to 'General Gurgaon' if no zone matches.
    """
    for zone_name, zone_info in GURGAON_ZONES.items():
        clat, clng = zone_info["center"]
        r = zone_info["radius_deg"]
        dist = np.sqrt((lat - clat) ** 2 + (lng - clng) ** 2)
        if dist <= r:
            return zone_name, zone_info["density_multiplier"], zone_info["zone_type"]

    return "General Gurgaon", 1.0, "mid_residential"


@st.cache_data(show_spinner=False)
def generate_demand_points(
    poi_df: pd.DataFrame,
    points_per_poi: int = 5,
    noise_std: float = 0.003,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic demand points from POI locations.

    Args:
        poi_df: DataFrame from Layer 1 (osm_data.fetch_osm_data)
        points_per_poi: Number of demand points to scatter around each POI
        noise_std: Standard deviation of Gaussian noise (in degrees, ~300m)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns:
            lat, lng, demand_weight, zone, zone_type, time_profile,
            source_category, peak_hour_demand
    """
    np.random.seed(seed)

    records = []

    for _, poi in poi_df.iterrows():
        # How many demand points to generate around this POI
        n_points = max(1, int(points_per_poi * poi["demand_weight"]))

        for _ in range(n_points):
            # Add Gaussian noise to POI location
            lat = poi["lat"] + np.random.normal(0, noise_std)
            lng = poi["lng"] + np.random.normal(0, noise_std)

            # Identify which zone this point falls in
            zone_name, density_mult, zone_type = _get_zone_for_point(lat, lng)

            # Calculate demand weight
            base_weight = poi["demand_weight"]
            zone_weight = base_weight * density_mult

            # Add some random variation (±30%)
            weight = zone_weight * np.random.uniform(0.7, 1.3)

            # Get time profile for this zone type
            time_profile = TIME_PROFILES.get(
                zone_type, TIME_PROFILES["mid_residential"]
            )["profile"]

            # Peak hour demand (orders/hour estimate)
            peak_demand = weight * np.random.uniform(5, 25)

            records.append({
                "lat": lat,
                "lng": lng,
                "demand_weight": round(weight, 3),
                "zone": zone_name,
                "zone_type": zone_type,
                "time_profile": time_profile,
                "source_category": poi["category"],
                "peak_hour_demand": round(peak_demand, 1),
            })

    df = pd.DataFrame(records)

    # Filter to Gurgaon bounds
    from modules.osm_data import GURGAON_BOUNDS

    df = df[
        (df["lat"] >= GURGAON_BOUNDS["south"])
        & (df["lat"] <= GURGAON_BOUNDS["north"])
        & (df["lng"] >= GURGAON_BOUNDS["west"])
        & (df["lng"] <= GURGAON_BOUNDS["east"])
    ].reset_index(drop=True)

    return df


def get_demand_summary(demand_df: pd.DataFrame) -> dict:
    """Summary statistics for the generated demand."""
    return {
        "total_points": len(demand_df),
        "avg_weight": round(demand_df["demand_weight"].mean(), 3),
        "max_weight": round(demand_df["demand_weight"].max(), 3),
        "by_zone": demand_df["zone"].value_counts().to_dict(),
        "by_source": demand_df["source_category"].value_counts().to_dict(),
        "total_peak_demand": round(demand_df["peak_hour_demand"].sum(), 0),
    }
