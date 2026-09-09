"""
Layer 3: Demand Prediction Model
=================================
A transparent, explainable scoring model that predicts demand intensity
at each point based on multiple weighted factors.

Scoring Factors:
  1. Population density (from zone data)
  2. POI proximity bonus (nearby restaurants/stores = existing demand)
  3. Accessibility score (proximity to metro/main roads)
  4. Competition penalty (reduce near existing grocery stores)
  5. Time-weighted demand (peak vs off-peak)

Each factor's weight is adjustable via Streamlit sliders for
real-time what-if analysis.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import streamlit as st


# ── Default model weights (user-adjustable via sliders) ──────────
DEFAULT_WEIGHTS = {
    "population_density": 0.30,
    "poi_proximity": 0.25,
    "accessibility": 0.20,
    "competition_penalty": 0.15,
    "time_factor": 0.10,
}


# ── Known metro stations in Gurgaon (accessibility anchors) ─────
METRO_STATIONS = [
    (28.4594, 77.0724, "Huda City Centre"),
    (28.4728, 77.0568, "IFFCO Chowk"),
    (28.4822, 77.0956, "Guru Dronacharya"),
    (28.4950, 77.0890, "Cyber City"),  # Rapid Metro
    (28.4700, 77.0710, "Sikanderpur"),
    (28.4852, 77.0958, "Phase 3"),
    (28.4880, 77.0830, "Moulsari Avenue"),
    (28.4780, 77.1020, "Sector 55-56"),
]

# ── Known competitor locations (existing grocery/dark stores) ────
COMPETITORS = [
    (28.4730, 77.0900, "BigBasket Warehouse"),
    (28.4590, 77.0620, "Grofers Hub"),
    (28.4480, 77.0700, "DMart Sector 54"),
    (28.4850, 77.0350, "Reliance Fresh Palam Vihar"),
    (28.4400, 77.0500, "More Supermarket Sohna Rd"),
    (28.4700, 77.0400, "Spencer's Sector 14"),
    (28.4950, 77.0800, "Nature's Basket Cyber Hub"),
    (28.4550, 77.0750, "BigBasket Sector 56"),
]

# ── Real Blinkit dark store locations in Gurgaon for accuracy validation ──
REAL_BLINKIT_STORES = [
    (28.4949, 77.0880, "Blinkit Store - DLF Cyber City"),
    (28.4795, 77.0800, "Blinkit Store - DLF Phase 2 / MG Road"),
    (28.4680, 77.0850, "Blinkit Store - DLF Phase 1 / Sushant Lok 1"),
    (28.4690, 77.0380, "Blinkit Store - Sector 14"),
    (28.4950, 77.0150, "Blinkit Store - Sector 22 / Palam Vihar"),
    (28.4550, 77.0420, "Blinkit Store - Sector 31"),
    (28.4480, 77.0600, "Blinkit Store - Sector 45 / South City 1"),
    (28.4320, 77.0550, "Blinkit Store - Sector 46 / Sector 50"),
    (28.4350, 77.0820, "Blinkit Store - Sector 56"),
    (28.4430, 77.0980, "Blinkit Store - Sector 57 / Golf Course Road"),
    (28.4120, 77.0780, "Blinkit Store - Sector 62 / Golf Course Ext"),
    (28.4180, 77.0400, "Blinkit Store - Sohna Road / Sector 49"),
    (28.4050, 76.9950, "Blinkit Store - Dwarka Expressway / Sector 82"),
    (28.4600, 76.9850, "Blinkit Store - Sector 104"),
    (28.4900, 76.9800, "Blinkit Store - Sector 110"),
]


def _haversine_km(lat1, lng1, lat2, lng2):
    """Calculate distance in km between two lat/lng points."""
    R = 6371  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def _compute_population_score(demand_df: pd.DataFrame) -> np.ndarray:
    """
    Score based on zone density multiplier.
    Higher density zone = higher score.
    """
    scores = demand_df["demand_weight"].values.copy()
    # Normalize to 0-1
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores


def _compute_poi_proximity_score(demand_df: pd.DataFrame, poi_df: pd.DataFrame) -> np.ndarray:
    """
    Score based on number and proximity of nearby POIs.
    More nearby food/grocery POIs = existing demand signal.
    """
    demand_coords = demand_df[["lat", "lng"]].values
    poi_coords = poi_df[["lat", "lng"]].values

    # Count POIs within ~500m (approx 0.005 degrees)
    scores = np.zeros(len(demand_df))

    # Use efficient batch distance computation
    if len(poi_coords) > 0:
        distances = cdist(demand_coords, poi_coords, metric="euclidean")
        # Count nearby POIs (within 0.005 degrees ≈ 500m)
        nearby_count = (distances < 0.005).sum(axis=1)
        scores = nearby_count.astype(float)

        if scores.max() > 0:
            scores = scores / scores.max()

    return scores


def _compute_accessibility_score(demand_df: pd.DataFrame) -> np.ndarray:
    """
    Score based on proximity to metro stations.
    Closer to metro = more accessible = higher delivery demand.
    """
    demand_coords = demand_df[["lat", "lng"]].values
    metro_coords = np.array([(s[0], s[1]) for s in METRO_STATIONS])

    if len(metro_coords) == 0:
        return np.ones(len(demand_df)) * 0.5

    distances = cdist(demand_coords, metro_coords, metric="euclidean")
    min_distances = distances.min(axis=1)

    # Inverse distance scoring (closer = higher score)
    # Use 0.02 degrees (~2km) as the reference distance
    scores = np.clip(1.0 - (min_distances / 0.02), 0.0, 1.0)

    return scores


def _compute_competition_score(demand_df: pd.DataFrame) -> np.ndarray:
    """
    Penalty for being too close to existing competitors.
    Far from competition = higher opportunity score.
    """
    demand_coords = demand_df[["lat", "lng"]].values
    comp_coords = np.array([(c[0], c[1]) for c in COMPETITORS])

    if len(comp_coords) == 0:
        return np.ones(len(demand_df)) * 0.5

    distances = cdist(demand_coords, comp_coords, metric="euclidean")
    min_distances = distances.min(axis=1)

    # Areas far from competitors get higher scores
    # Use 0.01 degrees (~1km) as reference
    scores = np.clip(min_distances / 0.01, 0.0, 1.0)

    return scores


def _compute_time_factor(demand_df: pd.DataFrame, hour: int = 19) -> np.ndarray:
    """
    Score based on the time-of-day demand profile.
    Default hour = 19 (7 PM, typical evening peak).
    """
    scores = np.zeros(len(demand_df))

    for i, row in demand_df.iterrows():
        profile = row.get("time_profile", [0.5] * 24)
        if isinstance(profile, (list, np.ndarray)) and len(profile) == 24:
            scores[i] = profile[hour]
        else:
            scores[i] = 0.5

    if scores.max() > 0:
        scores = scores / scores.max()

    return scores


def predict_demand(
    demand_df: pd.DataFrame,
    poi_df: pd.DataFrame,
    weights: dict = None,
    peak_hour: int = 19,
) -> pd.DataFrame:
    """
    Run the demand prediction model on all demand points.

    Args:
        demand_df: DataFrame from Layer 2
        poi_df: DataFrame from Layer 1
        weights: Dict of factor weights (default: DEFAULT_WEIGHTS)
        peak_hour: Hour of day for time factor (0-23)

    Returns:
        demand_df with added columns:
            predicted_score, population_score, poi_score,
            accessibility_score, competition_score, time_score
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    # Compute individual factor scores
    pop_scores = _compute_population_score(demand_df)
    poi_scores = _compute_poi_proximity_score(demand_df, poi_df)
    acc_scores = _compute_accessibility_score(demand_df)
    comp_scores = _compute_competition_score(demand_df)
    time_scores = _compute_time_factor(demand_df, peak_hour)

    # Weighted combination
    predicted = (
        weights["population_density"] * pop_scores
        + weights["poi_proximity"] * poi_scores
        + weights["accessibility"] * acc_scores
        + weights["competition_penalty"] * comp_scores
        + weights["time_factor"] * time_scores
    )

    # Normalize final score to 0-100
    if predicted.max() > 0:
        predicted = (predicted / predicted.max()) * 100

    # Add all scores to dataframe
    result = demand_df.copy()
    result["predicted_score"] = np.round(predicted, 2)
    result["population_score"] = np.round(pop_scores * 100, 2)
    result["poi_score"] = np.round(poi_scores * 100, 2)
    result["accessibility_score"] = np.round(acc_scores * 100, 2)
    result["competition_score"] = np.round(comp_scores * 100, 2)
    result["time_score"] = np.round(time_scores * 100, 2)

    return result


def get_score_breakdown(scored_df: pd.DataFrame) -> dict:
    """Get average scores by factor for the dashboard."""
    return {
        "avg_predicted": round(scored_df["predicted_score"].mean(), 1),
        "avg_population": round(scored_df["population_score"].mean(), 1),
        "avg_poi": round(scored_df["poi_score"].mean(), 1),
        "avg_accessibility": round(scored_df["accessibility_score"].mean(), 1),
        "avg_competition": round(scored_df["competition_score"].mean(), 1),
        "avg_time": round(scored_df["time_score"].mean(), 1),
    }
