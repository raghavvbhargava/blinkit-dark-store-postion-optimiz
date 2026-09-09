"""
Layer 4: Location Optimization Engine
=======================================
Two-stage optimization to find optimal dark store placements:

Stage 1: Weighted K-Means Clustering
  - Uses scikit-learn KMeans
  - Weighted by demand prediction scores
  - User-configurable K (number of stores)

Stage 2: Constraint-Based Refinement
  - Minimum inter-store distance (2km default)
  - Maximum service radius (3km default)
  - Coverage optimization

Output: List of optimal locations with scores and coverage metrics.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import streamlit as st


def _haversine_km(lat1, lng1, lat2, lng2):
    """Calculate distance in km between two points."""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def _weighted_kmeans(
    scored_df: pd.DataFrame,
    n_clusters: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """
    Stage 1: Run weighted K-Means clustering.

    Points with higher predicted_score are sampled more often,
    effectively pulling cluster centroids toward high-demand areas.

    Returns:
        Array of shape (n_clusters, 2) with [lat, lng] centroids
    """
    coords = scored_df[["lat", "lng"]].values
    weights = scored_df["predicted_score"].values

    # Normalize weights to create a probability distribution
    if weights.sum() > 0:
        sample_weights = weights / weights.sum()
    else:
        sample_weights = np.ones(len(weights)) / len(weights)

    # Oversample high-demand points for weighted clustering
    n_samples = min(len(coords) * 3, 15000)
    indices = np.random.RandomState(seed).choice(
        len(coords), size=n_samples, replace=True, p=sample_weights
    )
    weighted_coords = coords[indices]

    # Run K-Means
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=10,
        max_iter=300,
    )
    kmeans.fit(weighted_coords)

    return kmeans.cluster_centers_


def _enforce_min_distance(
    centroids: np.ndarray,
    min_distance_km: float = 2.0,
) -> np.ndarray:
    """
    Stage 2a: Ensure minimum inter-store distance.

    If two centroids are too close, merge them by keeping
    the one with better strategic position (further from edges).
    """
    from modules.osm_data import GURGAON_BOUNDS

    center_lat = (GURGAON_BOUNDS["north"] + GURGAON_BOUNDS["south"]) / 2
    center_lng = (GURGAON_BOUNDS["east"] + GURGAON_BOUNDS["west"]) / 2

    filtered = list(centroids.copy())
    changed = True

    while changed:
        changed = False
        i = 0
        while i < len(filtered):
            j = i + 1
            while j < len(filtered):
                dist = _haversine_km(
                    filtered[i][0], filtered[i][1],
                    filtered[j][0], filtered[j][1],
                )
                if dist < min_distance_km:
                    # Keep the one closer to city center (better coverage)
                    dist_i = _haversine_km(filtered[i][0], filtered[i][1], center_lat, center_lng)
                    dist_j = _haversine_km(filtered[j][0], filtered[j][1], center_lat, center_lng)

                    if dist_i <= dist_j:
                        filtered.pop(j)
                    else:
                        filtered.pop(i)
                    changed = True
                    break
                j += 1
            if changed:
                break
            i += 1

    return np.array(filtered)


def _compute_coverage(
    centroids: np.ndarray,
    scored_df: pd.DataFrame,
    service_radius_km: float = 3.0,
) -> list:
    """
    Compute coverage metrics for each centroid.

    Returns list of dicts with:
        - points_covered: number of demand points within radius
        - demand_covered: total predicted_score of covered points
        - coverage_pct: percentage of total demand covered
        - avg_distance_km: average distance to covered points
    """
    coords = scored_df[["lat", "lng"]].values
    scores = scored_df["predicted_score"].values
    total_demand = scores.sum()

    # Convert service radius to approximate degrees
    # 1 degree ≈ 111 km at this latitude
    radius_deg = service_radius_km / 111.0

    coverage_list = []

    for centroid in centroids:
        # Calculate distances to all demand points
        distances_deg = np.sqrt(
            (coords[:, 0] - centroid[0]) ** 2
            + (coords[:, 1] - centroid[1]) ** 2
        )

        within_radius = distances_deg <= radius_deg
        points_covered = within_radius.sum()
        demand_covered = scores[within_radius].sum()

        # Average distance in km for covered points
        covered_distances_km = distances_deg[within_radius] * 111.0
        avg_dist = covered_distances_km.mean() if len(covered_distances_km) > 0 else 0

        coverage_list.append({
            "points_covered": int(points_covered),
            "demand_covered": round(float(demand_covered), 1),
            "coverage_pct": round(float(demand_covered / total_demand * 100), 1) if total_demand > 0 else 0,
            "avg_distance_km": round(float(avg_dist), 2),
        })

    return coverage_list


def optimize_locations(
    scored_df: pd.DataFrame,
    n_stores: int = 8,
    min_distance_km: float = 2.0,
    service_radius_km: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run the full optimization pipeline.

    Args:
        scored_df: DataFrame from Layer 3 (with predicted_score)
        n_stores: Number of dark stores to place
        min_distance_km: Minimum distance between stores
        service_radius_km: Maximum delivery service radius
        seed: Random seed

    Returns:
        DataFrame with columns:
            store_id, lat, lng, points_covered, demand_covered,
            coverage_pct, avg_distance_km, score_rank
    """
    # Stage 1: Weighted K-Means
    centroids = _weighted_kmeans(scored_df, n_clusters=n_stores, seed=seed)

    # Stage 2a: Enforce minimum distance
    centroids = _enforce_min_distance(centroids, min_distance_km)

    # Stage 2b: Compute coverage
    coverage = _compute_coverage(centroids, scored_df, service_radius_km)

    # Build results dataframe
    results = []
    for i, (centroid, cov) in enumerate(zip(centroids, coverage)):
        results.append({
            "store_id": f"DS-{i+1:02d}",
            "lat": round(float(centroid[0]), 6),
            "lng": round(float(centroid[1]), 6),
            "points_covered": cov["points_covered"],
            "demand_covered": cov["demand_covered"],
            "coverage_pct": cov["coverage_pct"],
            "avg_distance_km": cov["avg_distance_km"],
        })

    result_df = pd.DataFrame(results)

    # Rank by demand covered
    result_df["score_rank"] = result_df["demand_covered"].rank(ascending=False).astype(int)
    result_df = result_df.sort_values("score_rank").reset_index(drop=True)

    return result_df


def get_total_coverage(
    optimal_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    service_radius_km: float = 3.0,
) -> dict:
    """
    Calculate overall coverage across all optimal stores.
    (Accounting for overlap between service areas)
    """
    coords = scored_df[["lat", "lng"]].values
    scores = scored_df["predicted_score"].values
    total_demand = scores.sum()

    radius_deg = service_radius_km / 111.0

    # Track which points are covered by ANY store
    covered_mask = np.zeros(len(coords), dtype=bool)

    for _, store in optimal_df.iterrows():
        distances = np.sqrt(
            (coords[:, 0] - store["lat"]) ** 2
            + (coords[:, 1] - store["lng"]) ** 2
        )
        covered_mask |= (distances <= radius_deg)

    total_covered_points = covered_mask.sum()
    total_covered_demand = scores[covered_mask].sum()

    return {
        "total_stores": len(optimal_df),
        "total_demand_points": len(coords),
        "covered_points": int(total_covered_points),
        "coverage_pct": round(float(total_covered_points / len(coords) * 100), 1),
        "demand_coverage_pct": round(float(total_covered_demand / total_demand * 100), 1) if total_demand > 0 else 0,
        "uncovered_points": int(len(coords) - total_covered_points),
        "avg_delivery_radius": round(float(optimal_df["avg_distance_km"].mean()), 2),
    }


def calculate_validation_metrics(
    optimal_df: pd.DataFrame,
    service_radius_km: float = 3.0,
) -> dict:
    """
    Calculate validation and accuracy metrics comparing optimized dark store
    placements to actual Blinkit dark store locations in Gurgaon.
    """
    from modules.demand_model import REAL_BLINKIT_STORES

    if len(optimal_df) == 0 or len(REAL_BLINKIT_STORES) == 0:
        return {
            "avg_dist_to_real_km": 0.0,
            "real_stores_covered_pct": 0.0,
            "optimized_precision_pct": 0.0,
            "overall_match_score": 0.0,
            "total_real_stores": len(REAL_BLINKIT_STORES),
        }

    opt_coords = optimal_df[["lat", "lng"]].values
    real_coords = np.array([[s[0], s[1]] for s in REAL_BLINKIT_STORES])

    # 1. Average distance from each optimized store to the closest real store
    opt_to_real_dists = []
    for opt in opt_coords:
        dists = [_haversine_km(opt[0], opt[1], real[0], real[1]) for real in real_coords]
        opt_to_real_dists.append(min(dists))
    avg_dist_to_real = np.mean(opt_to_real_dists)

    # 2. Percentage of real stores covered by at least one optimized store (Recall)
    real_covered = 0
    for real in real_coords:
        dists = [_haversine_km(real[0], real[1], opt[0], opt[1]) for opt in opt_coords]
        if min(dists) <= service_radius_km:
            real_covered += 1
    real_covered_pct = (real_covered / len(REAL_BLINKIT_STORES)) * 100

    # 3. Percentage of optimized stores that are within 2.0km of a real store (Precision)
    opt_precision_count = sum(1 for d in opt_to_real_dists if d <= 2.0)
    opt_precision_pct = (opt_precision_count / len(optimal_df)) * 100

    # 4. Overall Match Score / Alignment Index (0 - 100)
    # 100 is perfect. We subtract for distance and add slightly for coverage.
    overall_match_score = max(0.0, min(100.0, 100.0 - (avg_dist_to_real * 12.0) + (real_covered_pct * 0.12)))

    return {
        "avg_dist_to_real_km": round(float(avg_dist_to_real), 2),
        "real_stores_covered_pct": round(float(real_covered_pct), 1),
        "optimized_precision_pct": round(float(opt_precision_pct), 1),
        "overall_match_score": round(float(overall_match_score), 1),
        "total_real_stores": len(REAL_BLINKIT_STORES),
    }

