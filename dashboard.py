"""
Layer 6: Business Dashboard
==============================
Generates Plotly charts and KPI metrics for the Streamlit dashboard.

Charts:
  - Demand by zone (bar chart)
  - POI type distribution (donut chart)
  - Hourly demand profile (area chart)
  - Score breakdown radar chart
  - Store ranking comparison (horizontal bar)
  - Coverage analysis (gauge chart)
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# ── Color palette (Blinkit-inspired dark theme) ─────────────────
COLORS = {
    "primary": "#F7C948",      # Blinkit yellow
    "secondary": "#e94560",    # Accent red-pink
    "success": "#4CAF50",      # Green
    "info": "#2196F3",         # Blue
    "warning": "#FF9800",      # Orange
    "purple": "#9C27B0",       # Purple
    "cyan": "#00BCD4",         # Cyan
    "bg_dark": "#0D1117",      # Background
    "bg_card": "#161B22",      # Card background
    "text": "#E6EDF3",         # Text color
    "text_muted": "#8B949E",   # Muted text
}

CHART_COLORS = [
    "#F7C948", "#e94560", "#4CAF50", "#2196F3",
    "#FF9800", "#9C27B0", "#00BCD4", "#F44336",
    "#8BC34A", "#FF5722", "#607D8B",
]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="Segoe UI, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(
        bgcolor="rgba(22,27,34,0.8)",
        bordercolor=COLORS["text_muted"],
        borderwidth=1,
    ),
)


def demand_by_zone_chart(scored_df: pd.DataFrame) -> go.Figure:
    """Bar chart showing total demand by Gurgaon zone."""
    zone_demand = (
        scored_df.groupby("zone")["predicted_score"]
        .sum()
        .sort_values(ascending=True)
        .reset_index()
    )

    fig = px.bar(
        zone_demand,
        x="predicted_score",
        y="zone",
        orientation="h",
        color="predicted_score",
        color_continuous_scale=["#1a1a6c", "#6a3093", "#e94560", "#F7C948"],
        labels={"predicted_score": "Total Demand Score", "zone": "Zone"},
        title="📊 Demand Distribution by Zone",
    )

    fig.update_layout(**PLOTLY_LAYOUT, height=400, coloraxis_showscale=False)
    fig.update_traces(marker_line_width=0)

    return fig


def poi_distribution_chart(poi_df: pd.DataFrame) -> go.Figure:
    """Donut chart showing POI type distribution."""
    poi_counts = poi_df["category"].value_counts().reset_index()
    poi_counts.columns = ["category", "count"]

    # Prettify category names
    poi_counts["category"] = poi_counts["category"].str.replace("_", " ").str.title()

    fig = px.pie(
        poi_counts,
        values="count",
        names="category",
        hole=0.5,
        color_discrete_sequence=CHART_COLORS,
        title="🏪 POI Type Distribution",
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker=dict(line=dict(color=COLORS["bg_dark"], width=2)),
    )

    fig.update_layout(**PLOTLY_LAYOUT, height=350)

    return fig


def hourly_demand_chart(scored_df: pd.DataFrame) -> go.Figure:
    """Area chart showing average hourly demand profile."""
    # Aggregate time profiles across all demand points
    all_profiles = scored_df["time_profile"].tolist()
    if not all_profiles or not isinstance(all_profiles[0], list):
        # Fallback
        avg_profile = [0.5] * 24
    else:
        profiles_array = np.array(all_profiles)
        avg_profile = profiles_array.mean(axis=0)

    hours = list(range(24))
    hour_labels = [f"{h:02d}:00" for h in hours]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hour_labels,
        y=avg_profile,
        mode="lines",
        fill="tozeroy",
        line=dict(color=COLORS["primary"], width=3),
        fillcolor="rgba(247, 201, 72, 0.15)",
        name="Demand",
    ))

    # Mark peak hours
    peak_hour = int(np.argmax(avg_profile))
    fig.add_vline(
        x=peak_hour,
        line_dash="dash",
        line_color=COLORS["secondary"],
        annotation_text=f"Peak: {hour_labels[peak_hour]}",
        annotation_font_color=COLORS["secondary"],
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="⏰ Hourly Demand Profile",
        xaxis_title="Hour",
        yaxis_title="Demand Level",
        height=300,
    )

    fig.update_xaxes(showgrid=False, tickangle=45)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(139,148,158,0.1)")

    return fig


def score_breakdown_radar(scored_df: pd.DataFrame) -> go.Figure:
    """Radar chart showing average score breakdown."""
    categories = [
        "Population", "POI Proximity",
        "Accessibility", "Competition", "Time Factor"
    ]
    values = [
        scored_df["population_score"].mean(),
        scored_df["poi_score"].mean(),
        scored_df["accessibility_score"].mean(),
        scored_df["competition_score"].mean(),
        scored_df["time_score"].mean(),
    ]
    # Close the polygon
    categories = categories + [categories[0]]
    values = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(247, 201, 72, 0.2)",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=6, color=COLORS["primary"]),
        name="Score",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="🎯 Demand Score Breakdown",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="rgba(139,148,158,0.15)",
                color=COLORS["text_muted"],
            ),
            angularaxis=dict(
                gridcolor="rgba(139,148,158,0.15)",
                color=COLORS["text"],
            ),
        ),
        height=350,
    )

    return fig


def store_ranking_chart(optimal_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart comparing store performance."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=optimal_df["store_id"],
        x=optimal_df["demand_covered"],
        orientation="h",
        marker=dict(
            color=optimal_df["demand_covered"],
            colorscale=[[0, "#1a1a6c"], [0.5, "#e94560"], [1, "#F7C948"]],
            line=dict(width=0),
        ),
        text=[f'{v:.0f}' for v in optimal_df["demand_covered"]],
        textposition="auto",
        textfont=dict(color="white", size=11),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="🏆 Store Demand Ranking",
        xaxis_title="Demand Score Covered",
        yaxis_title="",
        height=max(250, len(optimal_df) * 40 + 80),
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(139,148,158,0.1)")

    return fig


def coverage_gauge(coverage_pct: float) -> go.Figure:
    """Gauge chart showing total coverage percentage."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=coverage_pct,
        number=dict(suffix="%", font=dict(size=40, color=COLORS["primary"])),
        title=dict(text="Population Coverage", font=dict(size=14, color=COLORS["text"])),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=COLORS["text_muted"]),
            bar=dict(color=COLORS["primary"]),
            bgcolor=COLORS["bg_card"],
            borderwidth=0,
            steps=[
                dict(range=[0, 40], color="rgba(233,69,96,0.2)"),
                dict(range=[40, 70], color="rgba(255,152,0,0.2)"),
                dict(range=[70, 100], color="rgba(76,175,80,0.2)"),
            ],
            threshold=dict(
                line=dict(color=COLORS["secondary"], width=3),
                thickness=0.8,
                value=85,
            ),
        ),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=250,
    )

    return fig


def demand_vs_coverage_scatter(optimal_df: pd.DataFrame) -> go.Figure:
    """Scatter plot: demand covered vs. average delivery distance per store."""
    fig = px.scatter(
        optimal_df,
        x="avg_distance_km",
        y="demand_covered",
        size="points_covered",
        color="coverage_pct",
        color_continuous_scale=["#1a1a6c", "#e94560", "#F7C948"],
        hover_name="store_id",
        labels={
            "avg_distance_km": "Avg Delivery Distance (km)",
            "demand_covered": "Demand Covered",
            "coverage_pct": "Coverage %",
            "points_covered": "Points Covered",
        },
        title="📏 Demand vs. Delivery Distance",
    )

    fig.update_layout(**PLOTLY_LAYOUT, height=350)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(139,148,158,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(139,148,158,0.1)")

    return fig
