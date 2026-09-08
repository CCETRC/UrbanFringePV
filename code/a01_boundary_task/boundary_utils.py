"""Geospatial helpers for constructing urban-fringe study areas."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid


def require_metric_crs(crs_like: Any) -> CRS:
    """Return a projected CRS whose first two axes use metres."""
    if crs_like is None:
        raise ValueError("A coordinate reference system is required.")

    try:
        crs = CRS.from_user_input(crs_like)
    except Exception as exc:
        raise ValueError(f"Invalid working CRS: {crs_like!r}") from exc

    if not crs.is_projected:
        raise ValueError(
            f"Working CRS {crs.to_string()} is not projected; distances must be in metres."
        )

    axes = crs.axis_info[:2]
    if len(axes) < 2 or any(
        axis.unit_conversion_factor is None
        or abs(float(axis.unit_conversion_factor) - 1.0) > 1e-9
        for axis in axes
    ):
        units = ", ".join(axis.unit_name or "unknown" for axis in axes) or "unknown"
        raise ValueError(
            f"Working CRS {crs.to_string()} must use metres on both axes; found {units}."
        )

    return crs


def _require_columns(frame: gpd.GeoDataFrame, fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in fields if field not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required field(s): {', '.join(missing)}")


def _polygon_parts(geometry: Any) -> list[Polygon]:
    """Extract non-empty polygon parts from a possibly repaired geometry."""
    if geometry is None or geometry.is_empty:
        return []

    repaired = geometry if geometry.is_valid else make_valid(geometry)
    if isinstance(repaired, Polygon):
        return [repaired] if repaired.area > 0 else []
    if isinstance(repaired, MultiPolygon):
        return [part for part in repaired.geoms if not part.is_empty and part.area > 0]
    if isinstance(repaired, GeometryCollection):
        parts: list[Polygon] = []
        for part in repaired.geoms:
            parts.extend(_polygon_parts(part))
        return parts
    return []


def _normalise_polygons(frame: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    if frame.empty:
        raise ValueError(f"{label} contains no features.")

    normalised = frame.copy()
    geometries = []
    invalid_rows = []
    for index, geometry in normalised.geometry.items():
        parts = _polygon_parts(geometry)
        if not parts:
            invalid_rows.append(str(index))
            geometries.append(None)
            continue
        geometries.append(parts[0] if len(parts) == 1 else unary_union(parts))

    if invalid_rows:
        preview = ", ".join(invalid_rows[:5])
        suffix = "..." if len(invalid_rows) > 5 else ""
        raise ValueError(
            f"{label} has {len(invalid_rows)} empty or non-polygon geometry row(s): "
            f"{preview}{suffix}"
        )

    normalised.geometry = geometries
    return normalised


def _validate_shared_metric_crs(
    polygons: gpd.GeoDataFrame, points: gpd.GeoDataFrame
) -> CRS:
    polygon_crs = require_metric_crs(polygons.crs)
    point_crs = require_metric_crs(points.crs)
    if not polygon_crs.equals(point_crs):
        raise ValueError(
            "Urban polygons and city points must be in the same metric working CRS."
        )
    return polygon_crs


def extract_polygons_with_points(
    polygons: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    polygon_id_field: str,
    population_field: str,
    point_buffer_distance_m: float = 3500.0,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Select urban polygons intersecting buffered city points.

    If more than one city matches the same polygon ID, the city with the
    largest population determines the retained match. The returned point
    report preserves all input points and adds ``match_cnt``.
    """
    if point_buffer_distance_m <= 0:
        raise ValueError("Point-buffer distance must be greater than zero metres.")

    _require_columns(polygons, (polygon_id_field,), "Urban polygon data")
    _require_columns(points, (population_field,), "City point data")
    if polygons[polygon_id_field].isna().any():
        raise ValueError(f"Urban polygon field {polygon_id_field!r} contains null IDs.")
    if polygons[polygon_id_field].duplicated().any():
        raise ValueError(f"Urban polygon field {polygon_id_field!r} must be unique.")
    if "match_cnt" in points.columns:
        raise ValueError("City point data already contains reserved field 'match_cnt'.")

    polygons = _normalise_polygons(polygons, "Urban polygon data")
    if points.empty:
        raise ValueError("City point data contains no features.")
    if points.geometry.isna().any() or points.geometry.is_empty.any():
        raise ValueError("City point data contains null or empty geometries.")
    unsupported = ~points.geom_type.isin(["Point", "MultiPoint"])
    if unsupported.any():
        types = ", ".join(sorted(set(points.loc[unsupported].geom_type)))
        raise ValueError(f"City point data must contain Point/MultiPoint geometries, found: {types}")

    _validate_shared_metric_crs(polygons, points)

    populations = pd.to_numeric(points[population_field], errors="coerce")
    if populations.isna().any():
        bad_count = int(populations.isna().sum())
        raise ValueError(
            f"City point field {population_field!r} has {bad_count} missing or non-numeric value(s)."
        )

    point_buffers = points.copy()
    point_buffers["__ufpv_row"] = range(len(point_buffers))
    point_buffers["__ufpv_population"] = populations.to_numpy()
    point_buffers.geometry = [
        geometry.buffer(point_buffer_distance_m) for geometry in point_buffers.geometry
    ]

    joined = gpd.sjoin(
        polygons,
        point_buffers[["__ufpv_row", "__ufpv_population", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        raise ValueError(
            "No urban polygon intersects a buffered city point; verify the inputs, CRS, "
            "and point-buffer distance."
        )

    joined = joined.sort_values(
        "__ufpv_population", ascending=False, kind="mergesort"
    ).drop_duplicates(subset=polygon_id_field, keep="first")
    selected = joined.loc[:, polygons.columns].copy().reset_index(drop=True)
    selected["__sort_id"] = selected[polygon_id_field].astype(str)
    selected = selected.sort_values("__sort_id", kind="mergesort").drop(
        columns="__sort_id"
    ).reset_index(drop=True)

    all_matches = gpd.sjoin(
        polygons[[polygon_id_field, "geometry"]],
        point_buffers[["__ufpv_row", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    match_counts = all_matches.groupby("__ufpv_row").size()
    point_report = points.copy().reset_index(drop=True)
    point_report["match_cnt"] = [
        int(match_counts.get(row_number, 0)) for row_number in range(len(point_report))
    ]

    return selected, point_report


def create_buffer(
    polygons: gpd.GeoDataFrame,
    buffer_distance_m: float = 10000.0,
) -> gpd.GeoDataFrame:
    """Create dissolved urban-fringe polygons as buffer minus built-up area."""
    if buffer_distance_m <= 0:
        raise ValueError("Urban-fringe buffer distance must be greater than zero metres.")

    polygons = _normalise_polygons(polygons, "Selected urban polygon data")
    require_metric_crs(polygons.crs)

    urban_union = unary_union(list(polygons.geometry))
    buffered_union = unary_union(
        [geometry.buffer(buffer_distance_m) for geometry in polygons.geometry]
    )
    fringe_geometry = buffered_union.difference(urban_union)
    fringe_parts = _polygon_parts(fringe_geometry)
    if not fringe_parts:
        raise ValueError("Urban-fringe construction produced no polygon features.")

    fringe_parts.sort(
        key=lambda geometry: (
            *(round(float(value), 9) for value in geometry.bounds),
            round(float(geometry.area), 6),
            geometry.wkb_hex,
        )
    )
    fringe = gpd.GeoDataFrame(
        {
            "feat_id": range(1, len(fringe_parts) + 1),
            "area": [float(geometry.area) for geometry in fringe_parts],
        },
        geometry=fringe_parts,
        crs=polygons.crs,
    )
    return fringe
