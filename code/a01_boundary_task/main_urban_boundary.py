"""Command-line workflow for constructing global urban-fringe boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def _positive_distance(value: str) -> float:
    distance = float(value)
    if distance <= 0:
        raise argparse.ArgumentTypeError("distance must be greater than zero")
    return distance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select urban polygons near city points and construct a dissolved "
            "urban-fringe layer as the metric buffer minus the built-up polygons."
        )
    )
    parser.add_argument("--city-points", required=True, type=Path, help="City point vector file.")
    parser.add_argument(
        "--urban-polygons", required=True, type=Path, help="Urban-area polygon vector file."
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--polygon-id-field",
        default="id_left",
        help="Unique urban-polygon identifier field (default: id_left).",
    )
    parser.add_argument(
        "--population-field",
        default="population",
        help="Numeric city-population field used to resolve duplicate matches (default: population).",
    )
    parser.add_argument(
        "--city-class-field",
        help="Optional city-class field used with one or more --city-class filters.",
    )
    parser.add_argument(
        "--city-class",
        dest="city_classes",
        action="append",
        help="City class to retain; repeat for multiple classes.",
    )
    parser.add_argument(
        "--working-crs",
        default="EPSG:8857",
        help="Projected metre-based CRS used for distance and area calculations (default: EPSG:8857).",
    )
    parser.add_argument(
        "--output-crs",
        default="EPSG:4326",
        help="CRS for the second urban-fringe output (default: EPSG:4326/WGS84).",
    )
    parser.add_argument(
        "--point-buffer-m",
        type=_positive_distance,
        default=3500.0,
        help="City point matching radius in metres (default: 3500).",
    )
    parser.add_argument(
        "--fringe-buffer-m",
        type=_positive_distance,
        default=10000.0,
        help="Urban-fringe buffer distance in metres (default: 10000).",
    )
    parser.add_argument(
        "--output-prefix",
        help="Output filename prefix (default: stem of --urban-polygons).",
    )
    return parser


def _validate_input_path(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    return path


def _validate_prefix(prefix: str) -> str:
    if not prefix or Path(prefix).name != prefix or prefix in {".", ".."}:
        raise ValueError("Output prefix must be a non-empty filename component.")
    return prefix


def _write_vector(frame, path: Path) -> None:
    frame.to_file(path, index=False)
    print(f"Wrote {len(frame):,} feature(s): {path}")


def urban_boundary_buffer_task(
    city_points_path: Path,
    urban_polygons_path: Path,
    output_dir: Path,
    *,
    polygon_id_field: str = "id_left",
    population_field: str = "population",
    city_class_field: str | None = None,
    city_classes: Sequence[str] | None = None,
    working_crs: str = "EPSG:8857",
    output_crs: str = "EPSG:4326",
    point_buffer_distance_m: float = 3500.0,
    fringe_buffer_distance_m: float = 10000.0,
    output_prefix: str | None = None,
) -> dict[str, Path]:
    """Run the city-to-urban-area selection and urban-fringe construction workflow."""
    import geopandas as gpd
    from pyproj import CRS

    from .boundary_utils import (
        create_buffer,
        extract_polygons_with_points,
        require_metric_crs,
    )

    city_points_path = _validate_input_path(city_points_path, "City point file")
    urban_polygons_path = _validate_input_path(urban_polygons_path, "Urban polygon file")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = _validate_prefix(output_prefix or urban_polygons_path.stem)
    metric_crs = require_metric_crs(working_crs)
    target_crs = CRS.from_user_input(output_crs)

    print(f"Reading city points: {city_points_path}")
    points = gpd.read_file(city_points_path)
    print(f"Reading urban polygons: {urban_polygons_path}")
    polygons = gpd.read_file(urban_polygons_path)
    if points.crs is None:
        raise ValueError("City point data has no CRS.")
    if polygons.crs is None:
        raise ValueError("Urban polygon data has no CRS.")

    if city_classes:
        if not city_class_field:
            raise ValueError("--city-class-field is required when --city-class is used.")
        if city_class_field not in points.columns:
            raise ValueError(f"City point data is missing field {city_class_field!r}.")
        requested_classes = {str(value) for value in city_classes}
        points = points[points[city_class_field].astype(str).isin(requested_classes)].copy()
        if points.empty:
            raise ValueError("No city points remain after applying the city-class filter.")
        print(f"Retained {len(points):,} city point(s) after city-class filtering.")

    print(f"Reprojecting inputs to metric working CRS {metric_crs.to_string()}.")
    points = points.to_crs(metric_crs)
    polygons = polygons.to_crs(metric_crs)

    selected_polygons, point_report = extract_polygons_with_points(
        polygons=polygons,
        points=points,
        polygon_id_field=polygon_id_field,
        population_field=population_field,
        point_buffer_distance_m=point_buffer_distance_m,
    )
    print(
        f"Selected {len(selected_polygons):,} urban polygon(s); "
        f"{int((point_report['match_cnt'] > 0).sum()):,} of {len(point_report):,} city point(s) matched."
    )

    city_stem = city_points_path.stem
    selected_path = output_dir / f"{prefix}_with_{city_stem}.shp"
    point_report_path = output_dir / f"{city_stem}_with_{prefix}.shp"
    _write_vector(selected_polygons, selected_path)
    _write_vector(point_report, point_report_path)

    # Pass the in-memory selection directly to the second stage.
    urban_fringe = create_buffer(
        selected_polygons,
        buffer_distance_m=fringe_buffer_distance_m,
    )
    working_path = output_dir / f"{prefix}_buffer_dissolve.shp"
    _write_vector(urban_fringe, working_path)

    output_epsg = target_crs.to_epsg()
    output_label = str(output_epsg) if output_epsg is not None else "output_crs"
    projected_path = output_dir / f"{prefix}_buffer_dissolve_{output_label}.shp"
    _write_vector(urban_fringe.to_crs(target_crs), projected_path)

    total_area_km2 = float(urban_fringe["area"].sum()) / 1_000_000.0
    print(f"Urban-fringe area in the working CRS: {total_area_km2:,.3f} km^2")
    return {
        "selected_polygons": selected_path,
        "point_report": point_report_path,
        "urban_fringe_working": working_path,
        "urban_fringe_output_crs": projected_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        urban_boundary_buffer_task(
            city_points_path=args.city_points,
            urban_polygons_path=args.urban_polygons,
            output_dir=args.output_dir,
            polygon_id_field=args.polygon_id_field,
            population_field=args.population_field,
            city_class_field=args.city_class_field,
            city_classes=args.city_classes,
            working_crs=args.working_crs,
            output_crs=args.output_crs,
            point_buffer_distance_m=args.point_buffer_m,
            fringe_buffer_distance_m=args.fringe_buffer_m,
            output_prefix=args.output_prefix,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
