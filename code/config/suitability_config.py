"""Configuration for the photovoltaic suitability workflow.

Paths and account-scoped Earth Engine identifiers can be overridden with
environment variables. Feature ranges default to ``None``, which processes all
available features. The numerical research thresholds retain their original
values but can be overridden for documented sensitivity analyses.
"""

from __future__ import annotations

import os
from pathlib import Path

from config.base_config import INPUT_DIR, OUTPUT_DIR


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _optional_int(name: str) -> int | None:
    value = os.getenv(name)
    return None if value is None or value.strip().lower() in {"", "none", "null"} else int(value)


def _optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return None if value is None or value.strip().lower() in {"", "none", "null"} else float(value)


# Shared local inputs.
BUFFER_SHP = _env_path(
    "URBANFRINGEPV_BUFFER_SHP",
    OUTPUT_DIR / "urban_boundary_buffer" / "urban_areas_full_proj_buffer_dissolve_4326.shp",
)
BUFFER_PROJ_SHP_PATH = _env_path(
    "URBANFRINGEPV_BUFFER_PROJ_SHP",
    OUTPUT_DIR / "urban_boundary_buffer" / "urban_areas_full_proj_buffer_dissolve.shp",
)
BASE_SHP_PATH = _env_path(
    "URBANFRINGEPV_COUNTRY_SHP",
    INPUT_DIR / "ne_10m_admin_0_countries" / "ne_10m_admin_0_countries.shp",
)
CURRENT_PV_DIR = _env_path(
    "URBANFRINGEPV_CURRENT_PV_DIR",
    INPUT_DIR / "global_pv_area" / "current_pv_2022",
)

# Earth Engine account-scoped settings. No personal project is a public default.
GEE_PROJECT_ID = os.getenv("URBANFRINGEPV_GEE_PROJECT_ID")
GEE_BOUNDARIES = os.getenv("URBANFRINGEPV_GEE_BOUNDARIES")
BATCH_SIZE = _env_int("URBANFRINGEPV_GEE_BATCH_SIZE", 30)
START_ID = _optional_int("URBANFRINGEPV_GEE_START_ID")
END_ID = _optional_int("URBANFRINGEPV_GEE_END_ID")

# Land-cover outputs.
LAND_DIR = _env_path("URBANFRINGEPV_LAND_DIR", OUTPUT_DIR / "land")
LAND_FEATURE_MASK_DIR = _env_path(
    "URBANFRINGEPV_LAND_FEATURE_MASK_DIR",
    LAND_DIR / "bare_rangeland_scale10",
)
LAND_GLOBAL_MASK_TIF = LAND_DIR / "global_land_mask.tif"

# Slope outputs and research threshold.
SLOPE_DEGREE_THRESHOLD = _env_float("URBANFRINGEPV_SLOPE_THRESHOLD_DEG", 10.0)
SLOPE_DIR = _env_path("URBANFRINGEPV_SLOPE_DIR", OUTPUT_DIR / "slope")
SLOPE_SOURCE_TIF_DIR = _env_path(
    "URBANFRINGEPV_SLOPE_SOURCE_TIF_DIR",
    SLOPE_DIR / "slope_scale30",
)
SLOPE_FEATURE_MASK_DIR = _env_path(
    "URBANFRINGEPV_SLOPE_FEATURE_MASK_DIR",
    SLOPE_DIR / "slope_masks",
)
SLOPE_GLOBAL_MASK_TIF = SLOPE_DIR / "global_slope_mask.tif"

# Solar-radiation inputs, intermediates, and research thresholds.
SOLAR_DIR = _env_path("URBANFRINGEPV_SOLAR_DIR", OUTPUT_DIR / "ssrd")
MIN_SOLAR_THRESHOLD = _env_float("URBANFRINGEPV_MIN_SOLAR_KWH_M2", 850.0)
MAX_SOLAR_THRESHOLD = _optional_float("URBANFRINGEPV_MAX_SOLAR_KWH_M2")
SOLAR_THRESHOLD_YEARS = _env_int("URBANFRINGEPV_SOLAR_THRESHOLD_YEARS", 1)
SOLAR_GLOBAL_MASK_TIF = SOLAR_DIR / (
    f"ssrd_yearly_sum_merged_landhourly_wholehourly_mask_"
    f"{MIN_SOLAR_THRESHOLD:g}to{MAX_SOLAR_THRESHOLD}_min{SOLAR_THRESHOLD_YEARS}years.tif"
)
SOLAR_NETCDF_WHOLEHOURLY_DIR = _env_path(
    "URBANFRINGEPV_SOLAR_ERA5_NETCDF",
    SOLAR_DIR / "ssrd_wholehourly_month12_hour24.nc",
)
SOLAR_NETCDF_LANDHOURLY_DIR = _env_path(
    "URBANFRINGEPV_SOLAR_ERA5_LAND_NETCDF",
    SOLAR_DIR / "ssrd_landhourly_month12_hour00.nc",
)
SOLAR_TIF_WHOLEHOURLY_DIR = SOLAR_DIR / "ssrd_wholehourly_month12_hour24_transformed_tif"
SOLAR_TIF_LANDHOURLY_DIR = SOLAR_DIR / "ssrd_landhourly_month12_hour00_transformed_tif"
SOLAR_NETCDF_DATASETS = [
    (SOLAR_NETCDF_WHOLEHOURLY_DIR, "ssrd", SOLAR_TIF_WHOLEHOURLY_DIR, 0.25),
    (SOLAR_NETCDF_LANDHOURLY_DIR, "ssrd", SOLAR_TIF_LANDHOURLY_DIR, 0.1),
]
SOLAR_TIF_DATASETS = [
    (SOLAR_TIF_WHOLEHOURLY_DIR, "Surface solar radiation downwards (J m-2)"),
    (SOLAR_TIF_LANDHOURLY_DIR, "Surface solar radiation downwards (J m-2)"),
]
SOLAR_YEARLY_SUM_BY_LANDHOURLY_FOLDER = SOLAR_DIR / "yearly_sum_by_landhourly_transformed"
SOLAR_YEARLY_SUM_BY_WHOLEHOURLY_FOLDER = SOLAR_DIR / "yearly_sum_by_wholehourly_transformed"
SOLAR_YEARLY_SUM_BY_MERGE_FOLDER = SOLAR_DIR / "yearly_sum_merged_landhourly_wholehourly"
SOLAR_POINT_CORRDINATES = [
    ("Talatan Solar Park", "China", 36.132639, 100.560700, 10),
    ("Hami Solar Base", "China", 43.018173, 93.633059, 5),
    ("Dalad Solar Base", "China", 40.283114, 109.680516, 3),
    ("Bhadla Solar Park", "India", 27.487014, 71.959229, 2.25),
    ("Mohammed bin Rashid Solar Park", "UAE", 24.727431, 55.432172, 2),
    ("Copper Mountain Solar Facility", "USA", 35.776, -114.972, 0.8),
    ("Topaz Solar Farm", "USA", 35.382, -120.058, 0.55),
    ("Desert Sunlight Solar Farm", "USA", 33.761505, -115.333101, 0.55),
    ("Benban Solar Park", "Egypt", 24.442, 32.717, 1.65),
    ("Limondale Solar Farm", "Australia", -34.773708, 143.506553, 0.349),
]

# Temperature inputs, intermediates, and research thresholds.
TEMP_DIR = _env_path("URBANFRINGEPV_TEMP_DIR", OUTPUT_DIR / "temp")
MIN_TEMP_THRESHOLD = _env_float("URBANFRINGEPV_MIN_TEMP_C", -20.0)
MAX_TEMP_THRESHOLD = _optional_float("URBANFRINGEPV_MAX_TEMP_C")
TEMP_THRESHOLD_WEEKS = _env_int("URBANFRINGEPV_TEMP_THRESHOLD_WEEKS", 53)
TEMP_GLOBAL_MASK_TIF = TEMP_DIR / (
    f"temp_weekly_mean_merged_landdaily_wholehourly_mask_"
    f"{MIN_TEMP_THRESHOLD:g}to{MAX_TEMP_THRESHOLD}_min{TEMP_THRESHOLD_WEEKS}weeks.tif"
)
TEMP_NETCDF_WHOLEHOURLY_DIR = _env_path(
    "URBANFRINGEPV_TEMP_ERA5_NETCDF",
    TEMP_DIR / "temp_wholehourly_month12_hour24.nc",
)
TEMP_NETCDF_LANDDAILY_DIR1 = _env_path(
    "URBANFRINGEPV_TEMP_ERA5_LAND_H1_NETCDF",
    TEMP_DIR / "temp_landdaily_month1to6_dailymean.nc",
)
TEMP_NETCDF_LANDDAILY_DIR2 = _env_path(
    "URBANFRINGEPV_TEMP_ERA5_LAND_H2_NETCDF",
    TEMP_DIR / "temp_landdaily_month7to12_dailymean.nc",
)
TEMP_TIF_WHOLEHOURLY_DIR = TEMP_DIR / "temp_wholehourly_month12_hour24_transformed_tif"
TEMP_TIF_LANDDAILY_DIR = TEMP_DIR / "temp_landhourly_month12_hour24_transformed_tif"
TEMP_NETCDF_DATASETS = [
    (TEMP_NETCDF_WHOLEHOURLY_DIR, "t2m", TEMP_TIF_WHOLEHOURLY_DIR, 0.25),
    (TEMP_NETCDF_LANDDAILY_DIR1, "t2m", TEMP_TIF_LANDDAILY_DIR, 0.1),
    (TEMP_NETCDF_LANDDAILY_DIR2, "t2m", TEMP_TIF_LANDDAILY_DIR, 0.1),
]
TEMP_TIF_DATASETS = [
    (TEMP_TIF_WHOLEHOURLY_DIR, "2 m temperature (K)"),
    (TEMP_TIF_LANDDAILY_DIR, "2 m temperature (K)"),
]
TEMP_WEEKLY_MEAN_BY_LANDDAILY_FOLDER = TEMP_DIR / "weekly_mean_by_landdaily_transformed"
TEMP_WEEKLY_MEAN_BY_WHOLEHOURLY_FOLDER = TEMP_DIR / "weekly_mean_by_wholehourly_transformed"
TEMP_WEEKLY_MEAN_BY_MERGE_FOLDER = TEMP_DIR / "weekly_mean_merged_landdaily_wholehourly"
TEMP_POINT_CORRDINATES = SOLAR_POINT_CORRDINATES

# Wind masks consume precomputed annual maximum wind-speed rasters. This package
# intentionally does not claim to derive them from ERA5 u/v components.
WIND_DIR = _env_path("URBANFRINGEPV_WIND_DIR", OUTPUT_DIR / "wind")
WIND_YEARLY_MAX_BY_LANDHOURLY_FOLDER = _env_path(
    "URBANFRINGEPV_WIND_ERA5_LAND_MAX_DIR",
    WIND_DIR / "yearly_max_by_landhourly_transformed",
)
WIND_YEARLY_MAX_BY_WHOLEHOURLY_FOLDER = _env_path(
    "URBANFRINGEPV_WIND_ERA5_MAX_DIR",
    WIND_DIR / "yearly_max_by_wholehourly_transformed",
)
WIND_YEARLY_MAX_BY_MERGE_FOLDER = WIND_DIR / "yearly_max_merged_landhourly_wholehourly"
MIN_WIND_THRESHOLD = _optional_float("URBANFRINGEPV_MIN_WIND_M_S")
MAX_WIND_THRESHOLD = _env_float("URBANFRINGEPV_MAX_WIND_M_S", 20.8)
WIND_THRESHOLD_YEARS = _env_int("URBANFRINGEPV_WIND_THRESHOLD_YEARS", 1)
WIND_GLOBAL_MASK_TIF = WIND_DIR / (
    f"wind_yearly_max_merged_landhourly_wholehourly_mask_"
    f"{MIN_WIND_THRESHOLD}to{MAX_WIND_THRESHOLD:g}_min{WIND_THRESHOLD_YEARS}years.tif"
)

# Combined-mask outputs.
COMBINE_DIR = _env_path("URBANFRINGEPV_COMBINE_DIR", OUTPUT_DIR / "combined_masks")
TARGET_CRS = os.getenv("URBANFRINGEPV_TARGET_CRS", "EPSG:8857")
TARGET_SCALE = _env_int("URBANFRINGEPV_TARGET_SCALE_M", 20)
START_FID = _optional_int("URBANFRINGEPV_COMBINE_START_FID")
END_FID = _optional_int("URBANFRINGEPV_COMBINE_END_FID")
EACH_MASK_FILENAME_PATTERNS = {
    "land": "feature_{fid}_bare_rangeland.tif",
    "slope": f"feature_{{fid}}_slope_let{SLOPE_DEGREE_THRESHOLD:g}_mask.tif",
    "ssrd": "feature_{fid}_ssrd_mask.tif",
    "temp": "feature_{fid}_temp_mask.tif",
    "wind": "feature_{fid}_wind_mask.tif",
}
DST_RES = _env_int("URBANFRINGEPV_GLOBAL_MOSAIC_RES_M", 1000)
GLOBAL_COMBINED_MASK_TIF = COMBINE_DIR / f"global_combined_mask_{DST_RES}.tif"
