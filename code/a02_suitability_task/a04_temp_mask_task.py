"""Build temperature suitability masks from local ERA5 inputs."""

import argparse

import pandas as pd

from a02_suitability_task.ERA5_data_utils import (
    accumulate_average_data,
    batch_merge_all,
    batch_plot_tif_compute_min_max_mean,
    export_netcdf_to_transformed_tif,
    extract_and_plot,
)
from a02_suitability_task.suitable_utils import (
    plot_tif,
    split_global_mask_by_features,
    weekly_aver_mask_by_local_weekly_aver_data,
)
from config.suitability_config import (
    BASE_SHP_PATH,
    BUFFER_SHP,
    MAX_TEMP_THRESHOLD,
    MIN_TEMP_THRESHOLD,
    TEMP_DIR,
    TEMP_GLOBAL_MASK_TIF,
    TEMP_NETCDF_DATASETS,
    TEMP_POINT_CORRDINATES,
    TEMP_THRESHOLD_WEEKS,
    TEMP_TIF_DATASETS,
    TEMP_TIF_LANDDAILY_DIR,
    TEMP_TIF_WHOLEHOURLY_DIR,
    TEMP_WEEKLY_MEAN_BY_LANDDAILY_FOLDER,
    TEMP_WEEKLY_MEAN_BY_MERGE_FOLDER,
    TEMP_WEEKLY_MEAN_BY_WHOLEHOURLY_FOLDER,
)


def _threshold_label():
    lower = f"≥ {MIN_TEMP_THRESHOLD:g}" if MIN_TEMP_THRESHOLD is not None else None
    upper = f"≤ {MAX_TEMP_THRESHOLD:g}" if MAX_TEMP_THRESHOLD is not None else None
    bounds = " and ".join(value for value in (lower, upper) if value) or "all values"
    return f"{bounds} °C for at least {TEMP_THRESHOLD_WEEKS} week(s)"

def netcdf_to_tif_task():
    """Convert configured ERA5 NetCDF variables to georeferenced TIFFs."""
    for nc_path, indicator, output_folder,resolution  in TEMP_NETCDF_DATASETS:
        export_netcdf_to_transformed_tif(nc_path,indicator, output_folder,resolution)

def batch_tif_plot_task():
    """Plot converted TIFFs and export their summary statistics."""
    for tif_path, label in TEMP_TIF_DATASETS:
        batch_plot_tif_compute_min_max_mean(tif_path, label)


def accumulate_average_task():
    """Aggregate global hourly and ERA5-Land daily temperature to weeks."""
    # Aggregate the global hourly product to weekly mean temperature.
    accumulate_average_data(TEMP_TIF_WHOLEHOURLY_DIR,TEMP_WEEKLY_MEAN_BY_WHOLEHOURLY_FOLDER,
                            frequency = 'week',compute_method='avg',unit = 'C')
    batch_plot_tif_compute_min_max_mean(TEMP_WEEKLY_MEAN_BY_WHOLEHOURLY_FOLDER, label ='weekly average temp (C)',
                                        base_shp_path=BASE_SHP_PATH)

    # Aggregate the ERA5-Land daily product to weekly mean temperature.
    accumulate_average_data(TEMP_TIF_LANDDAILY_DIR,TEMP_WEEKLY_MEAN_BY_LANDDAILY_FOLDER,
                            frequency = 'week',compute_method='avg',unit = 'C')
    batch_plot_tif_compute_min_max_mean(TEMP_WEEKLY_MEAN_BY_LANDDAILY_FOLDER, label ='weekly average temp (C)',
                                        base_shp_path=BASE_SHP_PATH)


def merge_average_tif_task():
    """Merge like-named ERA5-Land and ERA5 global weekly rasters."""
    batch_merge_all(TEMP_WEEKLY_MEAN_BY_LANDDAILY_FOLDER, TEMP_WEEKLY_MEAN_BY_WHOLEHOURLY_FOLDER,
                    TEMP_WEEKLY_MEAN_BY_MERGE_FOLDER)
    batch_plot_tif_compute_min_max_mean(TEMP_WEEKLY_MEAN_BY_MERGE_FOLDER,
                                        label ='Weekly Average Temperature(°C)',
                                        base_shp_path=BASE_SHP_PATH)


def find_temp_by_coordinates_task():
    """Extract weekly temperature at the configured reference sites."""
    df = pd.DataFrame(TEMP_POINT_CORRDINATES, columns=["name", "country",  "lat", "lon", "capacity_GW"])
    extract_and_plot(
        df_points=df,
        raster_folder=TEMP_WEEKLY_MEAN_BY_MERGE_FOLDER,
        output_folder=TEMP_DIR
    )

def global_temp_mask_tif_plot_task():
    """Create and plot the global temperature suitability mask."""
    # global mask
    weekly_aver_mask_by_local_weekly_aver_data(
        input_folder=TEMP_WEEKLY_MEAN_BY_MERGE_FOLDER,
        min_threshold=MIN_TEMP_THRESHOLD,
        max_threshold=MAX_TEMP_THRESHOLD,
        min_weeks = TEMP_THRESHOLD_WEEKS,
        output_path=TEMP_GLOBAL_MASK_TIF)

    plot_tif(
        TEMP_GLOBAL_MASK_TIF,
        label=None,
        output_folder=TEMP_DIR,
        base_shp_path=BASE_SHP_PATH,
        title="Temperature Suitability Map",
        categorical=True,
        category_values=[0, 1],
        categories=["Other", _threshold_label()],
    )

def split_global_temp_mask_to_features_task():
    """Clip the global temperature mask to each buffer feature."""
    split_global_mask_by_features(
        global_mask_tif=TEMP_GLOBAL_MASK_TIF,
        data_type = "temp",
        buffer_shp=BUFFER_SHP,
        output_folder=TEMP_DIR/"per_feature_masks",
        feat_id="feat_id",
        buffer_dist=10000
    )

def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "netcdf-to-tif",
            "plot-inputs",
            "aggregate",
            "merge",
            "global-mask",
            "split",
            "query-points",
        ),
        help="Run one explicit workflow stage.",
    )
    return parser


def main(argv=None):
    """Run one explicitly selected temperature-mask stage."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.stage is None:
        parser.print_help()
        return 0
    stages = {
        "netcdf-to-tif": netcdf_to_tif_task,
        "plot-inputs": batch_tif_plot_task,
        "aggregate": accumulate_average_task,
        "merge": merge_average_tif_task,
        "global-mask": global_temp_mask_tif_plot_task,
        "split": split_global_temp_mask_to_features_task,
        "query-points": find_temp_by_coordinates_task,
    }
    stages[args.stage]()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
