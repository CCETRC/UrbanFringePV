"""Build solar-radiation suitability masks from local ERA5 inputs."""

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
    MAX_SOLAR_THRESHOLD,
    MIN_SOLAR_THRESHOLD,
    SOLAR_DIR,
    SOLAR_GLOBAL_MASK_TIF,
    SOLAR_NETCDF_DATASETS,
    SOLAR_POINT_CORRDINATES,
    SOLAR_THRESHOLD_YEARS,
    SOLAR_TIF_DATASETS,
    SOLAR_TIF_LANDHOURLY_DIR,
    SOLAR_TIF_WHOLEHOURLY_DIR,
    SOLAR_YEARLY_SUM_BY_LANDHOURLY_FOLDER,
    SOLAR_YEARLY_SUM_BY_MERGE_FOLDER,
    SOLAR_YEARLY_SUM_BY_WHOLEHOURLY_FOLDER,
)


def _threshold_label():
    lower = f"≥ {MIN_SOLAR_THRESHOLD:g}" if MIN_SOLAR_THRESHOLD is not None else None
    upper = f"≤ {MAX_SOLAR_THRESHOLD:g}" if MAX_SOLAR_THRESHOLD is not None else None
    bounds = " and ".join(value for value in (lower, upper) if value) or "all values"
    return f"{bounds} kWh m⁻² for at least {SOLAR_THRESHOLD_YEARS} year(s)"


def netcdf_to_tif_task():
    """Convert configured ERA5 NetCDF variables to georeferenced TIFFs."""
    for nc_path, indicator, output_folder,resolution  in SOLAR_NETCDF_DATASETS:
        export_netcdf_to_transformed_tif(nc_path,indicator, output_folder,resolution)

def batch_tif_plot_task():
    """Plot converted TIFFs and export their summary statistics."""
    for tif_path, label in SOLAR_TIF_DATASETS:
        batch_plot_tif_compute_min_max_mean(tif_path, label)

def accumulate_average_task():
    """Aggregate hourly solar radiation into annual kWh m-2 sums."""
    # Input filenames must follow the timestamp pattern produced by the
    # NetCDF-to-TIFF conversion stage.
    accumulate_average_data(SOLAR_TIF_LANDHOURLY_DIR,SOLAR_YEARLY_SUM_BY_LANDHOURLY_FOLDER,
                            frequency = 'year',compute_method='sum',unit ='kWh/m²')
    batch_plot_tif_compute_min_max_mean(SOLAR_YEARLY_SUM_BY_LANDHOURLY_FOLDER,
                                        label = 'yearly Surface Solar Radiation Downwards (kWh/m²)',
                                        base_shp_path=BASE_SHP_PATH)

    accumulate_average_data(SOLAR_TIF_WHOLEHOURLY_DIR,
                            SOLAR_YEARLY_SUM_BY_WHOLEHOURLY_FOLDER,frequency = 'year',compute_method='sum',unit ='kWh/m²')
    batch_plot_tif_compute_min_max_mean(SOLAR_YEARLY_SUM_BY_WHOLEHOURLY_FOLDER,
                                        label = 'yearly Surface Solar Radiation Downwards (kWh/m²)',
                                        base_shp_path=BASE_SHP_PATH)

def merge_average_tif_task():
    """Merge like-named ERA5-Land and ERA5 global annual rasters."""
    batch_merge_all(SOLAR_YEARLY_SUM_BY_LANDHOURLY_FOLDER, SOLAR_YEARLY_SUM_BY_WHOLEHOURLY_FOLDER, SOLAR_YEARLY_SUM_BY_MERGE_FOLDER)
    batch_plot_tif_compute_min_max_mean(SOLAR_YEARLY_SUM_BY_MERGE_FOLDER,
                                        label ='yearly Surface Solar Radiation Downwards (kWh/m²)',
                                        base_shp_path=BASE_SHP_PATH)

def global_solar_mask_tif_plot_task():
    """Create and plot the global solar suitability mask."""
    weekly_aver_mask_by_local_weekly_aver_data(
        input_folder=SOLAR_YEARLY_SUM_BY_MERGE_FOLDER,
        min_threshold=MIN_SOLAR_THRESHOLD,
        max_threshold=MAX_SOLAR_THRESHOLD,
        min_weeks = SOLAR_THRESHOLD_YEARS,
        output_path=SOLAR_GLOBAL_MASK_TIF)

    plot_tif(SOLAR_GLOBAL_MASK_TIF, label=None, output_folder=SOLAR_DIR,
             base_shp_path=BASE_SHP_PATH,title = 'Annual Surface Solar Radiation Downwards (kWh m⁻²)',
             categorical=True,category_values=[0, 1],categories=['Other', _threshold_label()]
)

def split_global_solar_mask_to_features_task():
    """Clip the global solar mask to each configured buffer feature."""
    # list_shp_fields(buffer_shp)
    split_global_mask_by_features(
        global_mask_tif=SOLAR_GLOBAL_MASK_TIF,
        data_type = "ssrd",
        buffer_shp=BUFFER_SHP,
        output_folder= SOLAR_DIR / "per_feature_masks",
        feat_id="feat_id",
        buffer_dist=10000
    )

def find_solar_by_coordinates_task():
    """Extract annual solar radiation at the configured reference sites."""
    df = pd.DataFrame(SOLAR_POINT_CORRDINATES, columns=["name", "country",  "lat", "lon", "capacity_GW"])
    extract_and_plot(
        df_points=df,
        raster_folder=SOLAR_YEARLY_SUM_BY_MERGE_FOLDER,
        output_folder=SOLAR_DIR
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
    """Run one explicitly selected solar-mask stage."""
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
        "global-mask": global_solar_mask_tif_plot_task,
        "split": split_global_solar_mask_to_features_task,
        "query-points": find_solar_by_coordinates_task,
    }
    stages[args.stage]()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
