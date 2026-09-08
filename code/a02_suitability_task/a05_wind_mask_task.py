"""Build wind suitability masks from precomputed annual-maximum rasters.

This module does not derive wind speed from ERA5 u/v components. The two
configured input directories must contain like-named, precomputed annual
maximum wind-speed TIFFs for ERA5-Land and ERA5 global data.
"""

import argparse
from pathlib import Path

from a02_suitability_task.ERA5_data_utils import (
    batch_merge_all,
    batch_plot_tif_compute_min_max_mean,
)
from a02_suitability_task.suitable_utils import (
    plot_tif,
    split_global_mask_by_features,
    weekly_aver_mask_by_local_weekly_aver_data,
)
from config.suitability_config import (
    BASE_SHP_PATH,
    BUFFER_SHP,
    MAX_WIND_THRESHOLD,
    MIN_WIND_THRESHOLD,
    WIND_DIR,
    WIND_GLOBAL_MASK_TIF,
    WIND_THRESHOLD_YEARS,
    WIND_YEARLY_MAX_BY_LANDHOURLY_FOLDER,
    WIND_YEARLY_MAX_BY_MERGE_FOLDER,
    WIND_YEARLY_MAX_BY_WHOLEHOURLY_FOLDER,
)


def _require_tifs(path, description):
    directory = Path(path)
    if not directory.is_dir() or not any(directory.glob("*.tif")):
        raise FileNotFoundError(
            f"Missing {description}: {directory}. "
            "Provide precomputed annual-maximum wind-speed TIFFs; this package "
            "does not derive them from ERA5 u/v components."
        )


def _threshold_label():
    lower = f"≥ {MIN_WIND_THRESHOLD:g}" if MIN_WIND_THRESHOLD is not None else None
    upper = f"≤ {MAX_WIND_THRESHOLD:g}" if MAX_WIND_THRESHOLD is not None else None
    bounds = " and ".join(value for value in (lower, upper) if value) or "all values"
    return f"{bounds} m s⁻¹ for at least {WIND_THRESHOLD_YEARS} year(s)"


def merge_average_tif_task():
    """Merge like-named precomputed ERA5-Land and ERA5 global rasters."""
    _require_tifs(WIND_YEARLY_MAX_BY_LANDHOURLY_FOLDER, "ERA5-Land wind input")
    _require_tifs(WIND_YEARLY_MAX_BY_WHOLEHOURLY_FOLDER, "ERA5 global wind input")
    batch_merge_all(WIND_YEARLY_MAX_BY_LANDHOURLY_FOLDER, WIND_YEARLY_MAX_BY_WHOLEHOURLY_FOLDER, WIND_YEARLY_MAX_BY_MERGE_FOLDER)
    batch_plot_tif_compute_min_max_mean(WIND_YEARLY_MAX_BY_MERGE_FOLDER,
                                        label ='Annual Max Wind Speed (m/s)',
                                        base_shp_path=BASE_SHP_PATH)

def global_wind_mask_tif_plot_task():
    """Create and plot the global wind suitability mask."""
    _require_tifs(WIND_YEARLY_MAX_BY_MERGE_FOLDER, "merged wind input")
    weekly_aver_mask_by_local_weekly_aver_data(
        input_folder=WIND_YEARLY_MAX_BY_MERGE_FOLDER,
        min_threshold=MIN_WIND_THRESHOLD,
        max_threshold=MAX_WIND_THRESHOLD,
        min_weeks = WIND_THRESHOLD_YEARS,
        output_path=WIND_GLOBAL_MASK_TIF)

    plot_tif(WIND_GLOBAL_MASK_TIF, label=None, output_folder=WIND_DIR,
             base_shp_path=BASE_SHP_PATH,title = 'Annual Max Wind Speed (m/s)',
             categorical=True,category_values=[0,1],categories=['Other', _threshold_label()],
)

def split_global_wind_mask_to_features_task():
    """Clip the global wind mask to each configured buffer feature."""
    if not Path(WIND_GLOBAL_MASK_TIF).is_file():
        raise FileNotFoundError(
            f"Global wind mask not found: {WIND_GLOBAL_MASK_TIF}. "
            "Run the global-mask stage after supplying precomputed inputs."
        )
    split_global_mask_by_features(
        global_mask_tif=WIND_GLOBAL_MASK_TIF,
        data_type = "wind",
        buffer_shp=BUFFER_SHP,
        output_folder= WIND_DIR / "per_feature_masks",
        feat_id="feat_id",
        buffer_dist=10000
    )

def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("merge-precomputed", "global-mask", "split"),
        help="Run one explicit workflow stage.",
    )
    return parser


def main(argv=None):
    """Run one explicitly selected wind-mask stage."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.stage is None:
        parser.print_help()
        return 0
    stages = {
        "merge-precomputed": merge_average_tif_task,
        "global-mask": global_wind_mask_tif_plot_task,
        "split": split_global_wind_mask_to_features_task,
    }
    stages[args.stage]()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
