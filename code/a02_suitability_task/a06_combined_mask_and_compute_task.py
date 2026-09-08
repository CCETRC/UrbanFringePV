"""Combine per-feature suitability masks and calculate suitable area."""

import argparse

from a02_suitability_task.suitable_utils import (
    batch_process,
    calculate_mask_areas,
    combine_feature_masks,
    mosaic_resample,
)
from config.base_config import OUTPUT_DIR
from config.suitability_config import (
    BUFFER_PROJ_SHP_PATH,
    COMBINE_DIR,
    CURRENT_PV_DIR,
    DST_RES,
    EACH_MASK_FILENAME_PATTERNS,
    END_FID,
    GLOBAL_COMBINED_MASK_TIF,
    LAND_FEATURE_MASK_DIR,
    SLOPE_FEATURE_MASK_DIR,
    START_FID,
    TARGET_CRS,
    TARGET_SCALE,
)

def combine_mask_task():
    """Align five input masks, intersect them, and clip each feature."""
    combine_feature_masks(
        land_dir=LAND_FEATURE_MASK_DIR,
        slope_dir=SLOPE_FEATURE_MASK_DIR,
        ssrd_dir=OUTPUT_DIR/'ssrd'/'per_feature_masks',
        temp_dir=OUTPUT_DIR/'temp'/'per_feature_masks',
        wind_dir=OUTPUT_DIR / 'wind' / 'per_feature_masks',
        output_dir=COMBINE_DIR/'per_feature_masks',
        n_processes=None,
        target_crs=TARGET_CRS,target_scale = TARGET_SCALE,
        filename_patterns = EACH_MASK_FILENAME_PATTERNS,
        start_fid =START_FID,
        end_fid =END_FID,
        buffer_proj_shp_path=BUFFER_PROJ_SHP_PATH,
    )

def remove_current_pv_task():
    """Remove pixels occupied by the configured 2022 photovoltaic rasters."""
    batch_process(feature_dir=COMBINE_DIR/'per_feature_masks',
                  pv_dir=CURRENT_PV_DIR,
                  output_dir=COMBINE_DIR/'per_feature_masks_rm_current_pv',
                  num_workers=8,
                  start_fid =START_FID,end_fid =END_FID)


def calculate_areas_task():
    """Calculate suitable area from the current-PV-removed masks."""
    calculate_mask_areas(tif_folder=COMBINE_DIR/'per_feature_masks_rm_current_pv',
                          pattern="feature_*_combined_mask.tif",
                          output_csv=COMBINE_DIR/'per_feature_masks_rm_current_pv'/"mask_areas_summary.csv",
                          target_crs=TARGET_CRS, target_scale=TARGET_SCALE,
                          )

def global_combined_mask_task():
    """Mosaic per-feature masks to the configured global resolution."""
    mosaic_resample(tif_dir=COMBINE_DIR/'per_feature_masks',
                      output_tif=GLOBAL_COMBINED_MASK_TIF, dst_res=DST_RES)

def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("combine", "remove-current-pv", "calculate-area", "mosaic"),
        help="Run one explicit workflow stage.",
    )
    return parser


def main(argv=None):
    """Run one explicitly selected combined-mask stage."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.stage is None:
        parser.print_help()
        return 0
    stages = {
        "combine": combine_mask_task,
        "remove-current-pv": remove_current_pv_task,
        "calculate-area": calculate_areas_task,
        "mosaic": global_combined_mask_task,
    }
    stages[args.stage]()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
