"""Build slope suitability masks from FABDEM Earth Engine exports."""

import argparse
import time

import ee

from a02_suitability_task.suitable_utils import (
    export_tif_to_drive,
    generate_masks_from_tifs,
    initialize_ee,
    mosaic_global_tif,
    plot_tif,
    slope_mask_gee,
)
from config.suitability_config import (
    BASE_SHP_PATH,
    BATCH_SIZE,
    END_ID,
    GEE_BOUNDARIES,
    GEE_PROJECT_ID,
    SLOPE_DEGREE_THRESHOLD,
    SLOPE_DIR,
    SLOPE_FEATURE_MASK_DIR,
    SLOPE_GLOBAL_MASK_TIF,
    SLOPE_SOURCE_TIF_DIR,
    START_ID,
)


def _load_boundaries():
    """Load and optionally filter the configured Earth Engine boundaries."""
    if not GEE_BOUNDARIES:
        raise RuntimeError(
            "Set URBANFRINGEPV_GEE_BOUNDARIES to a readable Earth Engine asset."
        )
    boundaries = ee.FeatureCollection(GEE_BOUNDARIES)
    subset = boundaries
    if START_ID is not None:
        subset = subset.filter(ee.Filter.gte("feat_id", START_ID))
    if END_ID is not None:
        subset = subset.filter(ee.Filter.lte("feat_id", END_ID))
    return boundaries, subset


def original_slope_export_task():
    """Submit 30 m per-feature slope exports with a 300 m safety buffer."""
    initialize_ee(project_id=GEE_PROJECT_ID)
    boundaries, subset = _load_boundaries()

    size = boundaries.size().getInfo()
    print(f"Boundary feature count: {size}")
    ids = subset.aggregate_array('feat_id').getInfo()
    ids.sort()
    total = len(ids)
    print(f"Selected {total} features (start={START_ID}, end={END_ID}).")

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_ids = ids[start:end]
        print(f"\nProcessing batch {batch_ids}:")

        for fid in batch_ids:
            feat = boundaries.filter(ee.Filter.eq('feat_id', fid)).first()
            geom = feat.geometry()
            slope_mask_gee(geom, fid, SLOPE_DEGREE_THRESHOLD,
                           buffer_dist=300, export = True,crs = 'EPSG:4326',scale= 30)

        print('Sleeping 10 seconds between export batches...')
        time.sleep(10)

def slope_mask_from_tifs_task():
    """Convert downloaded per-feature slope rasters into binary masks."""
    generate_masks_from_tifs(
        input_folder=SLOPE_SOURCE_TIF_DIR,
        output_folder=SLOPE_FEATURE_MASK_DIR,
        threshold=SLOPE_DEGREE_THRESHOLD,
    )


def global_mask_from_gee():
    """Submit six coarse global FABDEM slope-mask tiles to Google Drive."""
    initialize_ee(project_id=GEE_PROJECT_ID)

    # Load the FABDEM digital terrain model collection.
    fabdem = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")

    # Divide the projectable EPSG:3857 extent into six export regions.
    xmin, ymin = -20037508.342789244, -20037508.342789244
    xmax, ymax = 20037508.342789244, 20037508.342789244

    x_third1 = xmin + (xmax - xmin) / 3
    x_third2 = xmin + (xmax - xmin) * 2 / 3
    y_mid = 0.0

    region1 = [xmin, ymin, x_third1, y_mid]
    region2 = [x_third1, ymin, x_third2, y_mid]
    region3 = [x_third2, ymin, xmax, y_mid]
    region4 = [xmin, y_mid, x_third1, ymax]
    region5 = [x_third1, y_mid, x_third2, ymax]
    region6 = [x_third2, y_mid, xmax, ymax]

    regions_3857 = {
        'region1': region1,
        'region2': region2,
        'region3': region3,
        'region4': region4,
        'region5': region5,
        'region6': region6
    }

    for region_name, region_geom in regions_3857.items():
        # Keep the export geometry in the same projected CRS as the raster.
        rect3857 = ee.Geometry.Rectangle(region_geom, proj='EPSG:3857', geodesic=False)

        elev = (
            fabdem.mosaic()
            .setDefaultProjection('EPSG:3857', None, 30)
            .clip(rect3857)
        )
        slope = ee.Terrain.slope(elev)
        # Pixels at or below the configured threshold are suitable.
        slope_mask = slope.lte(SLOPE_DEGREE_THRESHOLD).toUint8()

        # Exclude cells at or below sea level from the exported land mask.
        ocean = elev.lte(0)
        slope_mask_land = slope_mask.updateMask(ocean.Not()).rename('slope_mask')

        export_tif_to_drive(slope_mask_land,title = f'slope_mask_le{SLOPE_DEGREE_THRESHOLD:g}deg_scale1000_{region_name}',
                            boundary=rect3857,crs='EPSG:3857',scale=1000)

def plot_global_slope_mask():
    """Mosaic downloaded global slope tiles and create the slope map."""
    mosaic_global_tif(tif_dir = SLOPE_DIR/"global_slope_mask", output_tif= SLOPE_GLOBAL_MASK_TIF)

    plot_tif(SLOPE_GLOBAL_MASK_TIF, label=None, output_folder=SLOPE_DIR,
             base_shp_path=BASE_SHP_PATH, title='Slope Suitability Map',
             categorical=True, category_values=[1],
             categories=[f'Slope ≤ {SLOPE_DEGREE_THRESHOLD:g}°']
             )


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "per-feature-export",
            "build-per-feature-masks",
            "global-export",
            "plot-global",
        ),
        help="Run one explicit workflow stage.",
    )
    return parser


def main(argv=None):
    """Run one explicitly selected slope-mask stage."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.stage is None:
        parser.print_help()
        return 0
    stages = {
        "per-feature-export": original_slope_export_task,
        "build-per-feature-masks": slope_mask_from_tifs_task,
        "global-export": global_mask_from_gee,
        "plot-global": plot_global_slope_mask,
    }
    stages[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
