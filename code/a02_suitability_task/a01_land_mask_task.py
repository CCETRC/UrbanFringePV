"""Build land-cover suitability masks from Earth Engine exports."""

import argparse
import time

import ee

from a02_suitability_task.suitable_utils import (
    export_tif_to_drive,
    initialize_ee,
    landcover_mask,
    mosaic_global_tif,
    plot_tif,
)
from config.suitability_config import (
    BASE_SHP_PATH,
    BATCH_SIZE,
    END_ID,
    GEE_BOUNDARIES,
    GEE_PROJECT_ID,
    LAND_DIR,
    LAND_GLOBAL_MASK_TIF,
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


def land_mask_task():
    """Submit per-feature 2023 bare-ground and rangeland exports to Drive."""
    initialize_ee(project_id=GEE_PROJECT_ID)
    boundaries, subset = _load_boundaries()

    size = boundaries.size().getInfo()
    print(f"Boundary feature count: {size}")
    ids = subset.aggregate_array("feat_id").getInfo()
    ids.sort()
    total = len(ids)
    print(f"Selected {total} features (start={START_ID}, end={END_ID}).")

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_ids = ids[start:end]
        print(f"\nProcessing batch {batch_ids}:")

        for fid in batch_ids:
            feat = boundaries.filter(ee.Filter.eq("feat_id", fid)).first()
            geom = feat.geometry()
            landcover_mask(
                geom,
                fid,
                buffer_dist=150,
                export=True,
                crs="EPSG:4326",
                scale=10,
            )

        print("Sleeping 10 seconds between export batches...")
        time.sleep(10)


def global_mask_from_gee():
    """Submit six coarse global land-cover tiles to Google Drive."""
    initialize_ee(project_id=GEE_PROJECT_ID)

    def remap_lulc(image):
        # Remap ESRI classes: [1,2,4,5,7,8,9,10,11] -> [1..9]
        return image.remap([1, 2, 4, 5, 7, 8, 9, 10, 11], list(range(1, 10)))

    # Load the public Esri 2023 land-cover collection.
    esri_lulc = ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS")

    land_mosaic = esri_lulc.filterDate('2023-01-01', '2023-12-31').map(remap_lulc).mosaic()

    # Extract bare ground (class 6) and rangeland (9)
    bare_mask = land_mosaic.eq(6).Or(land_mosaic.eq(9))
    bare_ground = land_mosaic.updateMask(bare_mask)

    # Split the export into six tiles covering latitudes from 60°S to 90°N.
    regions = {
        "region1": ee.Geometry.Rectangle([-180, -60, -60, 0]),
        "region2": ee.Geometry.Rectangle([-180, 0, -60, 90]),
        "region3": ee.Geometry.Rectangle([-60, -60, 60, 0]),
        "region4": ee.Geometry.Rectangle([-60, 0, 60, 90]),
        "region5": ee.Geometry.Rectangle([60, -60, 180, 0]),
        "region6": ee.Geometry.Rectangle([60, 0, 180, 90]),
    }

    for region_name, region_geom in regions.items():
        export_tif_to_drive(
            bare_ground.clip(region_geom),
            title=f"land_mask_BareRangeland_scale1000_{region_name}",
            boundary=region_geom,
            crs="EPSG:4326",
            scale=1000,
        )


def plot_global_land_mask():
    """Mosaic downloaded global land tiles and create the land-cover map."""
    mosaic_global_tif(tif_dir = LAND_DIR/"global_land_mask", output_tif= LAND_GLOBAL_MASK_TIF)

    plot_tif(LAND_GLOBAL_MASK_TIF, label=None, output_folder=LAND_DIR,
             base_shp_path=BASE_SHP_PATH, title='Land Cover Suitability Map',
             categorical=True, category_values=[6,9],
             categories=['Bare Ground','Rangeland'],
             )
def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("per-feature-export", "global-export", "plot-global"),
        help="Run one explicit workflow stage.",
    )
    return parser


def main(argv=None):
    """Run one explicitly selected land-mask stage."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.stage is None:
        parser.print_help()
        return 0
    stages = {
        "per-feature-export": land_mask_task,
        "global-export": global_mask_from_gee,
        "plot-global": plot_global_land_mask,
    }
    stages[args.stage]()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
