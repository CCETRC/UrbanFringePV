import glob
import os
import re
import tempfile
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import datashader.transfer_functions as tf
import ee
import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
from datashader import Canvas
from matplotlib.colors import BoundaryNorm, ListedColormap, to_hex
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
from osgeo import gdal
import pandas as pd
from pyproj import CRS as pyprojCRS, Transformer
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
    transform_geom,
)
import rioxarray as rxr
from shapely.geometry import box, mapping, shape
from tqdm import tqdm



def mosaic_global_tif(tif_dir, output_tif):
    """Merge all TIFF files in a folder and export one mosaic."""
    tif_files = glob.glob(f"{tif_dir}/*.tif")
    if not tif_files:
        raise FileNotFoundError(f"No TIFF files found in {tif_dir}")

    src_files_to_mosaic = [rasterio.open(fp) for fp in tif_files]
    try:
        mosaic, out_trans = merge(src_files_to_mosaic)
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "lzw"
        })

        os.makedirs(os.path.dirname(os.path.abspath(output_tif)), exist_ok=True)
        with rasterio.open(output_tif, "w", **out_meta) as dest:
            dest.write(mosaic)
    finally:
        for source in src_files_to_mosaic:
            source.close()

    print(f"Mosaic completed! Output saved to {output_tif}")

def plot_tif_datashader(
    input_tif,
    output_folder,
    downsample_factor=10,
    base_shp_path=None,
    categorical=False,
    categories=None,
    cmap="viridis",
    plot_width=None,
    plot_height=None,
):
    """
    Render a GeoTIFF with an optional vector-boundary overlay.

    :param input_tif: Path to the input TIFF.
    :param output_folder: Folder for the output PNG.
    :param downsample_factor: Downsampling factor; 1 preserves native resolution.
    :param base_shp_path: Optional vector-boundary path.
    :param categorical: Whether to render the raster as categorical data.
    :param categories: Category labels corresponding to values 0..n.
    :param cmap: Colormap name.
    :param plot_width: Datashader canvas width; None uses the raster width.
    :param plot_height: Datashader canvas height; None uses the raster height.
    """
    # Read the first raster band.
    da = rxr.open_rasterio(input_tif, masked=True).isel(band=0)
    if downsample_factor > 1:
        da = da.coarsen(x=downsample_factor, y=downsample_factor, boundary='trim').mean()
    h, w = da.shape  # h rows, w columns

    # Mask zero values so they are not rendered.
    data = da.where(da != 0)

    # Coordinates and dimensions.
    x0, x1 = data.x.min().item(), data.x.max().item()
    y0, y1 = data.y.min().item(), data.y.max().item()

    # Canvas dimensions.
    pw = plot_width or w
    ph = plot_height or h
    cvs = Canvas(plot_width=pw, plot_height=ph, x_range=(x0, x1), y_range=(y0, y1))

    # Aggregate for rendering.
    agg = cvs.raster(data)

    # 3. Select the rendering method.
    if categorical:
        if categories is None:
            categories = [str(int(v)) for v in np.unique(data.values)]
        n = len(categories)
        base = plt.get_cmap(cmap, n)
        cmap_disc = ListedColormap(base.colors[:n])
        # Convert RGBA values to hexadecimal colors.
        color_key = {i: to_hex(cmap_disc(i)) for i in range(n)}
        img = tf.shade(agg, color_key=color_key, how='eq_hist')
    else:
        img = tf.shade(agg, cmap=plt.get_cmap(cmap), how='linear')

    # Set a transparent background.
    img = tf.set_background(img, None)

    if base_shp_path:
        gdf = gpd.read_file(base_shp_path)
        if gdf.crs != data.rio.crs:
            gdf = gdf.to_crs(data.rio.crs)
        # gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
        # Aggregate vector lines with Datashader.
        line_agg = cvs.line(gdf, geometry='geometry')
        line_img = tf.shade(line_agg, color="black", how='eq_hist')
        # Overlay the line layer.
        img = tf.stack(img, line_img, how='over')

    # 6. Add a legend with Matplotlib and preserve the raster dimensions.
    fig, ax = plt.subplots(
        figsize=(w / 100, h / 100), dpi=100
    )
    # Display the Datashader image.
    ax.imshow(img.to_pil(), origin='lower', extent=(x0, x1, y0, y1))

    if categorical:
        handles = [
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=cmap_disc(i), markersize=8,
                       label=categories[i])
            for i in range(n)
        ]
        ax.legend(handles=handles, title="Category",
                  loc='lower center', ncol=n, bbox_to_anchor=(0.5, -0.15))

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(os.path.splitext(os.path.basename(input_tif))[0])
    plt.tight_layout()

    # 5. Save the image.
    os.makedirs(output_folder, exist_ok=True)
    out_png = os.path.join(output_folder,
                           os.path.splitext(os.path.basename(input_tif))[0] + '.png')
    fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0,transparent=True)
    plt.close(fig)
    print(f"Saved: {out_png}")


def plot_tif(input_tif, label,output_folder,base_shp_path=None,
             title=None,
             categorical=False,
             category_values=None,
             categories=None,
             cmap="viridis",
):
    """
    :param input_tif: Path to the input TIFF.
    :param label: Unit label, for example J/m2, kWh/m2, or temperature in K.
    :param output_folder: Folder for the output image.
    :param base_shp_path: Optional vector-boundary path, such as a shapefile.
    :param title: Plot title; defaults to the filename when None.
    :param categorical: Whether to display a categorical legend.
    :param category_values: Original raster values, for example [6, 9].
    :param categories: Corresponding labels, such as Bare Ground and Rangeland.
    :param cmap: Colormap name.
    :return: None.
    """
    da = rxr.open_rasterio(input_tif)

    vmin =da.min().item()
    vmax = da.max().item()
    vmean = da.mean().item()  # Calculate the mean.

    # Extract the filename without its extension.
    fname = os.path.splitext(os.path.basename(input_tif))[0]

    # 2. Render the raster.
    fig, ax = plt.subplots(figsize=(10, 6))
    if categorical:
        # Render zero values as transparent.
        da = da.where(da != 0)
        # Both category values and labels are required.
        if category_values is None or categories is None:
            raise ValueError("category_values and categories must both be provided")
        if len(category_values) != len(categories):
            raise ValueError("category_values and categories must have equal lengths")

        # Build the category-code array.
        codes = np.full(da.values.shape, np.nan)
        for i, val in enumerate(category_values):
            codes[da.values == val] = i
        da_codes = rxr.open_rasterio(input_tif).copy(data=codes)

        # 1) Create a discrete colormap.
        base_cmap = plt.get_cmap(cmap, len(categories))
        listed_cmap = ListedColormap(base_cmap.colors[:len(categories)])

        # 2) Give each integer category one color interval.
        bounds = np.arange(-0.5, len(categories) + 0.5, 1)
        norm = BoundaryNorm(bounds, ncolors=len(categories), clip=True)
        # 3) Draw the raster without an automatic colorbar.
        da_codes.plot(ax=ax, cmap=listed_cmap, norm=norm, add_colorbar=False)

        # 4) Build custom legend handles.
        handles = [
            Patch(facecolor=listed_cmap(i), edgecolor='black', label=categories[i])
            for i in range(len(categories))
        ]
        ax.legend(handles=handles, title=label,
                  loc='lower center',
                  bbox_to_anchor=(0.5, -0.18),  # Centered horizontally below the axes.
                 ncol=len(categories),        # Show every category on one row.
                 borderaxespad=0.0,             # Distance between legend and axes.
                    frameon=True)

    else:
        # Use the continuous-color rendering path.
        da.plot(ax=ax, cmap=cmap, robust=True,
                cbar_kwargs={"label": label})

    # 4.2 Overlay the vector boundary on the same axes.
    # Read the GeoDataFrame when a base-map path is provided.
    if base_shp_path:
        base_gdf = gpd.read_file(base_shp_path)
        # Reproject the boundary when required.
        if base_gdf.crs != da.rio.crs:
            base_gdf = base_gdf.to_crs(da.rio.crs)

        base_gdf.plot(ax=ax, facecolor="none", edgecolor="grey", linewidth=0.7)

    # Format integer coordinates with cardinal directions.
    def format_lon(x, pos):
        direction = 'E' if x >= 0 else 'W'
        return f"{abs(int(x))}°{direction}"

    def format_lat(y, pos):
        direction = 'N' if y >= 0 else 'S'
        return f"{abs(int(y))}°{direction}"

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    # Set the title.
    full_title = title if title else f"{fname}\nmin={vmin:.2f}, max={vmax:.2f}, mean={vmean:.2f}"
    ax.set_title(full_title)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()

    # Save the image.
    os.makedirs(output_folder, exist_ok=True)
    out_png= os.path.join(output_folder, fname + '.png')

    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"png saved: {out_png}")

def weekly_aver_mask_by_local_weekly_aver_data(
    input_folder: str,
    min_threshold: float = -20.0,
    max_threshold: float = None,
    min_weeks: int = 20,
    output_path: str = 'ssrd_mask.tif'
) -> np.ndarray:
    """
    Create a mask from a sequence of equally aligned rasters.

    A pixel is retained when at least ``min_weeks`` input rasters fall within
    the optional lower and upper thresholds. The legacy parameter name
    ``min_weeks`` represents the required number of input time slices, which
    may be weeks or years depending on the calling workflow.

    :param input_folder: Folder containing aligned input TIFFs.
    :param max_threshold: Optional upper threshold for each time slice.
    :param min_threshold: Optional lower threshold for each time slice.
    :param min_weeks: Required number of time slices meeting the thresholds.
    :param output_path: Path for the output mask GeoTIFF.
    """
    files = sorted([
        os.path.join(input_folder, fname)
        for fname in os.listdir(input_folder)
        if fname.lower().endswith('.tif')
    ])
    if not files:
        raise FileNotFoundError(f"No TIFF files found in {input_folder}")
    if min_weeks < 1:
        raise ValueError("min_weeks must be at least 1")
    if min_weeks > len(files):
        raise ValueError(
            f"min_weeks={min_weeks} exceeds the {len(files)} available TIFF files"
        )
    print(f"Found {len(files)} raster files.")

    # Use the first raster as the required grid for every time slice.
    with rasterio.open(files[0]) as src0:
        meta = src0.meta.copy()
        reference_shape = src0.shape
        reference_transform = src0.transform
        reference_crs = src0.crs

    exceed_count = np.zeros(reference_shape, dtype=np.uint16)

    for f in files:
        with rasterio.open(f) as src:
            if (
                src.shape != reference_shape
                or src.transform != reference_transform
                or src.crs != reference_crs
            ):
                raise ValueError(
                    f"Raster grid does not match {files[0]}: {f}. "
                    "Reproject every input to one shape, transform, and CRS first."
                )
            arr_masked = src.read(1, masked=True)
            arr = np.asarray(arr_masked.data)
            valid = ~np.ma.getmaskarray(arr_masked) & np.isfinite(arr)

        if min_threshold is not None and max_threshold is not None:
            meets_condition = np.logical_and(arr >= min_threshold, arr <= max_threshold)
        elif min_threshold is not None:
            meets_condition = arr >= min_threshold
        elif max_threshold is not None:
            meets_condition = arr <= max_threshold
        else:
            meets_condition = np.ones_like(arr, dtype=bool)

        exceed_count += np.logical_and(valid, meets_condition).astype(np.uint16)

    # Create mask: 1 where count >= min_weeks, else 0
    mask = (exceed_count >= min_weeks).astype(np.uint8)

    # Update metadata for single band
    meta.update({
        'count': 1,
        'dtype': 'uint8',
        'nodata': 0
    })

    # Write output mask
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(mask, 1)

    print(f"Mask written to {output_path}")

    return mask


def list_shp_fields(buffer_shp: str):
    """
    Print every Shapefile attribute field except the geometry field.
    """
    with fiona.open(buffer_shp, "r") as shp:
        props = shp.schema["properties"]
        field_names = list(props.keys())
    print("Shapefile fields:")
    for fn in field_names:
        print(f" - {fn}")
    return field_names


def split_global_mask_by_features(
    global_mask_tif: str,
    data_type:str,
    buffer_shp: str,
    output_folder: str,
    feat_id: str = "feat_id",
    buffer_dist:int= 10000,
):
    """
    Buffer and crop the global mask for each Shapefile feature.

    Vector geometries are transformed to the raster CRS before clipping. The
    source raster CRS and resolution are preserved in each output subset.

    :param global_mask_tif: Global mask GeoTIFF with 0, 1, or NoData pixels.
    :param data_type: Data type used in output names, such as temp or ssrd.
    :param buffer_shp: Shapefile containing features with unique identifiers.
    :param output_folder: Output TIFF folder, created when necessary.
    :param feat_id: Attribute used in output names; defaults to "feat_id".
    :param buffer_dist: Buffer distance in meters.
    """
    # Ensure the output directory exists.
    os.makedirs(output_folder, exist_ok=True)

    with rasterio.open(global_mask_tif) as src:
        src_meta = src.meta.copy()
        src_crs = src.crs
        src_transform = src.transform
        src_nodata = src.nodata
        print(f"Raster CRS: {src_crs}")
        print(f"Raster bounds: {src.bounds}")

        # Open the vector Shapefile.
        with fiona.open(buffer_shp, 'r') as shp:
            shp_crs = shp.crs
            print(f"Shapefile CRS: {shp_crs}")
            for feat in shp:
                props = feat['properties']
                if feat_id not in props:
                    raise KeyError(f"Field '{feat_id}' was not found in the Shapefile properties")
                fid = props[feat_id]
                if fid is None:
                    print(f"Warning: feature {feat['id']} field {feat_id} is None; skipping.")
                    continue

                # Reproject when the vector and raster CRSs differ.
                geom = feat['geometry']
                if shp_crs != src_crs:
                    geom_proj = transform_geom(shp_crs, src_crs, geom)
                else:
                    geom_proj = geom

                # Convert a meter buffer to degrees for a geographic CRS.
                geom_shape = shape(geom_proj)
                if src_crs.is_geographic:
                    # Approximate the meter-to-degree conversion at the equator.
                    deg_buffer = buffer_dist / 111319.5 # 1 degree ≈ 111319.5 meters at equator
                else:
                    deg_buffer = buffer_dist
                buffered_poly = geom_shape.buffer(deg_buffer)
                buffered_geom = mapping(buffered_poly)

                # Crop and fill with NoData.
                try:
                    out_arr, out_transform = mask(
                        src,
                        [buffered_geom],
                        crop=True,
                        nodata=src_nodata,
                        filled=True
                    )
                except ValueError:
                    print(f"Warning: feature {fid} does not intersect the raster; skipping.")
                    continue

                # Update dimensions while preserving CRS and resolution.
                out_meta = src_meta.copy()
                out_meta.update({
                    'height': out_arr.shape[1],
                    'width': out_arr.shape[2],
                    'transform': out_transform,
                    'nodata': src_nodata
                })

                # Build the output path.
                out_path = os.path.join(
                    output_folder,
                    f"feature_{fid}_{data_type}_mask.tif"
                )
                with rasterio.open(out_path, 'w', **out_meta) as dst:
                    dst.write(out_arr)
                print(f"Saved clipped mask for feat {fid}: {out_path}")

    print("All features processed.")


def initialize_ee(project_id=None):
    """Initialize Earth Engine for an explicitly configured cloud project."""
    if not project_id:
        raise ValueError(
            "Earth Engine project is required. Set URBANFRINGEPV_GEE_PROJECT_ID."
        )
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)

def export_tif_to_drive(image,title,boundary,crs ='EPSG:4326',scale =20):
    """
    Export an Earth Engine image to Google Drive.

    :param image: Earth Engine image to export.
    :param title: Export-task description and filename prefix.
    :param boundary: Earth Engine geometry defining the export region.
    :param crs: Output coordinate reference system.
    :param scale: Output pixel size in meters.

    When exporting in degrees, Earth Engine applies a fixed equatorial
    conversion rather than adjusting pixel size by latitude. For reference,
    30 meters is approximately 0.000269 degrees at the equator.
    """

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=title,
        folder= None,
        fileNamePrefix=title,
        region=boundary,
        maxPixels=1e13,
        crs=crs,
        scale=scale,
        fileFormat='GeoTIFF',
    )
    task.start()
    print(f"Started export task '{title}' with CRS={crs}, scale={scale}")

def load_boundary(path,simplify_tolerance):
    print("\nProcessing boundary...")

    # Load the FeatureCollection.
    fc = ee.FeatureCollection(path)

    # 2. Merge the collection into one geometry.
    geom = fc.geometry()

    # 3. Simplify the geometry in meters to remove excess vertices.
    simplified = geom.simplify(simplify_tolerance)

    return simplified


def landcover_mask(geom, fid, buffer_dist = 150,export=True,crs = 'EPSG:4326',scale =10):
    '''
    Build a mask for bare ground (class 6) and rangeland (class 9).

    The integer GeoTIFF uses zero for masked pixels because integer bands
    cannot store NaN values.
    :param geom:
    :param fid:
    :param export:
    :return: Binary mask with values 0 and 1.
    '''
    print("\nProcessing land cover...")

    def remap_lulc(image):
        # Remap ESRI classes: [1,2,4,5,7,8,9,10,11] -> [1..9]
        return image.remap([1, 2, 4, 5, 7, 8, 9, 10, 11], list(range(1, 10)))

    # Get the 2023 land cover within the buffered geometry.
    esri_lulc = ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS")

    buffered_geom = geom.buffer(buffer_dist)
    land_mosaic = esri_lulc.filterDate('2023-01-01', '2023-12-31').map(remap_lulc).mosaic().clip(buffered_geom)

    # Extract bare ground (class 6) and rangeland (9)
    bare_mask = land_mosaic.eq(6).Or(land_mosaic.eq(9))  # ee.Image (0/1)
    bare_ground = land_mosaic.updateMask(bare_mask)  # Values 0, 6, and 9.

    # Export bare ground and rangeland to Google Drive.
    if export:
        export_tif_to_drive(bare_ground, f'feature_{fid}_bare_rangeland', buffered_geom,crs =crs,scale = scale)

    return bare_mask


def slope_mask_gee(geom, fid, slope_degree_threshold, buffer_dist = 300,export = True,crs = 'EPSG:4326',scale =30):
    '''
    Build a slope mask and export slope values from zero to the threshold.

    Masked pixels in a floating-point TIFF can be represented as NaN.
    :param scale:
    :param crs:
    :param buffer_dist:
    :param export:
    :param geom:
    :param fid:
    :param slope_degree_threshold:
    :return: Binary mask with values 0 and 1.
    '''
    print("\nProcessing slope...")
    buffered_geom = geom.buffer(buffer_dist)  # Buffer to avoid edge effects during clipping.

    # Load the FABDEM DTM ImageCollection.
    fabdem = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")

    # Mosaic the collection at 30-meter resolution in EPSG:3857.
    elev = (
        fabdem.mosaic().clip(buffered_geom)
        .setDefaultProjection('EPSG:3857', None, 30)
    )
    slope = ee.Terrain.slope(elev)

    mask = slope.lte(slope_degree_threshold)

    # Export the slope raster to Google Drive.
    if export:
        export_tif_to_drive(slope, f'feature_{fid}_slope',
                            buffered_geom, crs=crs, scale=scale)

    return mask


def generate_masks_from_tifs(input_folder, output_folder, threshold=30,
                             use_nodata=False, nodata_value=255):
    """
    Extract pixels at or below the slope threshold from a batch of TIFFs.

    Original NoData pixels remain NoData when requested. Valid pixels at or
    below the threshold become 1; all others become 0 or ``nodata_value``.

    :param input_folder: Folder containing slope TIFF files.
    :param output_folder: Folder for output mask TIFF files.
    :param threshold: Slope threshold; defaults to 30.
    :param use_nodata: Use ``nodata_value`` for excluded pixels when True.
    :param nodata_value: NoData value used when ``use_nodata`` is True.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Match filenames of the form feature_<number>_<text>.tif.
    pattern = re.compile(r'^feature_(\d+)_.*\.tif$', re.IGNORECASE)

    # Collect matching files.
    tif_files = []
    for fn in os.listdir(input_folder):
        m = pattern.match(fn)
        if not m:
            continue
        feature_id = int(m.group(1))
        tif_files.append((feature_id, fn))

    # Sort by feature ID.
    tif_files.sort(key=lambda x: x[0])

    for feature_id, fn in tif_files:
        if not fn.lower().endswith(".tif"):
            continue
        base = os.path.splitext(fn)[0]  # e.g. "feature_997_slope"
        in_path = os.path.join(input_folder, fn)
        threshold_text = f"{threshold:g}"
        out_path = os.path.join(output_folder, f"{base}_let{threshold_text}_mask.tif")

        with rasterio.open(in_path) as src:
            slope = src.read(1)
            src_nodata = src.nodata
            profile = src.profile

            # Use one uint8 band while preserving georeferencing metadata.
            profile.update(
                dtype=rasterio.uint8,
                count=1,
                nodata=(nodata_value if use_nodata else None)
            )

            # Initialize the output mask.
            mask = np.zeros_like(slope, dtype=np.uint8)

            # Map source NoData pixels to the requested output value.
            if src_nodata is not None:
                mask[slope == src_nodata] = nodata_value if use_nodata else 0

            # Apply the threshold to remaining valid pixels.
            valid = (src_nodata is None) | (slope != src_nodata)
            if use_nodata:
                # Valid matching pixels become 1; others retain nodata_value.
                mask[np.logical_and(valid, slope <= threshold)] = 1
            else:
                # Valid matching pixels become 1; others remain 0.
                mask[np.logical_and(valid, slope <= threshold)] = 1
                # The mask already defaults to zero.

            # Write the mask.
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(mask, 1)

        print(f"Exported mask: {out_path}")

def _reproject_to_target(src, data, target_crs ='EPSG:8857', target_scale = 20):
    """
    Reproject and resample a single-band array to TARGET_CRS and TARGET_SCALE (meters).
    Returns the reprojected array, transform, crs, width, height.
    """
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src.crs, target_crs,
        src.width, src.height,
        *src.bounds,
        resolution=(target_scale, target_scale)
    )
    dst_meta = src.meta.copy()
    dst_meta.update({
        'crs': target_crs,
        'transform': dst_transform,
        'width': dst_width,
        'height': dst_height
    })
    dst_array = np.zeros((dst_height, dst_width), dtype=data.dtype)
    reproject(
        source=data,
        destination=dst_array,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest
    )
    return dst_array, dst_transform, dst_meta


def _binary_valid_mask(src):
    """Return 1 for finite, nonzero data and 0 for NoData or zero values."""
    masked = src.read(1, masked=True)
    data = np.asarray(masked.data)
    invalid = np.ma.getmaskarray(masked) | ~np.isfinite(data) | (data == 0)
    return np.where(invalid, 0, 1).astype(np.uint8)


def combine_single_feature_masks(fid, config):
    """
    Process one feature for the multiprocessing mask-combination workflow.

    :param fid: feature ID
    :param config: Dictionary containing all processing parameters.
    :return: tuple (fid, success, message)
    """
    try:
        # Unpack configuration parameters.
        land_dir = config['land_dir']
        slope_dir = config['slope_dir']
        ssrd_dir = config['ssrd_dir']
        temp_dir = config['temp_dir']
        wind_dir = config['wind_dir']
        output_dir = config['output_dir']
        target_crs = config['target_crs']
        target_scale = config['target_scale']
        filename_patterns = config['filename_patterns']
        buffer_geoms = config['buffer_geoms']
        buffer_proj_shp_path = config['buffer_proj_shp_path']

        target_crs_obj = CRS.from_user_input(target_crs)

        # Build input paths.
        paths = {
            k: os.path.join(
                {'land': land_dir, 'slope': slope_dir, 'ssrd': ssrd_dir, 'temp': temp_dir, 'wind': wind_dir}[k],
                v.format(fid=fid)
            )
            for k, v in filename_patterns.items()
        }

        # Check that every input exists.
        missing_files = {k: os.path.exists(p) for k, p in paths.items()}
        if not all(missing_files.values()):
            return fid, False, f"Missing files: {missing_files}"

        # Read and align all masks.
        arrays = []
        out_meta = None
        template_transform = None
        template_shape = None

        # 1. Use the land mask as the template.
        p0 = paths['land']
        with rasterio.open(p0) as src0:
            mask0 = _binary_valid_mask(src0)

            source_scale_matches = (
                np.isclose(abs(src0.transform.a), target_scale)
                and np.isclose(abs(src0.transform.e), target_scale)
            )
            if src0.crs != target_crs_obj or not source_scale_matches:
                # Calculate the template reprojection parameters.
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src0.crs, target_crs,
                    src0.width, src0.height,
                    *src0.bounds,
                    resolution=(target_scale, target_scale)
                )
                template_shape = (dst_height, dst_width)
                template_transform = dst_transform
                out_meta = src0.meta.copy()
                target_crs_obj = CRS.from_user_input(target_crs)
                wkt = target_crs_obj.to_wkt()
                out_meta.update({
                    'crs': target_crs_obj,
                    'crs_wkt': wkt,
                    'transform': dst_transform,
                    'width': dst_width,
                    'height': dst_height,
                    'count': 1,
                    'dtype': rasterio.uint8,
                    'nodata': 0
                })
                # Reproject the template.
                dst0 = np.zeros(template_shape, dtype=np.uint8)
                reproject(
                    source=mask0,
                    destination=dst0,
                    src_transform=src0.transform,
                    src_crs=src0.crs,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )
            else:
                # Copy directly when reprojection is unnecessary.
                dst0 = mask0
                template_shape = mask0.shape
                template_transform = src0.transform
                out_meta = src0.meta.copy()
                out_meta.update({
                    'count': 1,
                    'dtype': rasterio.uint8,
                    'nodata': 0
                })

            arrays.append(dst0)

        # 2. Align every other mask to the template grid.
        for key in ['slope', 'ssrd', 'temp', 'wind']:
            with rasterio.open(paths[key]) as src:
                mask_bin = _binary_valid_mask(src)

                grid_matches = (
                    src.crs == target_crs_obj
                    and src.shape == template_shape
                    and src.transform == template_transform
                )
                if not grid_matches:
                    dst_arr = np.zeros(template_shape, dtype=np.uint8)
                    reproject(
                        source=mask_bin,
                        destination=dst_arr,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=template_transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest
                    )
                else:
                    dst_arr = mask_bin

                arrays.append(dst_arr)

        # 3. Combine all masks.
        combined = np.all(np.stack(arrays, axis=0), axis=0).astype(np.uint8)

        # 4. Crop when a buffer Shapefile is provided.
        if buffer_proj_shp_path and fid not in buffer_geoms:
            return fid, False, "No matching geometry in the buffer Shapefile"

        if buffer_proj_shp_path and fid in buffer_geoms:
            geometry = [buffer_geoms[fid]]

            # Wrap the array in an in-memory raster before cropping.
            with MemoryFile() as memfile:
                with memfile.open(**out_meta) as tmp_ds:
                    tmp_ds.write(combined, 1)
                    clipped_array, clipped_transform = mask(tmp_ds, geometry, crop=True, nodata=0)

            out_meta.update({
                "height": clipped_array.shape[1],
                "width": clipped_array.shape[2],
                "transform": clipped_transform
            })
            combined = clipped_array[0]  # Remove the band dimension.

        # 5. Write the result.
        out_path = os.path.join(output_dir, f'feature_{fid}_combined_mask.tif')
        with rasterio.open(out_path, 'w', **out_meta) as dst:
            dst.write(combined, 1)

        # Write a companion .prj file.
        try:
            crs_obj = pyprojCRS.from_user_input(target_crs)
            prj_path = out_path.replace('.tif', '.prj')
            with open(prj_path, 'w') as prj_file:
                prj_file.write(crs_obj.to_wkt())
        except ImportError:
            print("pyproj is unavailable; no companion .prj file was written.")

        return fid, True, f"Generated {out_path}"

    except Exception as e:
        return fid, False, f"Processing failed: {str(e)}"

def _fetch_feature_ids(input_dir, start_fid=None, end_fid=None):
    print("Scanning input files...")
    feature_ids = []
    ssrd_files = glob.glob(os.path.join(input_dir, '*.tif'))
    for p in ssrd_files:
        basename = os.path.basename(p)
        match = re.match(r'feature_(\d+)_', basename)
        if match:
            feature_ids.append(int(match.group(1)))
        else:
            print(f"Could not extract a feature ID from filename: {basename}")

    # Apply optional start and end filters.
    if start_fid is not None:
        feature_ids = [fid for fid in feature_ids if fid >= start_fid]
    if end_fid is not None:
        feature_ids = [fid for fid in feature_ids if fid <= end_fid]

    feature_ids = sorted(set(feature_ids))
    print(f"Found {len(feature_ids)} feature IDs to process.")
    return feature_ids

def combine_feature_masks(
        land_dir: str,
        slope_dir: str,
        ssrd_dir: str,
        temp_dir: str,
        wind_dir: str,
        output_dir: str,
        n_processes: int = None,
        target_crs='EPSG:8857',
        target_scale=20,
        filename_patterns: dict = None,
        start_fid: int = None,
        end_fid: int = None,
        buffer_proj_shp_path: str = None,
):
    """
    Combine land, slope, solar, temperature, and wind masks by feature ID.

    Each input is aligned to the target CRS and resolution before a logical
    AND operation. When provided, the matching buffer polygon crops the
    combined raster before it is written to ``output_dir``.

    :param land_dir: Folder containing per-feature land masks.
    :param slope_dir: Folder containing per-feature slope masks.
    :param ssrd_dir: Folder containing per-feature solar masks.
    :param temp_dir: Folder containing per-feature temperature masks.
    :param wind_dir: Folder containing per-feature wind masks.
    :param output_dir: Folder for combined masks.
    :param n_processes: Worker count; defaults to the available CPU count.
    :param target_crs: Target CRS; defaults to EPSG:8857.
    :param target_scale: Target resolution; defaults to 20.
    :param filename_patterns: Dictionary of input filename patterns.
    :param start_fid: Process only feature IDs at or above this value.
    :param end_fid: Process only feature IDs at or below this value.
    :param buffer_proj_shp_path: Path to the buffer Shapefile.
    """
    print("Starting parallel feature-mask combination...")

    # Create the output directory.
    os.makedirs(output_dir, exist_ok=True)

    # Set the worker count.
    if n_processes is None:
        n_processes = max(1, min(cpu_count() - 1, 8))

    print(f"Using {n_processes} worker processes")

    # Define the default filename patterns.
    if filename_patterns is None:
        filename_patterns = {
            'land': 'feature_{fid}_bare_rangeland.tif',
            'slope': 'feature_{fid}_slope_mask.tif',
            'ssrd': 'feature_{fid}_ssrd_mask.tif',
            'temp': 'feature_{fid}_temp_mask.tif',
            'wind': 'feature_{fid}_wind_mask.tif'
        }

    # Map feature IDs to geometries from the buffer Shapefile.
    buffer_geoms = {}
    target_crs_obj = CRS.from_user_input(target_crs)
    if buffer_proj_shp_path and not os.path.exists(buffer_proj_shp_path):
        raise FileNotFoundError(f"Buffer vector not found: {buffer_proj_shp_path}")
    if buffer_proj_shp_path:
        print("Reading buffer Shapefile...")
        with fiona.open(buffer_proj_shp_path, "r") as shapefile:
            if not shapefile.crs:
                raise ValueError(f"Buffer vector has no CRS: {buffer_proj_shp_path}")
            buffer_crs = CRS.from_user_input(shapefile.crs)
            if buffer_crs != target_crs_obj:
                raise ValueError(
                    f"Buffer CRS {buffer_crs} does not match target CRS {target_crs_obj}."
                )
            for feature in shapefile:
                fid = int(feature['properties']['feat_id'])
                buffer_geoms[fid] = feature['geometry']
        print(f"Read {len(buffer_geoms)} buffer geometries")

    # 1. Extract feature IDs from files in the solar-mask directory.
    feature_ids = _fetch_feature_ids(ssrd_dir,start_fid,end_fid)
    if not feature_ids:
        raise FileNotFoundError(
            f"No per-feature solar mask TIFFs were found in {ssrd_dir} "
            f"for start_fid={start_fid}, end_fid={end_fid}."
        )

    # Prepare worker configuration.
    config = {
        'land_dir': land_dir,
        'slope_dir': slope_dir,
        'ssrd_dir': ssrd_dir,
        'temp_dir': temp_dir,
        'wind_dir': wind_dir,
        'output_dir': output_dir,
        'target_crs': target_crs,
        'target_scale': target_scale,
        'filename_patterns': filename_patterns,
        'buffer_geoms': buffer_geoms,
        'buffer_proj_shp_path': buffer_proj_shp_path,
    }

    # Start parallel processing.
    start_time = time.time()
    successful = 0
    failed = 0

    print("Running parallel processing...")

    with Pool(processes=n_processes) as pool:
        # Bind shared configuration for each worker.
        process_func = partial(combine_single_feature_masks, config=config)

        # Display progress with tqdm.
        with tqdm(total=len(feature_ids), desc="Progress", unit="features") as pbar:
            # Use imap for ordered results and live progress updates.
            for fid, success, message in pool.imap(process_func, feature_ids):
                if success:
                    successful += 1
                    pbar.set_postfix({"successful": successful, "failed": failed})
                else:
                    failed += 1
                    print(f"\nFeature {fid}: {message}")
                    pbar.set_postfix({"successful": successful, "failed": failed})

                pbar.update(1)

    # Report completion statistics.
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\nProcessing complete")
    print("Statistics:")
    print(f"   - Total: {len(feature_ids)}")
    print(f"   - Successful: {successful}")
    print(f"   - Failed: {failed}")
    print(f"   - Elapsed time: {elapsed_time:.2f} seconds")
    print(f"   - Average rate: {len(feature_ids) / elapsed_time:.2f} features/second")

# Get TIFF bounds in the target CRS.
def get_bounds_in_crs(src, target_crs):
    bounds = src.bounds
    transformer = Transformer.from_crs(src.crs, target_crs, always_xy=True)
    minx, miny = transformer.transform(bounds.left, bounds.bottom)
    maxx, maxy = transformer.transform(bounds.right, bounds.top)
    return box(minx, miny, maxx, maxy)

# Resample PV data to match a feature mask.
def reproject_match(source_raster, match_raster):
    dst_array = np.empty((match_raster.height, match_raster.width), dtype=np.uint8)
    reproject(
        source=source_raster.read(1),
        destination=dst_array,
        src_transform=source_raster.transform,
        src_crs=source_raster.crs,
        dst_transform=match_raster.transform,
        dst_crs=match_raster.crs,
        resampling=Resampling.nearest
    )
    return dst_array


# Process one feature mask.
def process_one_feature(feature_file, feature_dir, pv_dir, output_dir):
    feature_path = os.path.join(feature_dir, feature_file)
    output_path = os.path.join(output_dir, feature_file)

    with rasterio.open(feature_path) as feature_src:
        feature_data = feature_src.read(1)
        feature_bounds = get_bounds_in_crs(feature_src, feature_src.crs)

        # Check each PV raster for intersection with the current feature.
        for pv_file in os.listdir(pv_dir):
            if not pv_file.endswith(".tif"):
                continue

            pv_path = os.path.join(pv_dir, pv_file)
            with rasterio.open(pv_path) as pv_src:
                pv_bounds = get_bounds_in_crs(pv_src, feature_src.crs)

                if not feature_bounds.intersects(pv_bounds):
                    continue  # Skip non-overlapping rasters.

                try:
                    # Resample and align the PV data.
                    pv_data = reproject_match(pv_src, feature_src)
                    # Exclude areas covered by existing PV installations.
                    feature_data = np.where(pv_data == 1, 0, feature_data)
                except Exception as e:
                    print(f"Error processing {feature_file} + {pv_file}: {e}")
                    continue

        # Save the result.
        meta = feature_src.meta.copy()
        meta.update(dtype=rasterio.uint8)
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(feature_data, 1)

# Run the multiprocessing workflow.
def batch_process(feature_dir, pv_dir, output_dir,
                  num_workers=4,
                  start_fid = None,
                  end_fid = None):

    os.makedirs(output_dir, exist_ok=True)

    if start_fid is not None and end_fid is not None :
        feature_ids= _fetch_feature_ids(feature_dir,start_fid,end_fid)
    else:
        feature_ids = None

    all_features = []
    for f in os.listdir(feature_dir):
        if not f.endswith('.tif'):
            continue
        match = re.match(r"(.+?)_(\d+)_combined_mask\.tif", f)
        if match:
            fid = int(match.group(2))
            if feature_ids is None or fid in feature_ids:
                all_features.append(f)
    if not all_features:
        print("No matching features found.")
        return

    # Bind shared arguments while retaining a pickle-compatible worker.
    worker = partial(process_one_feature,
                     feature_dir=feature_dir,
                     pv_dir=pv_dir,
                     output_dir=output_dir)

    # pool.map accepts the pickle-compatible partial object.
    with Pool(num_workers) as pool:
        # Use map so tqdm receives a stable sequence.
        list(tqdm(pool.map(worker, all_features), total=len(all_features)))


def calculate_mask_areas(
        tif_folder: str,
        pattern: str,
        output_csv: str,
        target_crs: str = 'EPSG:8857',
        target_scale: int = 20
):
    """
    Calculate valid-pixel areas for TIFF files matching a filename pattern.

    Each raster is reprojected to an equal-area CRS before the summary CSV is
    written.

    :param tif_folder: Folder containing TIFF files.
    :param pattern: Filename pattern, such as feature_*_combined_mask.tif.
    :param output_csv: Path for the summary CSV.
    :param target_crs: Equal-area CRS used for area calculations. Examples
        include World Mollweide EPSG:6933, ArcGIS Mollweide ESRI:54009, and
        Equal Earth EPSG:8857
    :return: DataFrame containing the area calculations.
    """
    records = []
    target_crs_obj = CRS.from_user_input(target_crs)
    # Find all TIFF files matching the filename pattern.
    file_pattern = os.path.join(tif_folder, pattern)
    tif_files = glob.glob(file_pattern)

    if not tif_files:
        raise FileNotFoundError(f"No files matched '{file_pattern}'")

    total_files = len(tif_files)
    print(f"Found {total_files} files matching pattern '{pattern}'")

    for i, tif_path in enumerate(tif_files, 1):
        # Extract the feature ID.
        basename = os.path.basename(tif_path)

        # Try to extract the ID from the filename.
        match = re.search(r'feature_(\d+)', basename)
        fid = match.group(1) if match else os.path.splitext(basename)[0]

        print(f"Processing file [{i}/{total_files}]: {basename}")

        with rasterio.open(tif_path) as src:
            src_crs = src.crs
            if src_crs is None:
                raise ValueError(f"Input raster has no CRS: {tif_path}")
            src_transform = src.transform
            src_nodata = src.nodata
            data = src.read(1)
            original_size = f"{src.width}x{src.height}"

            # Count valid source pixels for comparison.
            if src_nodata is not None:
                original_valid = np.sum((data != src_nodata) & (data != 0) & (~np.isnan(data)))
            else:
                original_valid = np.sum((data != 0) & (~np.isnan(data)))

            # Reproject when required.
            if src_crs != target_crs_obj:
                dst_array, dst_transform, dst_meta = _reproject_to_target(src, data, target_crs, target_scale)
                width, height = dst_array.shape[1], dst_array.shape[0]
            else:
                dst_array = data
                dst_transform = src_transform
                width, height = src.width, src.height


            # 4. Identify NoData pixels.
            invalid_mask = np.zeros_like(dst_array, dtype=bool)

            # Account for each invalid-value representation.
            if src_nodata is not None:
                invalid_mask |= (dst_array == src_nodata)

            invalid_mask |= (dst_array == 0)
            invalid_mask |= np.isnan(dst_array)

            # Count valid pixels.
            valid_count = np.sum(~invalid_mask)

            # 5. Calculate pixel area in square meters.
            pixel_area = abs(dst_transform.a * dst_transform.e)
            total_area_sqm = valid_count * pixel_area

            # 6. Estimate pixel area in the source CRS for comparison.
            if src_crs.is_geographic:
                # Geographic CRSs require a latitude adjustment.
                center_lat = (src.bounds.bottom + src.bounds.top) / 2
                # Earth radius in meters.
                R = 6371000
                # Approximate the length of one degree at this latitude.
                meter_per_lat = np.pi * R / 180  # About 111,195 meters per degree.
                meter_per_lon = meter_per_lat * np.cos(np.radians(center_lat))

                src_pixel_area = abs(src_transform.a * src_transform.e) * meter_per_lon * meter_per_lat
            else:
                # Projected CRSs allow a direct area calculation.
                src_pixel_area = abs(src_transform.a * src_transform.e)

            src_total_area_sqm = original_valid * src_pixel_area

        # Record the result.
        records.append({
            "feature_id": fid,
            "filename": basename,
            "original_crs": str(src_crs),
            "reprojected_crs": str(target_crs_obj),
            "original_size": original_size,
            "reprojected_size": f"{width}x{height}",
            "original_valid_pixels": int(original_valid),
            "reprojected_valid_pixels": int(valid_count),
            "pixel_difference": int(valid_count) - int(original_valid),
            "pixel_difference_percent": ((int(valid_count) - int(original_valid)) / int(
                original_valid) * 100) if original_valid > 0 else 0,
            "original_pixel_area_sqm": src_pixel_area,
            "reprojected_pixel_area_sqm": pixel_area,
            "original_area_sqm": src_total_area_sqm,
            "reprojected_area_sqm": total_area_sqm,
            "area_difference_percent": ((total_area_sqm - src_total_area_sqm) / src_total_area_sqm * 100) if src_total_area_sqm > 0 else 0,
            "final_area_sqkm": total_area_sqm / 1e6
        })

    # Create the DataFrame and save the CSV.
    df = pd.DataFrame.from_records(records)

    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Generated area-summary CSV: {output_csv}")

    return df


def mosaic_resample(tif_dir, output_tif, dst_res=1000,compute_coverage=False):
    """
    Mosaic TIFFs and resample the global raster to ``dst_res``.

    Parameters:
    tif_dir : str
        Directory containing all raster tiles.
    output_tif : str
        Output file path.
    dst_res : float
        Target resolution in CRS units, for example 1000 meters.
    compute_coverage : bool, optional
        If True, write a float32 coverage-ratio raster with values from 0 to 1.
        If False, resample the global raster with ``Resampling.average``.
    """
    # Apply the GDAL option only when this operation is explicitly requested.
    gdal.SetConfigOption("GTIFF_SRS_SOURCE", "EPSG")

    # 1. List all TIFF files.
    tif_files = glob.glob(os.path.join(tif_dir, "*.tif"))
    if not tif_files:
        raise ValueError("No .tif files were found")

    # 2. Create a temporary VRT.
    tmp_vrt = tempfile.NamedTemporaryFile(suffix='.vrt', delete=False).name
    vrt_options = gdal.BuildVRTOptions(
        resampleAlg='nearest',
        addAlpha=True,
        allowProjectionDifference=True
    )
    gdal.BuildVRT(tmp_vrt, tif_files, options=vrt_options)


    # 3. Read the global bounds and common CRS from the VRT.
    with rasterio.open(tmp_vrt) as vrt0:
        bounds = vrt0.bounds
        src_crs = vrt0.crs
        src_transform = vrt0.transform
    min_left, min_bottom = bounds.left, bounds.bottom
    max_right, max_top = bounds.right, bounds.top

    # 4. Calculate output dimensions and transform.
    width = int(np.ceil((max_right - min_left) / dst_res))
    height = int(np.ceil((max_top - min_bottom) / dst_res))
    dst_transform = from_bounds(
        min_left, min_bottom, max_right, max_top,
        width, height
    )

    if not compute_coverage:
        # Resample by average through a dynamically loaded WarpedVRT.
        with rasterio.open(tmp_vrt) as src_vrt:
            with WarpedVRT(
                    src_vrt,
                    crs=src_crs,
                    transform=dst_transform,
                    width=width,
                    height=height,
                    resampling=Resampling.average
            ) as vrt:
                data = vrt.read(out_shape=(vrt.count, height, width))
                out_meta = vrt.meta.copy()

        # 6. Write the result.
        out_meta.update({
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'transform': dst_transform,
            'crs': src_crs,
            'compress': 'lzw'
        })
        with rasterio.open(output_tif, 'w', **out_meta) as dst:
            dst.write(data)

        print(f"Global VRT mosaic complete at {dst_res}m resolution: {output_tif}")
    else:
        # Calculate the valid-pixel coverage ratio.
        # 5.1 Read the VRT mask: 0 is invalid and 255 is valid.
        with rasterio.open(tmp_vrt) as src_vrt:
            src_mask_full = src_vrt.read_masks(1).astype(np.float32)  # 0 or 255

            # Allocate the destination array for the resampled mask mean.
            dst_mask = np.zeros((height, width), dtype=np.float32)

            # 5.2 Reproject the mask to the target resolution by averaging.
            rasterio.warp.reproject(
                source=src_mask_full,
                destination=dst_mask,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=src_crs,
                resampling=Resampling.average
            )

            # 5.3 Scale the mean from [0, 255] to [0, 1].
            # Fully invalid areas become 0; fully valid areas become 1.
            coverage = dst_mask / 255.0

            # 6b. Write the single-band float32 coverage-ratio raster.
            out_meta = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': 1,
                'dtype': 'float32',
                'crs': src_crs,
                'transform': dst_transform,
                'compress': 'lzw'
            }
            with rasterio.open(output_tif, 'w', **out_meta) as dst:
                dst.write(coverage.astype(np.float32), 1)

        print(f"Mosaic complete with coverage ratio at {dst_res}m: {output_tif}")

    # 7. Remove the temporary VRT.
    if os.path.exists(tmp_vrt):
        os.remove(tmp_vrt)
