import os
from datetime import datetime
import glob
import geopandas as gpd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from natsort import natsorted
import pandas as pd
import re
import rioxarray as rxr
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import xarray as xr

def export_netcdf_to_tif(nc_path,indicator,output_folder):
    """
    Export a NetCDF variable to one EPSG:4326 GeoTIFF per time step.

    Args:
        nc_path: Path to the input NetCDF file.
        indicator: Name of the variable to export.
        output_folder: Directory for the output GeoTIFF files.

    Returns:
        None.
    """

    prefix = os.path.splitext(os.path.basename(nc_path))[0]  # Use the input stem as the prefix.
    os.makedirs(output_folder, exist_ok=True)  # Reuse the directory if it already exists.

    ds = xr.open_dataset(nc_path)
    ds2 = ds.rename({"valid_time": "time"})  # Standardize the time coordinate name.
    da = ds2[indicator]  # Select the requested variable.
    # Ensure that the data has a CRS.
    da = da.rio.write_crs("EPSG:4326")

    # Export one time slice at a time.
    for t in da.time.values:
        # Select one time step.
        da_t = da.sel(time=t)
        # Build an output name such as "ssrd_2024_01_01T01.tif".
        ts = str(t)[:13].replace("-", "_")
        tif_name = f"{prefix}_{ts}.tif"

        # Join the filename to the output directory.
        out_path = os.path.join(output_folder, tif_name)

        # Export the GeoTIFF.
        da_t.rio.to_raster(
            out_path,
            driver="GTiff",
            dtype="float32"  # Preserve the existing output data type.
        )
        print(f"Exported {out_path}")


def export_netcdf_to_transformed_tif(nc_path, indicator, output_folder, resolution):
    """
    Export a NetCDF variable to aligned EPSG:4326 GeoTIFF time slices.

    Args:
        nc_path: Path to the input NetCDF file.
        indicator: Variable name, such as ``ssrd``.
        output_folder: Directory for the output GeoTIFF files.
        resolution: Output resolution, such as 0.1 or 0.25 degrees.

    Returns:
        None.
    """

    prefix = os.path.splitext(os.path.basename(nc_path))[0]  # Use the input stem as the prefix.
    os.makedirs(output_folder, exist_ok=True)

    ds = xr.open_dataset(nc_path)
    ds2 = ds.rename({"valid_time": "time"})  # Standardize the time coordinate name.
    da = ds2[indicator]  # Select the requested variable.

    # Read longitude and latitude coordinates.
    lons = da['longitude'].values
    lats = da['latitude'].values

    # Calculate coordinate bounds rounded to three decimal places.
    min_lon = np.round(lons.min(), 3)
    max_lon = np.round(lons.max(), 3)
    min_lat = np.round(lats.min(), 3)
    max_lat = np.round(lats.max(), 3)

    # Normalize longitude coordinates to -180 to 180 and sort them.
    da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180))
    da = da.sortby("longitude")

    # Set the spatial dimensions and CRS.
    da = da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    da = da.rio.write_crs("EPSG:4326")

    # Export one time slice at a time.
    for t in da.time.values:
        da_t = da.sel(time=t)

        # Build the timestamp used in the output filename.
        ts = str(t)[:13].replace("-", "_")
        tif_name = f"{prefix}_{ts}.tif"
        out_path = os.path.join(output_folder, tif_name)

        # Align the transform to the upper-left corner at (-180, 90).
        transform = from_origin(west=-180, north=90, xsize=resolution, ysize=resolution)
        da_t = da_t.rio.write_transform(transform)

        # Clip each time slice to consistent global bounds.
        da_t = da_t.rio.clip_box(minx=-180, miny=-90, maxx=180, maxy=90)

        # Export the GeoTIFF.
        da_t.rio.to_raster(out_path, dtype="float32")
        print(f"Exported {out_path}")


def batch_plot_tif_compute_min_max_mean(tif_path, label, base_shp_path=None):
    """
    Plot each GeoTIFF and record its minimum, maximum, and mean value.

    Args:
        tif_path: Directory containing the input GeoTIFF files.
        label: Color-bar label, such as J/m², kWh/m², or temperature in K.
        base_shp_path: Optional path to a vector basemap, such as a shapefile.

    Returns:
        A list of per-file minimum, maximum, and mean records.
    """
    # Build the output directory.
    parent, foldername = os.path.split(os.path.abspath(tif_path))
    output_folder = os.path.join(parent, foldername + "_png")
    os.makedirs(output_folder, exist_ok=True)

    # Read the optional vector basemap.
    if base_shp_path:
        base_gdf = gpd.read_file(base_shp_path)
    else:
        base_gdf = None
    records = []  # Store the minimum, maximum, and mean for each GeoTIFF.

    # Iterate over all GeoTIFF files.
    for fname in natsorted(os.listdir(tif_path)):
        if not fname.lower().endswith('.tif'):
            continue

        in_path = os.path.join(tif_path, fname)
        da = rxr.open_rasterio(in_path)

        vmin =da.min().item()
        vmax = da.max().item()
        vmean = da.mean().item()  # Calculate the mean.

        records.append({'filename': fname, 'min_value': vmin, 'max_value': vmax,'mean_value':vmean})

        # Plot the raster.
        fig, ax = plt.subplots(figsize=(10, 6))

        # Draw the raster data.
        da.plot( ax = ax, cmap="viridis",  robust=True,  # Ignore extreme values automatically.
            cbar_kwargs={"label": label}
        )

        # Overlay the vector basemap on the same axes.
        if base_gdf is not None:
            # Reproject the basemap when its CRS differs from the raster CRS.
            if base_gdf.crs != da.rio.crs:
                base_gdf = base_gdf.to_crs(da.rio.crs)
                print("Reprojected basemap to:", base_gdf.crs)
            else:
                print("Basemap CRS already matches the raster CRS; no reprojection needed.")

            base_gdf.plot(ax=ax, facecolor="none", edgecolor="grey", linewidth=1 )

        ax.set_title(f"{fname}\nmin={vmin:.2f}, max={vmax:.2f}, mean={vmean:.2f}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()

        # Save the figure.
        out_png= os.path.join(output_folder, os.path.splitext(fname)[0] + '.png')

        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"png saved: {out_png}")

    # Export the summary CSV.
    df = pd.DataFrame(records)
    csv_path = os.path.join(parent, foldername + "_minmax.csv")
    df.to_csv(csv_path, index=False,encoding='utf-8-sig')
    print(f"Processed {len(records)} files.")
    print(f"PNGs in: {output_folder}")
    print(f"Summary CSV: {csv_path}")

    return records


def tifs_to_csv(tif_folder, output_folder):
    """
    Convert all GeoTIFF pixel values in a directory to a CSV table.

    Each row represents one raster cell, and each GeoTIFF is stored in a
    filename-based column.

    Args:
        tif_folder: Directory containing the GeoTIFF files.
        output_folder: Directory for the output CSV file.

    Returns:
        The output DataFrame and the template raster metadata.
    """
    os.makedirs(output_folder, exist_ok=True)

    # List all GeoTIFF files.
    tif_paths = sorted(glob.glob(os.path.join(tif_folder, "*.tif")))
    if not tif_paths:
        raise ValueError(f"No tifs found in folder: {tif_folder}")

    # Read the first GeoTIFF to determine the shape and metadata.
    with rasterio.open(tif_paths[0]) as src0:
        rows, cols = src0.height, src0.width
        meta = src0.meta.copy()

    # Store one raster cell per row and one GeoTIFF per value column.
    data = {}
    for path in tif_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        with rasterio.open(path) as src:
            arr = src.read(1)
        data[name] = arr.flatten()
        print(f'Finished {name}')

    df = pd.DataFrame(data)
    # Add raster row and column indices.
    rows_idx, cols_idx = np.indices((rows, cols))
    df['row'] = rows_idx.flatten()
    df['col'] = cols_idx.flatten()

    # Save the CSV file.
    output_csv= os.path.join(output_folder,"data.csv")
    df.to_csv(output_csv, index=False)
    print(f"CSV saved to: {output_csv}")
    return df, meta


def csv_to_tifs(input_csv, template_meta, out_folder):
    """
    Restore one GeoTIFF per value column from a CSV file.

    Args:
        input_csv: CSV file containing raster-cell values.
        template_meta: Rasterio metadata containing transform, CRS, width, and height.
        out_folder: Directory for the output GeoTIFF files.
    """
    os.makedirs(out_folder, exist_ok=True)
    df = pd.read_csv(input_csv)

    # Read the raster dimensions.
    rows = template_meta['height']
    cols = template_meta['width']

    # Exclude the raster index columns from the value columns.
    value_cols = [c for c in df.columns if c not in ('row', 'col')]

    for col in value_cols:
        arr_flat = df[col].values
        arr = arr_flat.reshape(rows, cols)

        meta = template_meta.copy()
        meta.update({
            'count': 1,
            'dtype': rasterio.float32
        })
        out_tif = os.path.join(out_folder, f"{col}.tif")
        with rasterio.open(out_tif, 'w', **meta) as dst:
            dst.write(arr.astype(rasterio.float32), 1)
        print(f"Wrote TIFF: {out_tif}")

def accumulate_average_data(tif_folder, output_folder, frequency ='week',compute_method = 'avg',unit = 'C'):
    """
    Group GeoTIFF files by year, month, or ISO week and aggregate each group.

    Expected filenames include ``ssrd_hourly_month12_hour24_2024_01_01T19.tif``
    and ``temp_monthly_by_hourly24_2024_01_01T19.tif``.

    Args:
        tif_folder: Directory containing the hourly GeoTIFF files.
        output_folder: Directory for the aggregated GeoTIFF files.
        frequency: Aggregation frequency: ``year``, ``month``, or ``week``.
        compute_method: Aggregation method: ``sum`` or ``avg``.
        unit: Output unit: ``C`` or ``kWh/m²``.
    """
    os.makedirs(output_folder, exist_ok=True)

    # List all GeoTIFF files and group them by the requested frequency.
    tif_paths = glob.glob(os.path.join(tif_folder, "*.tif"))
    groups = defaultdict(list)  # Create groups with empty lists by default.

    for path in tif_paths:
        fn = os.path.basename(path)
        # The filename includes a prefix such as "temp_monthly_by_hourly24_2024_01_01T19.tif".
        parts = fn.split("_")
        # Example parts: ["temp", "monthly", "by", "hourly24", "2024", "01", "01T19.tif"].
        if frequency == 'year':
            key = f"{parts[4]}"

        elif frequency == 'month':
            key = f"{parts[4]}_{parts[5]}"

        elif frequency =='week':
            # Extract the YYYY_MM_DD date from the full filename.
            m = re.search(r"(\d{4})_(\d{2})_(\d{2})", fn)
            if not m:
                continue
            date_str = "".join(m.groups())  # YYYYMMDD
            dt = datetime.strptime(date_str, "%Y%m%d")
            iso_year, iso_week, _ = dt.isocalendar()  # Returns year, week number, and weekday.
            key = f"{iso_year}_W{iso_week:02d}"  # Example: "2024_W01".

        else:
            raise ValueError("frequency must be 'month' or 'week'")

        groups[key].append(path)


    # Sum or average each frequency group.
    for key, paths in sorted(groups.items()):
        arr_sum = None
        count = 0
        meta = None

        for p in sorted(paths):
            with rasterio.open(p) as src:
                data = src.read(1).astype(np.float64)  # Use float64 to prevent overflow.
                if arr_sum is None:
                    arr_sum = np.zeros_like(data)
                    meta = src.meta.copy()
                arr_sum += data
                count += 1
        # Apply the selected aggregation method.
        if compute_method== 'avg':
            result = arr_sum / count
        elif compute_method == 'sum':
            result = arr_sum
        else:
            raise ValueError("compute_method must be 'sum' or 'avg'")

        # Convert units.
        if unit == 'C':
            result = result - 273.15  # K to C.
        elif unit == 'kWh/m²':
            result = result /3.6e6  # J/m² to kWh/m²; 1 kWh/m² = 3.6 MJ/m².
        else:
            raise ValueError("unit must be 'C' or 'kWh/m²'")

        # Write the result.
        meta.update({
            "dtype": "float32",
            "count": 1
        })
        out_tif = os.path.join(output_folder, f"{parts[0]}_{frequency}_{key}_aver.tif")
        with rasterio.open(out_tif, "w", **meta) as dst:
            dst.write(result.astype(np.float32), 1)

        print(f"Wrote {frequency} average for {key}: {out_tif}")


def _merge_era5_land_global(land_path, global_path, output_path):
    """
    Merge ERA5-Land and ERA5-Global, preferring land data, and write a GeoTIFF.
    """
    # Read both rasters.
    land = rxr.open_rasterio(land_path).squeeze()
    glob = rxr.open_rasterio(global_path).squeeze()

    # Assign the geographic CRS.
    land = land.rio.write_crs("EPSG:4326", inplace=True)
    glob = glob.rio.write_crs("EPSG:4326", inplace=True)

    # Handle nodata values.
    land_nodata = land.rio.nodata
    if land_nodata is not None:
        land = land.where(land != land_nodata)
    else:
        land = land.where(~np.isnan(land))

    # Resample the global data to match the land raster.
    glob_resampled = glob.rio.reproject_match(land, resampling=Resampling.bilinear)

    # Use land values where available and global values elsewhere.
    combined = land.where(~land.isnull(), other=glob_resampled)

    # Write the merged raster.
    combined.rio.to_raster(output_path)
    print(f"Created: {output_path}")

def batch_merge_all(land_dir, global_dir, output_dir, suffix=".tif"):
    """
    Match same-named ERA5-Land and ERA5-Global rasters and merge each pair.

    Args:
        land_dir: Directory containing ERA5-Land files.
        global_dir: Directory containing ERA5-Global files.
        output_dir: Directory for merged output files.
        suffix: Filename suffix to process; defaults to ``.tif``.
    """
    os.makedirs(output_dir, exist_ok=True)

    file_list = sorted([f for f in os.listdir(land_dir) if f.endswith(suffix)])

    for i, filename in enumerate(file_list, 1):
        land_path = os.path.join(land_dir, filename)
        global_path = os.path.join(global_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if not os.path.exists(global_path):
            print(f"Missing from the global directory: {filename}; skipping")
            continue

        print(f"[{i}/{len(file_list)}] Merging: {filename}")
        _merge_era5_land_global(land_path, global_path, output_path)


def extract_and_plot(df_points, raster_folder, output_folder, num_weeks=None):
    """
    Extract weekly raster values at points and plot every point in one figure.

    Args:
        df_points: DataFrame containing point ``id``, ``lat``, and ``lon`` columns.
        raster_folder: Directory containing weekly GeoTIFF files.
        output_folder: Directory for the summary CSV and combined plot.
        num_weeks: Optional limit on the number of raster files to process.
    """
    # Read point coordinates.
    points = [(lon, lat) for lon, lat in zip(df_points['lon'], df_points['lat'])]
    result_df = df_points.copy()

    # List and sort all GeoTIFF files.
    tif_files = sorted([f for f in os.listdir(raster_folder) if f.lower().endswith('.tif')])
    if num_weeks is not None:
        tif_files = tif_files[:num_weeks]

    # Extract raster values.
    for tif in tif_files:
        col_name = os.path.splitext(tif)[0]  # Use the filename stem as the column name.
        raster_path = os.path.join(raster_folder, tif)

        with rasterio.open(raster_path) as src:
            vals = [v[0] if v.size > 0 else None for v in src.sample(points)]
            result_df[col_name] = vals

    # Export the CSV file.
    os.makedirs(output_folder, exist_ok=True)

    csv_path = os.path.join(output_folder, "points_summary.csv")
    result_df.to_csv(csv_path, index=False)

    # Plot all point series.
    week_cols = [os.path.splitext(f)[0] for f in tif_files]
    plt.figure(figsize=(10, 6))
    for _, row in result_df.iterrows():
        temps = row[week_cols].values.astype(float)
        plt.plot(range(1, len(week_cols) + 1), temps, label=str(row['name']))

    plt.title("Weekly Temperature for All Points")
    plt.xlabel("Week Index")
    plt.ylabel("Temperature")
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.tight_layout()

    combined_plot_path = os.path.join(output_folder, "all_points.png")
    plt.savefig(combined_plot_path)
    plt.close()

    print(f"Exported CSV: {csv_path}")
    print(f"Saved combined plot: {combined_plot_path}")
