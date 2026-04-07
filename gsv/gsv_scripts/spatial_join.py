import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

meta = pd.read_csv('data/metadata/lagos_metadata.csv')
gdf_panos = gpd.GeoDataFrame(
    meta,
    geometry=[Point(lon, lat) for lon, lat in zip(meta.lon, meta.lat)],
    crs='EPSG:4326'
)

rwi_raw = pd.read_csv('data/raw/labels/nga_relative_wealth_index.csv')
# Columns are typically: latitude, longitude, rwi, error
gdf_rwi = gpd.GeoDataFrame(
    rwi_raw,
    geometry=[Point(lon, lat) for lon, lat in zip(rwi_raw.longitude, rwi_raw.latitude)],
    crs='EPSG:4326'
)

# --- Nearest neighbor join ---
# Project to a metric CRS first so distances are meaningful
gdf_panos_proj = gdf_panos.to_crs('EPSG:32632')   # UTM zone 32N covers Lagos
gdf_rwi_proj   = gdf_rwi.to_crs('EPSG:32632')

joined = gpd.sjoin_nearest(
    gdf_panos_proj,
    gdf_rwi_proj[['rwi', 'geometry']],
    how='left',
    distance_col='rwi_dist_m'   # lets you QC how far the nearest cell was
)

# Drop any duplicate pano_ids (shouldn't happen with nearest, but just in case)
joined = joined.drop_duplicates(subset='pano_id', keep='first')

# Bring rwi back to original meta
meta['rwi'] = joined['rwi'].values
meta['rwi_dist_m'] = joined['rwi_dist_m'].values  # keep for QC
meta.to_csv('data/metadata/lagos_metadata_labeled.csv', index=False)