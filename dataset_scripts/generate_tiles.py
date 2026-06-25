import pandas as pd

def generate_tiles_from_clusters(df, tile_size=0.05, buffer=1):
    """
    Generate tiles centered on each cluster coordinate.
    
    tile_size: degrees per tile (~5.5km at Nigerian latitudes)
    buffer: number of tiles to extend in each direction
    """
    tiles = set()
    
    for _, row in df.iterrows():
        lat, lon = row["LATNUM"], row["LONGNUM"]
        
        if lat == 0.0 and lon == 0.0:
            continue
        
        # snap to tile grid
        for dlat in range(-buffer, buffer + 1):
            for dlon in range(-buffer, buffer + 1):
                min_lon = round(lon - (lon % tile_size) + dlon * tile_size, 6)
                min_lat = round(lat - (lat % tile_size) + dlat * tile_size, 6)
                max_lon = round(min_lon + tile_size, 6)
                max_lat = round(min_lat + tile_size, 6)
                tiles.add((min_lon, min_lat, max_lon, max_lat))
    
    return list(tiles)


# usage
clusters = pd.read_csv("dhs_kwale_clusters.csv")
tiles = generate_tiles_from_clusters(clusters, tile_size=0.05, buffer=1)
print(f"Generated {len(tiles)} tiles for {len(clusters)} clusters")
print(tiles)