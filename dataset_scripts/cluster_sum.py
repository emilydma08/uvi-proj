import pandas as pd

df = pd.read_csv('data/processed/clusters_cities_pt3.csv')
print(df['image_count'].sum())