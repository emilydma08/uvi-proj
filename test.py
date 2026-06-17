import pandas as pd

df = pd.read_csv('data/processed/clusters_all_cities.csv')
print(df['image_count'].sum())