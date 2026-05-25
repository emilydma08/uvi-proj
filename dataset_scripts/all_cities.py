import pandas as pd
import glob

enugu = pd.read_csv('enugu/enugu_clusters_gsvmatched.csv')
edo = pd.read_csv('edo/edo_clusters_gsvmatched.csv')
kano = pd.read_csv('kano/kano_clusters_gsvmatched_final.csv')
oyo = pd.read_csv('oyo/oyo_clusters_gsvmatched.csv')
edo = pd.read_csv('edo/edo_clusters_gsvmatched.csv')
abuja = pd.read_csv('data/metadata/abuja/abuja_clusters_gsvmatched.csv')
lagos = pd.read_csv('data/metadata/lagos/lagos_clusters_gsvmatched.csv')


combined = pd.concat([enugu, edo, kano, oyo, edo, abuja, lagos])
print(f"Total clusters: {len(combined)}")
print(combined["wealth_num"].describe())
combined.to_csv("clusters_all_cities.csv", index=False)