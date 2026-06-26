import pandas as pd
import glob

enugu = pd.read_csv('data_kenya/metadata/embu/embu_clusters_gsvmatched.csv')
edo = pd.read_csv('data_kenya/metadata/kitui/kitui_clusters_gsvmatched.csv')
kano = pd.read_csv('data_kenya/metadata/kwale/kwale_clusters_gsvmatched.csv')
oyo = pd.read_csv('data_kenya/metadata/machakos/machakos_clusters_gsvmatched.csv')
edo = pd.read_csv('data_kenya/metadata/makueni/makueni_clusters_gsvmatched.csv')
abuja = pd.read_csv('data_kenya/metadata/meru/meru_clusters_gsvmatched.csv')
lagos = pd.read_csv('data_kenya/metadata/mombasa/mombasa_clusters_gsvmatched.csv')
nairobi = pd.read_csv('data_kenya/metadata/nairobi/nairobi_clusters_gsvmatched.csv')


combined = pd.concat([enugu, edo, kano, oyo, edo, abuja, lagos, nairobi])
print(f"Total clusters: {len(combined)}")
print(combined["wealth_num"].describe())
combined.to_csv("clusters_cities_pt1.csv", index=False)