# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 14:45:00 2018

# Project:  Local Multivariate Geary’s c (Anselin, 2017)
# Purpose:  Mean-median comparison on identifyed spatial multivariate clusters/outliers for classifying them
# Author:   Daniele Oxoli (daniele.oxoli@polimi.it)
# Affiliation: Department of Civil and Environmental Engineering | GEOlab, Politecnico di Milano, P.zza Leonardo da Vinci 32, 20133, Milano, Italy

"""

import geopandas as gpd
import pysal as ps
import numpy as np


# read the output of the "multivar_c.py" module

out_path = ".../result.shp"

df = gpd.read_file(out_path)

w = ps.weights.Queen.from_dataframe(df)

neigh = w.neighbors

df_sig_clust = df.loc[df['sig_loc'] == 1]
df_sig_out = df.loc[df['sig_loc'] == -1]
#df_sig_not = df.loc[df['sig_loc'] == 0].sample(frac=0.1)


neigh_sig_clust = { key: neigh[key] for key in list(df_sig_clust.index) }
neigh_sig_out = { key: neigh[key] for key in list(df_sig_out.index) }
#neigh_sig_not = { key: neigh[key] for key in list(df_sig_not.index) }

attribute_to_keep = ['k1','k2','k3','k1_stand','k2_stand','k3_stand','C_ki','p_sim','p_sim_fdr','sig_loc','geometry']
att_for_pca = ['k1','k2','k3']
att_for_class = ['k1_stand','k2_stand','k3_stand']
df_reduced = df[attribute_to_keep]


# Mm spatial clusters

clust_ids = []
clust_comp = {}
ref_mean = df_reduced[att_for_class].median().mean()
clust_class = {}
dist_class = {}

for keys in neigh_sig_clust.keys():  
    clust_ids.append(keys)
    clust_ids.extend(neigh_sig_clust[keys])
    
    #temporary dataframe with attribute values for location i and its neighgbors
    df_clust_class = df_reduced[att_for_class].iloc[clust_ids]
    clust_ids = []
    
    #label for cluster classification
    clust_mean =  df_clust_class.median().mean()
    if clust_mean > ref_mean:
        lab = 'hh'
    else:
        lab = 'll'
    clust_class.update({keys: lab})
    dist_class.update({keys: round(abs(ref_mean-clust_mean),3)})

# Mm spatial outliers

out_ids = []
out_var ={}
out_comp = {}
out_class = {}

for keys in neigh_sig_out.keys():  
    out_ids.append(keys)
    out_ids.extend(neigh_sig_out[keys])
    df_out_class = df_reduced[att_for_class].iloc[out_ids]
    out_ids = []
    
    #label for cluster classification
    id_list =  df_out_class.index.values
    neigh_id_list = np.delete(id_list, 0)  
    neigh_mean =  df_out_class.loc[neigh_id_list].median().mean()
    loc_mean = df_out_class.loc[keys].mean()
    if loc_mean > neigh_mean:
        lab = 'hl'
    else:
        lab = 'lh'
    out_class.update({keys: lab})
    dist_class.update({keys: round(abs(loc_mean-neigh_mean),3)})

 
'''add proposed classification column'''

#clusters

df_sig_clust['class'] = np.nan
df_sig_clust['diff'] = np.nan
for keys in clust_class:
    df_sig_clust['class'].loc[keys] = clust_class[keys]
    df_sig_clust['diff'].loc[keys] = dist_class[keys]

#outliers
df_sig_out['class'] = np.nan
df_sig_out['diff'] = np.nan
for keys in out_class:
    df_sig_out['class'].loc[keys] = out_class[keys]
    df_sig_out['diff'].loc[keys] = dist_class[keys]


# write results in a shapefile
        
Mm_path = '.../Mm_class.shp'
df_sig_pca = df_sig_clust.append(df_sig_out)
df_sig_pca.to_file(driver = 'ESRI Shapefile', filename= Mm_path)
    

