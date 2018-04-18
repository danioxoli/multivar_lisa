# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 14:45:00 2018

# Project:  Local Multivariate Geary’s c (Anselin, 2017)
# Purpose:  Principal Componet Analysis on identifyed spatial multivariate clusters/outliers
# Author:   Daniele Oxoli (daniele.oxoli@polimi.it)
# Affiliation: Department of Civil and Environmental Engineering | GEOlab, Politecnico di Milano, P.zza Leonardo da Vinci 32, 20133, Milano, Italy

"""

import geopandas as gpd
import pysal as ps

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA

# read the output of the "multivar_c.py" module
out_path = "/Users/daniele/Google Drive/Delivery&Application/melbourne/uni/multiLISA_UADI/result.shp"

df = gpd.read_file(out_path)

w = ps.weights.Queen.from_dataframe(df)

neigh = w.neighbors

df_sig_clust = df.loc[df['sig_loc'] == 1]
df_sig_out = df.loc[df['sig_loc'] == -1]
df_sig_not = df.loc[df['sig_loc'] == 0].sample(frac=0.1)


neigh_sig_clust = { key: neigh[key] for key in list(df_sig_clust.index) }
neigh_sig_out = { key: neigh[key] for key in list(df_sig_out.index) }
neigh_sig_not = { key: neigh[key] for key in list(df_sig_not.index) }

attribute_to_keep = ['k1','k2','k3','C_ki','p_sim','sig_loc','geometry']
att_for_pca = ['k1','k2','k3']

df_reduced = df[attribute_to_keep]

#PCA on spatial cluster

clust_ids = []
clust_var ={}

for keys in neigh_sig_clust.keys():  
    clust_ids.append(keys)
    clust_ids.extend(neigh_sig_clust[keys])
    df_clust = df_reduced[att_for_pca].iloc[clust_ids]
    clust_ids = []

    X = df_clust.iloc[:,0:len(list(df_clust))].values
    X_std = StandardScaler().fit_transform(X)
    
    
#    cov_mat = np.cov(X_std.T)
#    eig_vals, eig_vecs = np.linalg.eig(cov_mat)   
#    
#    tot = sum(eig_vals)
#    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
#    cum_var_exp = np.cumsum(var_exp)
    
    sklearn_pca = sklearnPCA(n_components=X_std.shape[1])
    Y_sklearn = sklearn_pca.fit_transform(X_std)
    var_exp = sklearn_pca.explained_variance_ratio_
    
    clust_var.update({keys: var_exp})

#PCA on spatial outliers

out_ids = []
out_var ={}

for keys in neigh_sig_out.keys():  
    out_ids.append(keys)
    out_ids.extend(neigh_sig_out[keys])
    df_out = df_reduced[att_for_pca].iloc[out_ids]
    out_ids = []

    X = df_out.iloc[:,0:len(list(df_out))].values
    X_std = StandardScaler().fit_transform(X)    
    
    sklearn_pca = sklearnPCA(n_components=X_std.shape[1])
    Y_sklearn = sklearn_pca.fit_transform(X_std)
    var_exp = sklearn_pca.explained_variance_ratio_
    
    out_var.update({keys: var_exp})

 
#PCA on sample not interesting locations   

not_ids = []
not_var ={}

for keys in neigh_sig_not.keys():  
    not_ids.append(keys)
    not_ids.extend(neigh_sig_not[keys])
    df_not = df_reduced[att_for_pca].iloc[not_ids]
    not_ids = []

    X = df_not.iloc[:,0:len(list(df_not))].values
    X_std = StandardScaler().fit_transform(X)    
    
    sklearn_pca = sklearnPCA(n_components=X_std.shape[1])
    Y_sklearn = sklearn_pca.fit_transform(X_std)
    var_exp = sklearn_pca.explained_variance_ratio_
    
    not_var.update({keys: var_exp})
