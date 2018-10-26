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
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA

# read the output of the "multivar_c.py" module
out_path = "your_path_to_file"
g_path = "your_path_to_graph_folder"


df = gpd.read_file(out_path)

w = ps.weights.Queen.from_dataframe(df)

neigh = w.neighbors

df_sig = df.loc[df['sig_loc'] != 0]
neigh_sig = { key: neigh[key] for key in list(df_sig.index) }

attribute_to_keep = ['k1','k2','k3','C_ki','p_sim','sig_loc','geometry']
att_for_pca = ['k1','k2','k3']

df_reduced = df[attribute_to_keep]

#Number of Principal Components
N_comps = 2


#PCA on all significant locations with graphic output

df_sig_pca = df_reduced.loc[df['sig_loc'] != 0]
reg_x = []
reg_y = []
d_sum = 0
d_list_clust={}
d_list_out={}
m1 = []
m2 = []

for ids in df_sig.index:
    for value in neigh_sig[ids]:
        if value in df_sig_pca.index:          
            continue
        else:
            df_sig_pca.loc[value] = df_reduced.loc[value]
            

X_s = df_sig_pca.loc[:,'k1':'k3'].values
y_s = df_sig_pca.index.values
X_s_std = StandardScaler().fit_transform(X_s)
sklearn_pca_s = sklearnPCA(n_components=N_comps)
Y_sklearn_s = sklearn_pca_s.fit_transform(X_s_std)


for keys in neigh_sig:
    loi = np.array(neigh_sig[keys])
       
    if df_reduced.loc[keys]['sig_loc'] == 1:
        color = 'r'
    else:
        color = 'b' 
        
    with plt.style.context('seaborn-whitegrid'):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(Y_sklearn_s[:, 0], Y_sklearn_s[:, 1],
                    alpha=0.5, s=50, c='g', marker="P")
                    
        m1.append(Y_sklearn_s[y_s == keys, 0][0])
        m2.append(Y_sklearn_s[y_s == keys, 1][0])       
                    
        for l in loi:
            lab =   str(df.loc[l]['sa2_name'])          
            ax1.scatter(Y_sklearn_s[y_s == l, 0], Y_sklearn_s[y_s == l, 1],
                        label = lab, s=100, c='k', marker="v")
            
            reg_x.append(Y_sklearn_s[y_s == l, 0])
            reg_y.append(Y_sklearn_s[y_s == l, 1])
            
            x1 = Y_sklearn_s[y_s == keys, 0][0]
            y1 = Y_sklearn_s[y_s == keys, 1][0]
            x2 = Y_sklearn_s[y_s == l, 0][0]
            y2 = Y_sklearn_s[y_s == l, 1][0]
            dist = np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
            d_sum += dist
                        
            m1.append(x2)
            m2.append(y2)
                          
        ax1.scatter(Y_sklearn_s[y_s == keys, 0], Y_sklearn_s[y_s == keys, 1],
                label= str(df.loc[keys]['sa2_name']), s=200, c=color, marker="*")
        
        reg_x.append(Y_sklearn_s[y_s == keys, 0])
        reg_y.append(Y_sklearn_s[y_s == keys, 1])

        coeff = linregress(np.concatenate(reg_x, axis=0), np.concatenate(reg_y, axis=0))        
        reg_x = []
        reg_y = []

        cg1 = np.sum(m1)/len(m1)
        cg2 = np.sum(m2)/len(m2)    
        m1 = []
        m2 = []
        
        ax1.scatter(cg1, cg2,
                label= 'mass centre', s=180, c='indigo', marker="P")  
        
#        ax1.scatter(0, 0,
#                label= 'origin (0,0)', s=80, c='r', marker="o") 
#        
#        at1 = AnchoredText(u"\u03D0"+'='+str(round(coeff[0],3))+
#              ', d='+str(round(d_sum/len(loi),3))+', x_c='
#              +str(round(cg1,3))+', y_c='+str(round(cg2,3)), 
#                           prop=dict(size=10), frameon=True, loc=1)

        ax1.scatter(0, 0,
                label= 'origin (0,0)', s=80, c='r', marker="o") 
        
        at1 = AnchoredText('dispersion = '+str(round(d_sum/len(loi),3))
#        +', p-value (FDR) = '+str(df.loc[keys]['p_sim_fdr'])
        , 
                           prop=dict(size=10), frameon=True, loc=1)
        at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax1.add_artist(at1)               
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')  
        plt.tight_layout()      
        plt.show()
        d_sum = 0
        fig.savefig(g_path+title, dpi=300, interpolation='bilinear')

