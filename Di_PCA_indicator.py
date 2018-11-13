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
out_path = "your_path_to__input_file"
g_path = "your_path_to_graph_folder"
out2_path = "your_path_to__output_file"

df = gpd.read_file(out_path)

w = ps.weights.Queen.from_dataframe(df)

neigh = w.neighbors

df_sig = df
neigh_sig = { key: neigh[key] for key in list(df_sig.index) }

attribute_to_keep = ['k1','k2','k3','C_ki','p_sim','sig_loc','geometry']
att_for_pca = ['k1','k2','k3']

df_reduced = df[attribute_to_keep]

#Number of Principal Components
N_comps = 2


#PCA on all locations with graphic output

df_sig_pca = df_reduced
reg_x = []
reg_y = []
d_sum = 0
Di = []
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
        facecolor='r'
        edgecolor='r'
        title = 'C_'+str(df.loc[keys]['sa2_name'])+'.png'
    else:
        facecolor='b'
        edgecolor='b'
        title = 'O_'+str(df.loc[keys]['sa2_name'])+'.png'
    
    if df_reduced.loc[keys]['sig_loc'] == 0:
        facecolor='white'
        edgecolor='k'
        title = 'N_'+str(df.loc[keys]['sa2_name'])+'.png'
        
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

                        
            m1.append(x2)
            m2.append(y2)
        
        cg1 = np.sum(m1)/len(m1)
        cg2 = np.sum(m2)/len(m2)
        
        #compute the Di    
        for i in range(0,len(m1)):
            dist = np.sqrt( (m1[i] - cg1)**2 + (m2[i] - cg2)**2 )
            d_sum += dist
                          
        ax1.scatter(Y_sklearn_s[y_s == keys, 0], Y_sklearn_s[y_s == keys, 1],
                label= str(df.loc[keys]['sa2_name']), s=200, marker="*", facecolors=facecolor, edgecolors=edgecolor )
        
        m1 = []
        m2 = []
        
        ax1.scatter(cg1, cg2,
                label= 'mass centre', s=180, c='indigo', marker="P")  
        

        ax1.scatter(0, 0,
                label= 'origin (0,0)', s=80, c='r', marker="o") 
        
        at1 = AnchoredText('Di = '+str(round(d_sum/len(loi),3))
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
        Di.append(round(d_sum/len(loi),3))
        d_sum = 0
        fig.savefig(g_path+title, dpi=300, interpolation='bilinear')
        
df_sig_pca["SA2"] = df["SA2_main"]
df_sig_pca["Di"] = Di
df_sig_pca.to_file(driver = 'ESRI Shapefile', filename= out2_path)

