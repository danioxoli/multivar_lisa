# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 09:59:51 2018

@author: daniele
"""

import geopandas as gpd
import pysal as ps
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA

# custom method
def multiple_testing_correction(pvalues, correction_type="FDR"):
    from numpy import array, empty
    pvalues = array(pvalues)
    sample_size = pvalues.shape[0]
    qvalues = empty(sample_size)
    if correction_type == "Bonferroni":
        # Bonferroni correction
        qvalues = sample_size * pvalues
    elif correction_type == "Bonferroni-Holm":
        # Bonferroni-Holm correction
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        for rank, vals in enumerate(values):
            pvalue, i = vals
            qvalues[i] = (sample_size-rank) * pvalue
    elif correction_type == "FDR":
        # Benjamini-Hochberg, AKA - FDR test
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        values.reverse()
        new_values = []
        for i, vals in enumerate(values):
            rank = sample_size - i
            pvalue, index = vals
            new_values.append((sample_size/rank) * pvalue)
        for i in range(0, int(sample_size)-1):
            if new_values[i] < new_values[i+1]:
                new_values[i+1] = new_values[i]
        for i, vals in enumerate(values):
            pvalue, index = vals
            qvalues[index] = new_values[i]
    return qvalues
    
# formal lib for FDR
import statsmodels.stats.multitest as mt



# read the output of the "multivar_c.py" module
in_path = "/Users/daniele/Google Drive/Delivery&Application/melbourne/uni/multiLISA_UADI/result.shp_9999_fdr/result.shp"
out_path = "/Users/daniele/Desktop/not_sig_PCA_graphs/Di_full.shp"

df = gpd.read_file(in_path)

w = ps.weights.Queen.from_dataframe(df)
neigh = w.neighbors
neigh_dic = { key: neigh[key] for key in list(df.index) }

attribute_to_keep = ["SA2_main",'k1','k2','k3','C_ki','p_sim','sig_loc','geometry']
att_for_pca = ['k1','k2','k3']

df_reduced = df[attribute_to_keep]

#Number of Principal Components
N_comps = 2

#PCA on all locations with graphic output

df_sig_pca = df_reduced

d_sum = 0
Di = []
m1 = []
m2 = []

for ids in df.index:
    for value in neigh_dic[ids]:
        if value in df_sig_pca.index:          
            continue
        else:
            df_sig_pca.loc[value] = df_reduced.loc[value]
            

X_s = df_sig_pca.loc[:,'k1':'k3'].values
y_s = df_sig_pca.index.values
X_s_std = StandardScaler().fit_transform(X_s)
sklearn_pca_s = sklearnPCA(n_components=N_comps)
Y_sklearn_s = sklearn_pca_s.fit_transform(X_s_std)


for keys in neigh_dic:
    loi = np.array(neigh_dic[keys])
    
    #append location i PCA coordinates          
    m1.append(Y_sklearn_s[y_s == keys, 0][0]) #x1
    m2.append(Y_sklearn_s[y_s == keys, 1][0]) #y1  

    #append neighbours PCA coordinates                    
    for l in loi:       
        x2 = Y_sklearn_s[y_s == l, 0][0]
        y2 = Y_sklearn_s[y_s == l, 1][0]
        m1.append(x2)
        m2.append(y2)

    #compute the mass centre coordinates
    cg1 = np.sum(m1)/len(m1)
    cg2 = np.sum(m2)/len(m2) 
    
    #compute the Di 
    for i in range(0,len(m1)):
        dist = np.sqrt( (m1[i] - cg1)**2 + (m2[i] - cg2)**2 )
        d_sum += dist
                        
    m1 = []
    m2 = []
    
    Di.append(round(d_sum/len(loi),3))
    d_sum = 0
        

# Di CSR significance using random permutations

df_sig_pca = df_reduced

d_sum = 0
Di_obs = Di
Di = []
permutations = 9999
alpha = 0.03
fdr_sim = True
Di_sim = np.zeros((len(df),permutations))
m1 = []
m2 = []


for rp in range(0,permutations):

    shuff_x = np.random.permutation([row[0] for row in Y_sklearn_s])
    shuff_y = np.random.permutation([row[1] for row in Y_sklearn_s])

    for keys in neigh_dic:
        loi = np.array(neigh_dic[keys])
        
        #append location i PCA coordinates
        m1.append(Y_sklearn_s[y_s == keys, 0][0])
        m2.append(Y_sklearn_s[y_s == keys, 1][0])   
        
        #append neighbours PCA coordinates under CRS
        for l in loi:       
            x2 = shuff_x[l]
            y2 = shuff_y[l]
            m1.append(x2)
            m2.append(y2)

        #compute the mass centre coordinates
        cg1 = np.sum(m1)/len(m1)
        cg2 = np.sum(m2)/len(m2) 

        #compute the Di under CRS
        for i in range(0,len(m1)):
            dist = np.sqrt( (m1[i] - cg1)**2 + (m2[i] - cg2)**2 )
            d_sum += dist
                            
        m1 = []
        m2 = []
        
        Di.append(round(d_sum/len(loi),3))
        d_sum = 0
        
    Di_sim[:,rp] = np.asarray(Di)
    Di = []


# simulated p-values 
p_sim = np.zeros((np.shape(Di_sim)[0]))

for i in range(0,np.shape(Di_sim)[0]):
    above = Di_sim[i] > Di_obs[i] 
    larger = above.sum(0)
    if (permutations - larger) < larger:
        larger = permutations - larger
    p_sim[i] = ((larger + 1.0) / (permutations + 1.0))
    
# correction for simulated p-values (FDR) - custom
if fdr_sim == True:
    p_sim_fdr = multiple_testing_correction(p_sim, correction_type="FDR")
else:
    p_sim_fdr = p_sim

# correction for simulated p-values (FDR) - formal
p_sim_fdr2 = mt.fdrcorrection(p_sim, alpha=alpha)    
    
# label clusters and outlier according to the Di mean - custom
sig = p_sim_fdr <= alpha
corr_lower =  Di_obs >= np.mean(Di_obs)
corr_higher = Di_obs < np.mean(Di_obs)
locations = np.zeros(len(df))
locations[sig*corr_higher] = 1
locations[sig*corr_lower] = -1

# label clusters and outlier according to the Di mean - formal
sig = p_sim_fdr2[0]
corr_lower =  Di_obs >= np.mean(Di_obs)
corr_higher = Di_obs < np.mean(Di_obs)
locations = np.zeros(len(df))
locations[sig*corr_higher] = 1
locations[sig*corr_lower] = -1

# save the file and the Di attributes
df_reduced["Di"] = Di_obs
df_reduced["p_sim_Di"] = p_sim_fdr
df_reduced['sig_loc_Di'] = locations
df_reduced.to_file(driver = 'ESRI Shapefile', filename= out_path)


# plot FDR effect on p-values distribution
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

df["fdr2"] = p_sim_fdr2[1]

x = df["t_p_uni"]
y = df["t_p_sim"]
z = df["t_p_fdr"]
#bins = np.linspace(-10, 10, 30)

plt.hist([x, y, z],bins=20, color=['black', 'red', 'green'], label=['p-values (no multiple comparisions)','p-values (multiple comparisions)', 'FDR adjusted p-values'],
         alpha=0.5, normed=True, histtype='bar', rwidth=0.8)
plt.ylabel("Frequency")
plt.xlabel("p-values")
plt.legend(loc='upper right')
plt.show()
