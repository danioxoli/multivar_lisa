# -*- coding: utf-8 -*-
"""
Created on Wed Nov 08 15:44:19 2017
#
# Project:  Local Multivariate Geary’s c (Anselin, 2017)
# Purpose:  Application of the Local Geary c statistic to a multivariate context using vector layers 
# Author:   Daniele Oxoli (daniele.oxoli@polimi.it)
# Affiliation: Department of Civil and Environmental Engineering | GEOlab, Politecnico di Milano, P.zza Leonardo da Vinci 32, 20133, Milano, Italy
#
"""

'''
#REQUIRED PACKAGES
'''

import pysal as ps
import geopandas as gpd
import numpy as np
import scipy.stats as st


'''
#INPUT DATA 
'''
path = "C:/Users/utente/Desktop/multivar_SA/simple_test_c/grid5_rnd.shp"

df = gpd.read_file(path)

df.plot(column='k1', cmap='OrRd', scheme='quantiles', edgecolor='black')
df.plot(column='k2', cmap='OrRd', scheme='quantiles', edgecolor='black')
df.plot(column='k3', cmap='OrRd', scheme='quantiles', edgecolor='black')


'''
#INPUT PARAMETER
'''

weigth_type = 'r' # 'o' = original binary, 'r' = row-stand.

permutations = 19 # number of random permutations


'''
#SPATIAL WEIGHTS AND ATTRIBUTE MATRICES EXTRACTION
'''


w = ps.weights.Queen.from_dataframe(df)
w.transform= 'r'
wf = w.full()[0]
		
att_arrs = [df['k1'],df['k2'],df['k3']]
att_mtx = np.array(att_arrs).transpose()

att_arrs_norm = [(df['k1']-df['k1'].mean())/df['k1'].std(),
                 (df['k2']-df['k2'].mean())/df['k2'].std(), 
                 (df['k3']-df['k3'].mean())/df['k3'].std()]
att_mtx_norm = np.array(att_arrs_norm).transpose()

norm_sum = np.zeros(np.shape(att_mtx_norm)[0])
for j in range (0, len(norm_sum)):
    norm_sum[j] = sum(att_mtx_norm[j])
    
df['norm_sum'] = norm_sum

df.plot(column='norm_sum', cmap='OrRd', scheme='quantiles', edgecolor='black')

'''
#REAL STATISTIC COMPUTATION
'''

d_square= np.zeros((np.shape(att_arrs_norm)[1],np.shape(att_arrs_norm)[1]))

for i in range(0,np.shape(att_arrs_norm)[1]):
    for j in range(0,np.shape(att_arrs_norm)[1]):
        ks_i = att_mtx_norm[i] 
        ks_j = att_mtx_norm[j]
        d_i_j = ((ks_i - ks_j)**2).sum()
        d_square[i][j] = d_i_j
        
C_vi = wf * d_square

C_ki = np.sum(C_vi,axis=0)


'''
#INFERENCE UNDER NORMALITY ASSUMPTION
'''

C_ki_z_norm = (C_ki - np.mean(C_ki))/np.std(C_ki)

p_norm = st.norm.sf(abs(C_ki_z_norm))*2 #twosided


'''
# INFERENCE UNDER RANDOMIZATION ASSUMPTION (CONDITIONAL PERMUTATION)
'''

#sumulated statistics values

np.random.seed(12345)

C_ki_perm_list = []

for k in range(0,permutations):
    d_square_perm= np.zeros((np.shape(att_arrs_norm)[1],np.shape(att_arrs_norm)[1]))
    perm_mtx_norm=np.random.permutation(att_mtx_norm)
    for i in range(0,np.shape(att_arrs_norm)[1]):
        for j in range(0,np.shape(att_arrs_norm)[1]):
            if i == j:
                ks_i = att_mtx_norm[i] 
                ks_j = att_mtx_norm[j]
                d_i_j = ((ks_i - ks_j)**2).sum()
                d_square_perm[i][j] = d_i_j
            else:                
                ks_i = att_mtx_norm[i] 
                ks_j = perm_mtx_norm[j]
                d_i_j = ((ks_i - ks_j)**2).sum()
                d_square_perm[i][j] = d_i_j
    
    C_ki_perm_list.append(np.sum(wf * d_square_perm,axis=0))
    
C_ki_perm = np.array(C_ki_perm_list).transpose()


E_C_ki_perm= [np.mean(C_ki_perm[i]) for i in range(0,len(C_ki_perm))]

S_C_ki_perm = [np.std(C_ki_perm[i]) for i in range(0,len(C_ki_perm))]

C_ki_z_sim = (C_ki - E_C_ki_perm)/S_C_ki_perm

# p-values based on standard normal approximation from permutations

p_z_sim = st.norm.sf(abs(C_ki_z_sim))*2 #twosided

# simulated p values based on permutations (one-sided), null: spatial randomness

p_sim = np.zeros((np.shape(att_arrs_norm)[1]))

for i in range(0,np.shape(att_arrs_norm)[1]):
    above = C_ki_perm[i] >= C_ki[i] 
    larger = above.sum(0)
    p_sim[i] = ((larger + 1.0) / (permutations + 1.0))

 
'''
# INTERPRETATION AND PLOT OF INTERESTING LOCATIONS
'''

# plot reference distribution from permutation for the i_th location  

import matplotlib.pyplot as plt
import seaborn as sns

i = 23

sns.kdeplot(C_ki_perm[i], shade=True)
plt.vlines(C_ki_perm[i], 0, 0.005)
plt.vlines(C_ki[i], 0, 10, 'r')


sns.kdeplot(p_norm, shade=True)
plt.vlines(p_norm, 0, 0.05)
plt.xlim([0, 1])


# plot locations of interest in the dataset 

sig = p_norm >= 0.9

locations = np.zeros((np.shape(att_arrs_norm)[1]))

locations[sig] = 1

df['sig'] = locations

df.plot(column='sig', cmap='OrRd', edgecolor='black')




'''--------'''

y = df['k2']

lg = ps.G_Local(y,w)