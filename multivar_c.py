# -*- coding: utf-8 -*-
"""
Created on Wed Nov 08 15:44:19 2017
#
# Project:  Local Multivariate Geary’s c (Anselin, 2017)
# Purpose:  Application of the Local Geary c statistic to a multivariate context using vector layers. Example with 3 variables. 
# Author:   Daniele Oxoli (daniele.oxoli@polimi.it)
# Affiliation: Department of Civil and Environmental Engineering | GEOlab, Politecnico di Milano, P.zza Leonardo da Vinci 32, 20133, Milano, Italy
#
"""

'''
#REQUIRED PACKAGES
'''
# data management libraries

import geopandas as gpd

# stats and computing libraries

import pysal as ps
import numpy as np
import scipy.stats as st

# plotting libraries

import matplotlib.pyplot as plt
import seaborn as sns
from pylab import figure, show

'''
# ADDITION FUNCTIONS
'''
# simulated p-values correction function (Copyright 2017 Francisco Pina Martins <f.pinamartins@gmail.com>)

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

# if you need to check run time
import time

'''
# INPUT DATA 
'''

in_path = ".../input.shp"
out_path = ".../result.shp"

df = gpd.read_file(in_path)

# rename the column of the selected attributes with a short name (here used: 3 attributes)
df.rename(columns={"attribute 1": "k1", "attribute 2": "k2", "attribute 3": "k3"},inplace=True)

'''
# INPUT PARAMETER
'''

weigth_type = 'r' # 'o' = original binary, 'r' = row-stand.

permutations = 999 # number of random permutations

significance = 0.01 # significance level for hypothesis testing

significance_fdr = 0.03 # significance level for hypothesis testing with FDR

fdr_sim = True # FDR correction flag for simulated p-values (set False to not apply the correction)
fdr_norm = False # FDR correction flag for p-values based on standard normal approximation (set True to apply the correction)
fdr_sim_norm =  False # FDR correction flag for p-values based on standard normal approximation from permutations (set True to apply the correction)

'''
# SPATIAL WEIGHTS AND ATTRIBUTE MATRICES EXTRACTION
'''

w = ps.weights.Queen.from_dataframe(df)
w.transform= weigth_type
wf = w.full()[0]

# inverting attributes to give them an equal meaning verse if needed (eg High = good, Low = bad)
#df['k1']=1/df['k1']
#df['k2']=1/df['k2']
#df['k3']=1/df['k3']

att_arrs = [df['k1'],df['k2'],df['k3']]
att_mtx = np.array(att_arrs).transpose()

att_arrs_norm = [(df['k1']-df['k1'].mean())/df['k1'].std(),
                 (df['k2']-df['k2'].mean())/df['k2'].std(), 
                 (df['k3']-df['k3'].mean())/df['k3'].std()]
att_mtx_norm = np.array(att_arrs_norm).transpose()

'''
# REAL STATISTIC COMPUTATION
'''

d_square= np.zeros((np.shape(att_arrs_norm)[1],np.shape(att_arrs_norm)[1]))

for i in range(0,np.shape(att_arrs_norm)[1]):
    for j in range(0,np.shape(att_arrs_norm)[1]):
        ks_i = att_mtx_norm[i] 
        ks_j = att_mtx_norm[j]
        d_i_j = ((ks_i - ks_j)**2).sum()
        d_square[i][j] = d_i_j
        
C_vi = wf * d_square

C_ki = np.sum(C_vi,axis=0) # real statistic values vector

'''
#INFERENCE UNDER NORMALITY ASSUMPTION
'''

C_ki_z_norm = (C_ki - np.mean(C_ki))/np.std(C_ki) # standard variates from standard normal approximation

p_norm = st.norm.sf(abs(C_ki_z_norm))*2 # p-values based on standard normal approximation

'''
# INFERENCE UNDER RANDOMIZATION ASSUMPTION (CONDITIONAL PERMUTATIONS)
'''

# simulated statistics 
np.random.seed(12345)

C_ki_perm_list = []

for k in range(0,permutations):
    d_square_perm= np.zeros((np.shape(att_arrs_norm)[1],np.shape(att_arrs_norm)[1]))
# permutation of the single attribute array    
    perm_att_norm = []
    for i in range(0,np.shape(att_arrs_norm)[0]):
        perm_att_norm_i=np.random.permutation(att_arrs_norm[i])
        perm_att_norm.append(perm_att_norm_i)
    perm_mtx_norm = np.array(perm_att_norm).transpose()       
#    perm_mtx_norm=np.random.permutation(att_mtx_norm) # this substitutes lines 158->162 for permutation of the whole matrix 
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

E_C_ki_perm= [np.mean(C_ki_perm[i]) for i in range(0,len(C_ki_perm))] # mean from permutations

S_C_ki_perm = [np.std(C_ki_perm[i]) for i in range(0,len(C_ki_perm))] # standard deviation from permutations

C_ki_z_sim = (C_ki - E_C_ki_perm)/S_C_ki_perm # standard variates from permutations

p_z_sim = st.norm.sf(abs(C_ki_z_sim))*2 # p-values based on standard normal approximation from permutations (two-sided)

# simulated p-values based on permutations (one-sided), null: spatial randomness

p_sim = np.zeros((np.shape(att_arrs_norm)[1]))

for i in range(0,np.shape(att_arrs_norm)[1]):
    above = C_ki_perm[i] > C_ki[i] 
    larger = above.sum(0)
    if (permutations - larger) < larger:
        larger = permutations - larger
    p_sim[i] = ((larger + 1.0) / (permutations + 1.0))

# correction for simulated p-values (FDR)

if fdr_sim == True:
    p_sim_fdr = multiple_testing_correction(p_sim, correction_type="FDR")
else:
    p_sim_fdr = p_sim

# correction for p-values based on standard normal approximation (FDR)

if fdr_norm == True:
    p_sim_fdr = multiple_testing_correction(p_norm, correction_type="FDR")
else:
    p_norm_fdr = p_norm

# correction for p-values based on standard normal approximation from permutations (FDR)

if fdr_sim_norm == True:
    p_sim_norm_fdr = multiple_testing_correction(p_z_sim, correction_type="FDR")
else:
    p_sim_norm_fdr = p_z_sim
    
'''
# ADD TO THE ATTRIBUTE TABLE THE COMPUTED STATISTICS
''' 
df['k1_stand']= (df['k1']-df['k1'].mean())/df['k1'].std()
df['k2_stand']= (df['k2']-df['k2'].mean())/df['k2'].std()
df['k3_stand']=  (df['k3']-df['k3'].mean())/df['k3'].std()
df['C_ki'] = C_ki    
df['p_sim_fdr'] = p_sim_fdr
df['p_sim'] = p_sim
df['z_sim'] = C_ki_z_sim
#df['p_norm'] = p_norm_fdr
#df['z_norm'] = C_ki_z_norm
#df['z_sim_norm'] = C_ki_z_sim
#df['p_sim_norm'] = p_sim_norm_fdr

# define locations of interest in the dataset and add flags 

sig = p_sim_fdr <= significance_fdr

corr_lower =  C_ki >= np.mean(C_ki)

corr_higher = C_ki <np.mean(C_ki)

locations = np.zeros((np.shape(att_arrs_norm)[1]))

locations[sig*corr_higher] = 1
locations[sig*corr_lower] = -1

# add flags column to the dataframe (0 = not significant, -1 = negative association, 1 = positive association)

df['sig_loc'] = locations

''' 
# SAVE THE MODIFIED GEODATAFRAME TO A NEW SHAPEFILE 
'''

df.to_file(driver = 'ESRI Shapefile', filename= out_path)

'''
# INTERPRETATION AND PLOT OF INTERESTING LOCATIONS
'''
#simplest plot
#df.plot(column='k1', cmap='OrRd', scheme='quantiles', k=4, edgecolor='black', figsize=(7,7), legend= True)

# attributes plot

fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(21,7))
df.plot(column='k1', cmap='OrRd', scheme='quantiles', k=4, edgecolor='black', legend= True , ax=axes[0])
df.plot(column='k2', cmap='OrRd', scheme='quantiles', k=4, edgecolor='black', legend= True, ax=axes[1])
df.plot(column='k3', cmap='OrRd', scheme='quantiles', k=4, edgecolor='black', legend= True, ax=axes[2])
fig.suptitle('INDICATOR QUANTILES MAPS', fontsize=16)
axes[0].set_title("attribute 1", fontstyle='italic')
axes[1].set_title("attribute 2", fontstyle='italic')
axes[2].set_title("attribute 3", fontstyle='italic')

# plot reference distribution from permutations for the i_th location  

i = 51 #location ID

fig, ax = plt.subplots(1, figsize=(10,7))
sns.distplot(C_ki, color='g', label='obs. dist. with mean +- std')
#plt.vlines(C_ki, 0, 0.005, 'r')
plt.xlim([0.0, 30])
plt.vlines(np.mean(C_ki), 0, 0.6, 'g')
#plt.vlines(np.mean(C_ki)-np.std(C_ki), 0, 10, 'g','dotted')
#plt.vlines(np.mean(C_ki)+np.std(C_ki), 0, 10, 'g','dotted')

#sns.kdeplot(C_ki_perm[i], shade=True, color="c", label='perm. dist. with mean +- std')
sns.distplot(C_ki_perm[i], color="c", label='perm. dist. with mean +- std')
plt.vlines(C_ki_perm[i], 0, 0.005, 'k', label='perm')
#plt.vlines(np.mean(C_ki_perm[i]), 0, 0.8, 'c')
#plt.vlines(np.mean(C_ki_perm[i])-np.std(C_ki_perm[i]), 0, 10, 'c', 'dashed')
#plt.vlines(np.mean(C_ki_perm[i])+np.std(C_ki_perm[i]), 0, 10, 'c', 'dashed')
plt.xlim([0.0, 30])

plt.vlines(C_ki[i], 0, 0.8, 'r', label='obs statistic')


# multivariate geary's c
fig, ax = plt.subplots(1, figsize=(8,10))
ax = df.plot(column='sig_loc', cmap='bwr', edgecolor='black', legend= True, categorical=True, axes=ax)
fig.suptitle("MULTIVARIATE SPATIAL ASSOCIATION - GEARY'S C",fontsize=16)
ax.set_title("1 - 2 - 3" , fontstyle='italic')
plt.show()

# plot interesting location count and names
print ('not significant locations = ' +str(sum(df['sig_loc'] == 0)))
print ('significant locations of high values = ' + str(sum(df['sig_loc'] == 1)))
print ('significant locations of low values= ' + str(sum(df['sig_loc'] == -1)))

list_neg_autocorr = []
list_pos_autocorr = []
for i in range(0,len(df)):
    if df['sig_loc'][i] == -1:
        list_neg_autocorr.append(df['sa2_name'][i])
    if df['sig_loc'][i] == 1:
        list_pos_autocorr.append(df['sa2_name'][i])
        
# plot pseudo p-values and computed z-score maps
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,7))
df.plot(column='z_norm', cmap='OrRd_r', scheme='quantiles', k=5, edgecolor='black', legend= True,  ax=axes[0])
df.plot(column='p_sim', cmap='Greys_r', scheme='quantiles', k=5, edgecolor='black', legend= True,  ax=axes[1])
fig.suptitle("Quntiles  maps - Computed statistics", fontsize=16)
axes[0].set_title("Computed MULTIVARIATE GEARY'S C z-norm ", fontstyle='italic')
axes[1].set_title("Pseudo p-values", fontstyle='italic')
plt.show()

''' PLOT NEIGHBORS '''

a = w.histogram
# sort in-place from highest to lowest
a.sort(key=lambda x: x[1], reverse=True) 

# save the beans and their respective elements separately
# reverse the tuples to go from most frequent to least frequent 
n_bean = zip(*a)[0]
score = zip(*a)[1]
x_pos = np.arange(len(n_bean)) 

plt.bar(n_bean, score, align='center', color = 'r')
plt.xticks(n_bean) 
plt.suptitle("Neighbors Distribution", fontsize=16)
plt.xlabel('n. of neighbours')
plt.ylabel('n. of precincts')
plt.show()

# plot connectivity grapth
centroids = np.array([list([poly.x, poly.y]) for poly in df.geometry.centroid])

fig = plt.figure(figsize=(9,9))
plt.plot(centroids[:,0], centroids[:,1],'.')
plt.title('Centroids')
show()

fig = figure(figsize=(9,9))

plt.plot(centroids[:,0], centroids[:,1],'.')
for k,neighs in w.neighbors.items():
    origin = centroids[k]
    for neigh in neighs:
        segment = centroids[[k,neigh]]
        plt.plot(segment[:,0], segment[:,1], '-')
plt.title('Queen Neighbor Graph')
show()

runtime = (time.clock() - start_time)/60
print runtime, "minutes"


