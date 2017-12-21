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

#plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import figure, show

'''
#INPUT DATA 
'''

in_path = ".../input.shp"
out_path = ".../result.shp"

df = gpd.read_file(in_path)

# rename the column of the selected attributes with a short name (here used: 3 attributes)
df.rename(columns={"attribute 1": "k1", "attribute 2": "k2", "attribute 3": "k3"},inplace=True)

#simplest plot
#df.plot(column='k1', cmap='OrRd', scheme='quantiles', k=4, edgecolor='black', figsize=(7,7), legend= True)


fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(21,7))
df.plot(column='k1', cmap='OrRd', scheme='quantiles', k=4, edgecolor='black', legend= True , ax=axes[0])
df.plot(column='k2', cmap='OrRd', scheme='quantiles', k=4, edgecolor='black', legend= True, ax=axes[1])
df.plot(column='k3', cmap='OrRd', scheme='quantiles', k=4, edgecolor='black', legend= True, ax=axes[2])
fig.suptitle('INDICATOR QUANTILES MAPS', fontsize=16)
axes[0].set_title("VAMPIRE", fontstyle='italic')
axes[1].set_title("IRSD", fontstyle='italic')
axes[2].set_title("OECD", fontstyle='italic')


'''
#INPUT PARAMETER
'''

weigth_type = 'r' # 'o' = original binary, 'r' = row-stand.

permutations = 999 # number of random permutations

significance = 0.01

'''
#SPATIAL WEIGHTS AND ATTRIBUTE MATRICES EXTRACTION
'''

w = ps.weights.Queen.from_dataframe(df)
w.transform= weigth_type
wf = w.full()[0]

# inverting attributes to give them an equal meaning verse (eg High = good, Low = bad)

#a=1/df['k1']
#b=1/df['k2']
#c=1/df['k3']
#
#att_arrs_norm = [(a-a.mean())/a.std(),
#                 (b-b.mean())/b.std(), 
#                 (c-c.mean())/c.std()]

att_arrs = [df['k1'],df['k2'],df['k3']]
att_mtx = np.array(att_arrs).transpose()

att_arrs_norm = [(df['k1']-df['k1'].mean())/df['k1'].std(),
                 (df['k2']-df['k2'].mean())/df['k2'].std(), 
                 (df['k3']-df['k3'].mean())/df['k3'].std()]
att_mtx_norm = np.array(att_arrs_norm).transpose()

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
# INFERENCE UNDER RANDOMIZATION ASSUMPTION (CONDITIONAL PERMUTATIONS)
'''

#simulated statistics 

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

# simulated p-values based on permutations (one-sided), null: spatial randomness

p_sim = np.zeros((np.shape(att_arrs_norm)[1]))

for i in range(0,np.shape(att_arrs_norm)[1]):
    above = C_ki_perm[i] > C_ki[i] 
    larger = above.sum(0)
    if (permutations - larger) < larger:
        larger = permutations - larger
    p_sim[i] = ((larger + 1.0) / (permutations + 1.0))
    
    
'''
# ADD TO THE ATTRIBUTE TABLE THE COMPUTED STATISTICS
''' 
df['k1_stand']= (df['k1']-df['k1'].mean())/df['k1'].std()
df['k2_stand']= (df['k2']-df['k2'].mean())/df['k2'].std()
df['k3_stand']=  (df['k3']-df['k3'].mean())/df['k3'].std()
df['C_ki'] = C_ki    
df['p_sim'] = p_sim
df['z_sim'] = C_ki_z_sim
df['p_norm'] = p_norm
df['z_norm'] = C_ki_z_norm


# define locations of interest in the dataset and add flags 

sig = p_sim <= (significance/3)

corr_lower =  C_ki >= np.mean(C_ki)

corr_higher = C_ki <np.mean(C_ki)


locations = np.zeros((np.shape(att_arrs_norm)[1]))

locations[sig*corr_higher] = 1
locations[sig*corr_lower] = -1

df['sig_loc'] = locations


'''-------- TEST UNIVARIATE LOCAL MORAN'S I '''

y1 = df['k1']
lm1 = ps.Moran_Local(y1, w, transformation = weigth_type, permutations = permutations)

sig = lm1.p_sim <= significance
locations = np.zeros((np.shape(att_arrs_norm)[1]))
locations[lm1.q==1 * sig] = 1
locations[lm1.q==3 * sig] = 1
locations[lm1.q==2 * sig] = -1
locations[lm1.q==4 * sig] = -1

df['lm1'] = locations

#--------------

y2 = df['k2']
lm2 = ps.Moran_Local(y2, w, transformation = weigth_type, permutations = permutations)


sig = lm2.p_sim <= significance
locations = np.zeros((np.shape(att_arrs_norm)[1]))
locations[lm2.q==1 * sig] = 1
locations[lm2.q==3 * sig] = 1
locations[lm2.q==2 * sig] = -1
locations[lm2.q==4 * sig] = -1
df['lm2'] = locations

#--------------

y3 = df['k3']
lm3 = ps.Moran_Local(y3, w, transformation = weigth_type, permutations = permutations)


sig = lm3.p_sim <= significance
locations = np.zeros((np.shape(att_arrs_norm)[1]))
locations[lm3.q==1 * sig] = 1
locations[lm3.q==3 * sig] = 1
locations[lm3.q==2 * sig] = -1
locations[lm3.q==4 * sig] = -1

df['lm3'] = locations



'''-------- TEST BIVARIATE LOCAL MORAN'''

y1 = df['k1']
x1 = df['k2']
lmb1 = ps.esda.moran.Moran_Local_BV(x1, y1, w, transformation = weigth_type, permutations = permutations)

sig = lmb1.p_sim <= (significance/2)
locations = np.zeros((np.shape(att_arrs_norm)[1]))
locations[lmb1.q==1 * sig] = 1
locations[lmb1.q==3 * sig] = 1
locations[lmb1.q==2 * sig] = -1
locations[lmb1.q==4 * sig] = -1

df['lmb1'] = locations

#--------------

y2 = df['k1']
x2 = df['k3']
lmb2 = ps.esda.moran.Moran_Local_BV(x2, y2, w, transformation = weigth_type, permutations = permutations)

sig = lmb2.p_sim <= (significance/2)
locations = np.zeros((np.shape(att_arrs_norm)[1]))
locations[lmb2.q==1 * sig] = 1
locations[lmb2.q==3 * sig] = 1
locations[lmb2.q==2 * sig] = -1
locations[lmb2.q==4 * sig] = -1

df['lmb2'] = locations

#--------------

y3 = df['k2']
x3 = df['k3']
lmb3 = ps.esda.moran.Moran_Local_BV(x3, y3, w, transformation = weigth_type, permutations = permutations)

sig = lmb3.p_sim <= (significance/2)
locations = np.zeros((np.shape(att_arrs_norm)[1]))
locations[lmb3.q==1 * sig] = 1
locations[lmb3.q==3 * sig] = 1
locations[lmb3.q==2 * sig] = -1
locations[lmb3.q==4 * sig] = -1

df['lmb3'] = locations

#--------------

y4 = df['k2']
x4 = df['k1']
lmb4 = ps.esda.moran.Moran_Local_BV(x4, y4, w, transformation = weigth_type, permutations = permutations)

sig = lmb4.p_sim <= (significance/2)
locations = np.zeros((np.shape(att_arrs_norm)[1]))
locations[lmb4.q==1 * sig] = 1
locations[lmb4.q==3 * sig] = 1
locations[lmb4.q==2 * sig] = -1
locations[lmb4.q==4 * sig] = -1

df['lmb4'] = locations

#--------------

y5 = df['k3']
x5 = df['k1']
lmb5 = ps.esda.moran.Moran_Local_BV(x5, y5, w, transformation = weigth_type, permutations = permutations)

sig = lmb5.p_sim <= (significance/2)
locations = np.zeros((np.shape(att_arrs_norm)[1]))
locations[lmb5.q==1 * sig] = 1
locations[lmb5.q==3 * sig] = 1
locations[lmb5.q==2 * sig] = -1
locations[lmb5.q==4 * sig] = -1

df['lmb5'] = locations

#--------------

y6 = df['k3']
x6 = df['k2']
lmb6 = ps.esda.moran.Moran_Local_BV(x6, y6, w, transformation = weigth_type, permutations = permutations)

sig = lmb6.p_sim <= (significance/2)
locations = np.zeros((np.shape(att_arrs_norm)[1]))
locations[lmb6.q==1 * sig] = 1
locations[lmb6.q==3 * sig] = 1
locations[lmb6.q==2 * sig] = -1
locations[lmb6.q==4 * sig] = -1

df['lmb6'] = locations

'''-------- SAVE THE MODIFIED GEODATAFRAME TO A NEW SHAPEFILE '''

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
axes[0].set_title("VAMPIRE", fontstyle='italic')
axes[1].set_title("IRSD", fontstyle='italic')
axes[2].set_title("OECD", fontstyle='italic')

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

# plot univar and bivar cluster and multivar significant locations

# local moran
fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(21,7))
df.plot(column='lm1', cmap = 'bwr', edgecolor='black', legend= True, categorical=True, ax=axes[0])
df.plot(column='lm2', cmap='bwr', edgecolor='black', legend= True, categorical=True, ax=axes[1])
df.plot(column='lm3', cmap='bwr', edgecolor='black', legend= True, categorical=True, ax=axes[2])
fig.suptitle("LOCAL MORAN'I MAPS", fontsize=16)
axes[0].set_title("VAMPIRE", fontstyle='italic')
axes[1].set_title("IRSD", fontstyle='italic')
axes[2].set_title("OECD", fontstyle='italic')

# local moran bivariate
fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(21,7))
df.plot(column='lmb1', cmap = 'bwr', edgecolor='black', legend= True, categorical=True, ax=axes[0])
df.plot(column='lmb2', cmap= 'bwr', edgecolor='black', legend= True, categorical=True, ax=axes[1])
df.plot(column='lmb3', cmap= 'bwr', edgecolor='black', legend= True, categorical=True, ax=axes[2])
fig.suptitle("LOCAL MORAN BIVARIATE MAPS", fontsize=16)
axes[0].set_title("VAMPIRE vs IRSD", fontstyle='italic')
axes[1].set_title("VAMPIRE vs OECD", fontstyle='italic')
axes[2].set_title("IRSD vs OECD", fontstyle='italic')

fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(21,7))
df.plot(column='lmb4', cmap = 'bwr', edgecolor='black', legend= True, categorical=True, ax=axes[0])
df.plot(column='lmb5', cmap='bwr', edgecolor='black', legend= True, categorical=True, ax=axes[1])
df.plot(column='lmb6', cmap='bwr', edgecolor='black', legend= True, categorical=True, ax=axes[2])
fig.suptitle("LOCAL MORAN BIVARIATE MAPS", fontsize=16)
axes[0].set_title("IRSD vs VAMPIRE", fontstyle='italic')
axes[1].set_title("OECD vs VAMPIRE", fontstyle='italic')
axes[2].set_title("OECD vs IRSD", fontstyle='italic')


# bivariate geary's c

#simplest plot
#df.plot(column='sig_locations', cmap='RdYlBu_r', edgecolor='black', legend= True, categorical=True,figsize=(10,10))

fig, ax = plt.subplots(1, figsize=(8,10))
ax = df.plot(column='sig_loc', cmap='bwr', edgecolor='black', legend= True, categorical=True, axes=ax)
fig.suptitle("MULTIVARIATE SPATIAL ASSOCIATION - GEARY'S C",fontsize=16)
ax.set_title("VAMPIRE - IRSD - OECD" , fontstyle='italic')
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



