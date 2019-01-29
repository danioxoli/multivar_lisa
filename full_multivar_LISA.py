# -*- coding: utf-8 -*-
"""
#
# Project:  Multivariate LISA 
#
# Purposes: a) Multivariate Local Geary c computation (Anselin, 2018) 
#           b) Di statistics computation (Oxoli, n.a.)
#           c) Multivariate spatial clusters and outliers classification Mm_i (Oxoli, n.a.)
#
# target data: Vector layers (GeoDataframe)
#            
# Author:   Daniele Oxoli (daniele.oxoli@polimi.it)
#
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

# simulated p-values correction 
import statsmodels.stats.multitest as mt

#runtime check
import time 

'''
# INPUT DATA and PARAMETERS SET UP
'''
start = time.time()

in_path = "/Users/daniele/Google Drive/Delivery&Application/FOSS4GIT_2019/dati/comuni_2012.shp"
out_path = "/Users/daniele/Google Drive/Delivery&Application/FOSS4GIT_2019/outputs/results.shp"

df = gpd.read_file(in_path)

# have a look into the df attributes and list them
list(df)

# rename columns containing the attribute values and list them (on your choice)
attributes_dic = {'landslide': 'lslide'}
df.rename(columns=attributes_dic,inplace=True)


# invert attributes value to meet comparable attributes semantics
#df['inv_income'] = df['income'].pow(-1.0)

att_list = ['gvi', 'flood','lslide']

# normalize attributes for the analysis
att_list_norm = []
for att in att_list:
    df['n_'+att]= (df[att]-df[att].mean())/df[att].std()
    att_list_norm.append('n_'+att)


# simple quantile plot of the selected attributes
i = 0

fig, axes = plt.subplots(nrows=1, ncols=len(att_list_norm), sharey=True, figsize=(7*len(att_list_norm),7))
for att_norm in att_list_norm:
    df.plot(column=att_norm, cmap='OrRd', scheme='quantiles', k=4, edgecolor=None, legend= True , ax=axes[i])
    axes[i].set_title(att_list[i], fontstyle='italic')
    i += 1

# parameters set up
weigth_type = 'o' # 'o' = original binary, 'r' = row-stand.

permutations = 9999 # number of random permutations

significance = 0.05

# spatial weight matrix creation
w = ps.weights.Queen.from_dataframe(df)
w.transform= weigth_type
wf = w.full()[0]


# fit the normalized attribute arrays into a matrix
att_arrs_norm = [df[att_norm] for att_norm in att_list_norm]
att_mtx_norm = np.array(att_arrs_norm).transpose()
n_of_att = np.shape(att_mtx_norm)[1]

'''
# MULTIVARIATE LOCAL GEARY'C COMPUTATION 
'''

neigh = w.neighbors
neigh_dic = {key: neigh[key] for key in list(df.index)}


d_square_list = []
C_ki_list = []

for keys in neigh_dic:
    loi = np.hstack((keys,neigh_dic[keys]))
    loi_att_mtx = att_mtx_norm[loi, :]
    focal_att_array = att_mtx_norm[keys, :]
    wf_loi = wf[keys, loi]
    
    for i in range(0, n_of_att):
        dist_square = ((loi_att_mtx[:, i] - focal_att_array[i])**2)
        d_square_list.append(dist_square)
        
    d_square_array = np.array(d_square_list)
    C_vi = np.sum(wf_loi * d_square_array, axis=1)
    C_ki_list.append(np.sum(C_vi))
    d_square_list = []

C_ki = np.array(C_ki_list) # observed statistics vector
    
# add results to the dataframe
df['C_ki'] = C_ki   


# MM_i CLASSIFICATION  (Di)

class_Cki = []

Mmc_Di_ref = df[att_list_norm].median().mean()


for keys in neigh_dic:
    loi = np.hstack((keys,neigh_dic[keys]))    
    if df['C_ki'].iloc[keys] <= df['C_ki'].mean(): # in case of candidate cluster
        Mmc_i = df[att_list_norm].iloc[loi, :].median().mean()
        if Mmc_i > Mmc_Di_ref:
            class_Cki.append('hh')
        else:
            class_Cki.append('ll')    
    else: # in case of candidate outlier
        Mmo_Di_ref = df[att_list_norm].iloc[neigh_dic[keys], :].median().mean()
        Mmo_i = df[att_list_norm].iloc[keys, :].mean()
        if Mmo_i > Mmo_Di_ref:
            class_Cki.append('hl')
        else:
            class_Cki.append('lh')

# add results to the dataframe
df['class_Cki'] = class_Cki

# LOCAL GEARY'C INFERENCE 

# normality assumption
C_ki_z_norm = (C_ki - np.mean(C_ki))/np.std(C_ki) # standard variates from standard normal approximation
C_p_norm = st.norm.sf(abs(C_ki_z_norm))*2 #p-values based on standard normal approximation
C_p_norm_fdr  = mt.fdrcorrection(C_p_norm, alpha=significance) # correction for p-values based on standard normal approximation (FDR)

# add results to the dataframe
df['C_p_norm'] = C_p_norm
df['C_norm_fdr'] = C_p_norm_fdr[1]
df['C_z_norm'] = C_ki_z_norm


# conditional permutations 
np.random.seed(123)

C_ki_perm_list = []

for k in range(0,permutations):
    C_ki_list = []
    perm_att = []
    Di_perm = []
    for a in range(0,n_of_att):
        perm_att_a = np.random.permutation(att_arrs_norm[a])
        perm_att.append(perm_att_a)
    perm_mtx = np.array(perm_att).transpose()
   
    for keys in neigh_dic:
        loi = np.array(neigh_dic[keys])
        focal_att_array = att_mtx_norm[keys, :] # keep focal location observed attributes
        loi_att_mtx = np.vstack((focal_att_array, perm_mtx[loi, :]))
        wf_loi = np.hstack(([0.0], wf[keys, loi]))
        
        for i in range(0, n_of_att):
            dist_square = ((loi_att_mtx[:, i] - focal_att_array[i])**2)
            d_square_list.append(dist_square)
            
        d_square_array = np.array(d_square_list)
        C_vi = np.sum(wf_loi * d_square_array, axis=1)
        C_ki_list.append(np.sum(C_vi))
        d_square_list = []
    
    C_ki_perm_list.append(C_ki_list)
    

# simulated statistics matrix 
C_ki_perm = np.array(C_ki_perm_list).transpose()


## save simulated stats in a file for future computations
#np.savetxt("/Users/daniele/Google Drive/Delivery&Application/FOSS4GIT_2019/outputs/C_ki_perm64.csv", C_ki_perm , delimiter=",")
#C_ki_perm = np.genfromtxt('/Users/daniele/Google Drive/Delivery&Application/FOSS4GIT_2019/outputs/C_ki_perm99.csv', delimiter=',')


# simulated statistics moments 
E_C_ki_perm= [np.mean(C_ki_perm[i]) for i in range(0,len(C_ki_perm))] # mean from permutations
S_C_ki_perm = [np.std(C_ki_perm[i]) for i in range(0,len(C_ki_perm))] # standard deviation from permutations
C_ki_z_sim = (C_ki - E_C_ki_perm)/S_C_ki_perm # standard variates from permutations

# simulated p-values based on permutations (one-sided), null: spatial randomness

C_p_sim = np.zeros((np.shape(C_ki)[0]))

for i in range(0,np.shape(C_ki)[0]):
    above = C_ki_perm[i] > C_ki[i] 
    larger = above.sum(0)
    if (permutations - larger) < larger:
        larger = permutations - larger
    C_p_sim[i] = ((larger + 1.0) / (permutations + 1.0))

# correction for simulated p-values (FDR)
C_p_sim_fdr = mt.fdrcorrection(C_p_sim*2, alpha=significance)


# add results to the dataframe  
df['C_p_sim'] = C_p_sim*2
df['C_sim_fdr'] = C_p_sim_fdr[1]
df['C_z_sim'] = C_ki_z_sim


# save a shapefile with the results of C computation
df.to_file(driver = 'ESRI Shapefile', filename= out_path)

end_ci = time.time()
print('time_ci')
print(end_ci - start)

'''
# Di COMPUTATION 
'''

## clean dataframe for di (no island)
#df_no_island = df.drop(df.index[w.islands])
#new_index= [i for i in range(0, len(df_no_island))]
#df_no_island.reindex(new_index)
#
## write and read to avoid indexing issues
#df_no_island.to_file(driver = 'ESRI Shapefile', filename= in_path_di)
#df_di = gpd.read_file(in_path_di)

# clean dataframe for di (no island)
df_di = df

# define focal locations with neighbours (loi)
w_Di = ps.weights.Queen.from_dataframe(df_di)
neigh = w_Di.neighbors
neigh_dic = {key: neigh[key] for key in list(df_di.index)}

# full attribute space and attributes count (after island removal)
att_arrs_di = [df_di[att_norm] for att_norm in att_list_norm]
att_mtx_di = np.array(att_arrs_di).transpose()
n_of_att = np.shape(att_mtx_di)[1]


# observed statistics
d_sum = 0
loi_att_mtx = []
D = []

for keys in neigh_dic:
    loi = np.hstack((keys,neigh_dic[keys]))
    # reduce attribute space of the loi
    loi_att_mtx = att_mtx_di[loi, :]
    # centre of mass coordinates
    mc = loi_att_mtx.mean(axis=0)        
    #compute the Di 
    for i in range(0, n_of_att):
        dist = np.sqrt(np.sum((loi_att_mtx[:, i] - mc[i])**2))
        d_sum += dist
                            
    D.append(round(d_sum/len(loi),3))
    d_sum = 0

Di = np.array(D) # observed statistics vector

# add results to the dataframe
df_di['Di'] = Di


# MM_i CLASSIFICATION  (Di)

neigh = w_Di.neighbors
neigh_dic = {key: neigh[key] for key in list(df_di.index)}

class_Di = []

Mmc_Di_ref = df_di[att_list_norm].median().mean()


for keys in neigh_dic:
    loi = np.hstack((keys,neigh_dic[keys]))
    # in case of candidate cluster
    if df_di['Di'].iloc[keys] <= df_di['Di'].mean():
        Mmc_i = df_di[att_list_norm].iloc[loi, :].median().mean()
        if Mmc_i > Mmc_Di_ref:
            class_Di.append('hh')
        else:
            class_Di.append('ll')
    # in case of candidate outlier
    else:
        Mmo_Di_ref = df_di[att_list_norm].iloc[neigh_dic[keys], :].median().mean()
        Mmo_i = df_di[att_list_norm].iloc[keys, :].mean()
        if Mmo_i > Mmo_Di_ref:
            class_Di.append('hl')
        else:
            class_Di.append('lh')
                  
 # add results to the dataframe     
df_di["class_Di"] = class_Di
        
# Di INFERENCE 

# normality assumption
Di_z_norm = (Di - np.mean(Di))/np.std(Di) # standard variates from standard normal approximation
Di_p_norm = st.norm.sf(abs(Di_z_norm))*2 #p-values based on standard normal approximation
Di_p_norm_fdr  = mt.fdrcorrection(Di_p_norm, alpha=significance) # correction for p-values based on standard normal approximation (FDR)

# add results to the dataframe
df_di['Di_p_norm'] = Di_p_norm
df_di['Di_norm_fdr'] = Di_p_norm_fdr[1]
df_di['D_z_norm'] = Di_z_norm


# conditional permutations 
np.random.seed(123)

d_sum_perm = 0
Di_perm_list = []


for k in range(0,permutations):
    perm_att = []
    Di_perm = []
    for a in range(0,n_of_att):
        perm_att_a = np.random.permutation(att_arrs_di[a])
        perm_att.append(perm_att_a)
    perm_mtx = np.array(perm_att).transpose()
   
    for keys in neigh_dic:
        loi = np.hstack((keys,neigh_dic[keys]))
        # reduce attribute space of the loi from permutation
        loi_neigh_perm = perm_mtx[neigh_dic[keys], :]
        loi_att_focal = att_mtx_di[keys, :]
        perm_loi_mxt = np.vstack((loi_att_focal,loi_neigh_perm))
        # centre of mass coordinates
        mc = perm_loi_mxt.mean(axis=0)        
        #compute the Di 
        for i in range(0, n_of_att):
            dist_perm = np.sqrt(np.sum((perm_loi_mxt[:, i] - mc[i])**2))
            d_sum_perm += dist_perm                                
        Di_perm.append(round(d_sum_perm/np.shape(loi)[0],3))               
        d_sum_perm = 0 
    Di_perm_list.append(Di_perm)

# simulated statistics matrix 
Di_sim = np.array(Di_perm_list).transpose()


# simulated statistics moments 
E_Di_perm = np.mean(Di_sim, axis=1)  # mean from permutations
S_Di_perm = np.std(Di_sim, axis=1) # standard deviation from permutations
Di_z_sim = (Di - E_Di_perm)/S_Di_perm # standard variates from permutations

# simulated p-values based on permutations (one-sided), null: spatial randomness
Di_p_sim = np.zeros((np.shape(Di_sim)[0]))

for i in range(0,np.shape(Di_sim)[0]):
    above = Di_sim[i] > Di[i] 
    larger = above.sum(0)
    if (permutations - larger) < larger:
        larger = permutations - larger
    Di_p_sim[i] = ((larger + 1.0) / (permutations + 1.0))

# correction for simulated p-values (FDR)
Di_sim_fdr = mt.fdrcorrection(Di_p_sim*2, alpha=significance)
    

# add results to the dataframe
df_di['Di_p_sim'] = Di_p_sim*2
df_di['Di_sim_fdr'] = Di_sim_fdr[1]
df_di['Di_z_sim'] = Di_z_sim


# save a shapefile with the results of Di computation
df_di.to_file(driver = 'ESRI Shapefile', filename= out_path)

end_di = time.time()
print('time_di')
print(end_di - end_ci)


'''--------------------------------------------------------------------------------'''

# PLOTS C RESULTS

# plot reference distribution from permutations for the i_th location  
i = 89 #location ID

fig, ax = plt.subplots(1, figsize=(10,7))
sns.distplot(C_ki, color='g', label='obs. dist. with mean +- std')
#plt.vlines(C_ki, 0, 0.005, 'r')
#plt.xlim([0.0, 30])
#plt.vlines(np.mean(Di),'g')
#plt.vlines(np.mean(C_ki)-np.std(C_ki), 0, 10, 'g','dotted')
#plt.vlines(np.mean(C_ki)+np.std(C_ki), 0, 10, 'g','dotted')

sns.kdeplot(C_ki_perm[i], shade=True, color="c", label='perm. dist. with mean +- std')
sns.distplot(Di_perm_list[i], color="c", label='perm. dist. with mean +- std')
plt.vlines(C_ki_perm[i], 0, 0.005, 'k', label='perm')
#plt.vlines(np.mean(C_ki_perm[i]), 0, 0.8, 'c')
#plt.vlines(np.mean(C_ki_perm[i])-np.std(C_ki_perm[i]), 0, 10, 'c', 'dashed')
#plt.vlines(np.mean(C_ki_perm[i])+np.std(C_ki_perm[i]), 0, 10, 'c', 'dashed')
plt.xlim([0.0, 30])
plt.vlines(C_ki[i], 0, 0.8, 'r', label='obs statistic')
   
# plot pseudo p-values and computed z-score maps
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,7))
df.plot(column='z_norm', cmap='OrRd_r', scheme='quantiles', k=5, edgecolor='black', legend= True,  ax=axes[0])
df.plot(column='p_sim', cmap='Greys_r', scheme='quantiles', k=5, edgecolor='black', legend= True,  ax=axes[1])
fig.suptitle("Quntiles  maps - Computed statistics", fontsize=16)
axes[0].set_title("Computed MULTIVARIATE GEARY'S C z-norm ", fontstyle='italic')
axes[1].set_title("Pseudo p-values", fontstyle='italic')
plt.show()


# neighbours distribution graph
a = w.histogram
# sort in-place from highest to lowest
a.sort(key=lambda x: x[1], reverse=True) 

# save the beans and their respective elements separately
# reverse the tuples to go from most frequent to least frequent 
n_bean = zip(*a)[0]
score = zip(*a)[1]
x_pos = np.arange(len(n_bean)) 

plt.bar(n_bean, score, align='center', color = 'k')
plt.xticks(n_bean) 
plt.suptitle("Neighbours distribution", fontsize=16)
plt.xlabel('N. of neighbours')
plt.ylabel('N. of locations ')
plt.show()

# plot connectivity grapth
centroids = np.array([list([poly.x, poly.y]) for poly in df.geometry.centroid])

fig = plt.figure(figsize=(9,9))
plt.plot(centroids[:,0], centroids[:,1],'*', color = 'k')
plt.title('Centroids')
show()

fig = figure(figsize=(19,23))

for k,neighs in w.neighbors.items():
    origin = centroids[k]
    for neigh in neighs:
        segment = centroids[[k,neigh]]
        plt.plot(segment[:,0], segment[:,1], '-')
plt.plot(centroids[:,0], centroids[:,1],'*', color = 'k')
plt.title("Queen's case connectivity graph ($1^{st}$ order)", fontsize=16)
show()

