# import libraries 
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import nbinom
from scipy import misc
from itertools import combinations
%matplotlib inline
from scipy import spatial

# Make the dataframe from the file
df = pd.read_table('w10_data.tbl.txt', delim_whitespace=True,header=None)
dimensions = len(df.loc[0,:])
p = len(df.loc[:,0])

# Take the log of the data
data = df.loc[1:,:].values
data = data.astype(float)
data = np.log(data)


# Function to initialize centroids for N dimensions and K centroids
def initialize_centroids(k, data):
    '''
    Initialize the centroid values for N dimensions and K clusters randomly, bounded by max and min of data

    Parameters
    ==========

    n : int
        number of dimensions
        
    p : int
        number of 200 cells (datapoints)

    k : int 
        number of clusters

    df : dataframe

    return centroids
        N x K array where N is number of dimensions and K is number of centroids
    '''
    # Pick a random 8 x 2001 array from the data
    centroids = data[np.random.choice(data.shape[0],k)] 
    
    return centroids

# Test 8 x 2001
centroids = initialize_centroids(8,data) 


def single_kmeans(data, centroids, n, k, p):
    '''
    Do single kmeans sorting
    
    Parameters
    ==========
    
    centroids : n x k matrix of floats
        initial crappy centroid guesses
        
    n : int
        number of dimensions
        
    k : int
        number of clusters
        
    p : int
        number of datapoints (cells)
        
    df : dataframe
    
    return the vector that contains all cluster assignments for all cell types
    '''
    
    # calculate the distance from each point to each cluster 
        
    distances = np.zeros((p,k))
    for j in range(p):
        for i in range(k):
            distances[j,i] = spatial.distance.euclidean(data[j,:],centroids[i,:])
    
    # Assign each of the 200 cells to a cluster based on minimum distance, this is a 1 x 200 matrix assigning cells to clusters 
    assignments = np.zeros((200,1))
    cell_dist = np.zeros((200,1))
    for m in range(p):
        assignments[m] = np.argmin(distances[m,:])
        cell_dist[m] = np.min(distances[m,:])
        totdist = np.sum(cell_dist)
       
    # Returns length 8 list of lists that have the cell indexes that belong to that cluster
    indexes = [np.where(np.asarray(assignments) == i) for i in range(k)]
    
    # Make sure all clusters have at least one assigned point
    for i in range(k):
        if len(indexes[i][0]) == 0:
            indexes = [np.where(np.asarray(assignments))]

    # Now calculate new centroid values from the means of the columns
    for j in range(k):
        centroids[j,:] = np.mean(data[indexes[j][0],:],axis=0)
        
    return centroids, assignments, totdist

test = single_kmeans(data, centroids, 2001, 8, 200)

# Run kmeans several times
totdist_min = -1000000000
totdist = -10000000
centroids = initialize_centroids(8, data)
a = list(range(0,8))*100
assignments = np.asarray(a[0:200])
assignments_prev = np.asarray(a[10:210])

for m in range(100):
    while np.any(assignments_prev != assignments):
        assignments_prev = assignments
        totdist_min = totdist
        [centroids,assignments,totdist] = single_kmeans(data, centroids, 2001, 8, 200)
        if totdist < totdist_min:
            totdist_min = totdist
print(totdist_min)



# Perform normalization
for i in range(2001):
    denoms = np.mean(data,axis=0)
    data[:,i] = data[:,i] - denoms[i]

# Plug into SVD
# u is p x p (deviation of each observation from each eigenvalue)
# s is 1 x 200 --> multiply by eyematrix to get p x p (says weights of eigenvalues (how much to spread the data))
# w is n x n --> truncate to perform multiplication to p x n (direction of each PC)
[u,s,w] = np.linalg.svd(data)

V = w # is the directions of each eigenvectors
s = s*np.identity(p-1) 

# m = np.dot((u*s),w.T) # p x p * p x p * p x n  = p x n

# Function to get eigenvalues (limited by observations (200))
def get_eigen(s,n):
    return [(sval**2)/(n-1) for sval in s]

eigenvalues = np.diag(get_eigen(s,2001))

# Function to get PCs
def get_PC(V,data):
    return np.dot(V,data.T)

PC = get_PC(V,data)

# Get scores
Scores = np.dot(data,PC)
 # first two colums are the two dimesnions we cluster along 

# 1. Plot all 200 expression patterns 
plt.scatter(PC[0],PC[1])
plt.show()

# 2 and 3. Plot the eigenvalues for crappy data and the good data
crappy_eigenvalues = pd.read_table('w10_eigen.tbl.txt', delim_whitespace=True,header=None)
CE = crappy_eigenvalues.loc[:,1:].values
CE = CE.astype(float)
plt.plot(eigenvalues,'o')
plt.plot(CE,'o')
plt.show()

# 4. Loadings (the first two eigenvectors)
plt.scatter(V[0],V[1])
plt.show()
# plot 0.05 lines to box in the threshold

noncrapdata = []
for i in range(2001):
    if abs(V[0,i]) > 0.05 or abs(V[1,i]) > 0.05:
        noncrapdata.append(i)
print("There are " + str(len(noncrapdata)) + " important genes")