#!/usr/bin/env python
# coding: utf-8

# # HomeWork 3
# Adam Kulas  
# watiam: akulas  
# student number: 20302000  
# 
# References:
# https://archive.ics.uci.edu/ml/machine-learning-databases/communities/

# In[1]:


import pandas as pd
import numpy as np
from numpy import linalg as LA
from IPython.display import display, HTML
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

pd.options.display.max_rows = None
pd.options.display.max_columns = None


# In[2]:


data = pd.read_csv('data/communities.data', header=None, na_values='?')
with open('data/communities.names') as f:
    columns = [line.split(' ')[1] for line in f.readlines() if line.startswith('@attribute ')]
data.columns = columns
print(data.shape)
display(data.head(15))


# ## Clean Data
# 
# Some if the data is missing in the dataset. This missing data is filled with the mean value of the feature.

# In[3]:


data_filled = data
data_filled.iloc[:,4:] = data.iloc[:,4:].apply(lambda x: x.fillna(x.mean()),axis=0)
data_filled.columns = columns
display(data_filled.head(30))


# ## Perform PCA
# 
# The data used for the PCA starts from column 4 onwards.  
# This decision was made because the county state or community cannot use mean to fill NaN values. Alternatively the mode could be used to replace the NaN values or the sample could have been dropped

# In[4]:


# Create a PCA instance: pca
pca = PCA()

# Fit the data
pca.fit(data_filled.iloc[:,4:])

# Plot the explained variances
n = 20
features = range(pca.n_components_)
plt.bar(features[:n], pca.explained_variance_[:n])
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features[:n])
plt.show()
eigen_values = pd.Series(pca.explained_variance_)[:n]
display(eigen_values)


# In[5]:


# Determine how many features must be kept to retain 95% of the variance
total_variance = np.sum(eigen_values)
num_pca_features = []
for num_components in range(20):
    ratio = np.sum(eigen_values[:num_components]) / total_variance
    num_pca_features.append(ratio)
    print('number of PCs: ', num_components, ' Variance Ratio: ', ratio)


# We can see the top 20 pca features in the above table and scree plot figure.  
# To me there is not clear point where the dimensions could be cut off, however there are a couple techniques we could use to make the decision.
# - Looking at the scree plot, the variance begins to drop more drastically starting at PCA feature 3. Therefore we could arbitrarily choose PCA 0, 1, and 2 to represent the data
# - The second way would be to sum the value of the eigen_values, and choose enough features to retain some ratio of the total variance. We can see that if we keep the top 9 highest variance features we will retain over 95% of the variance

# ## Dimsionality Reduction
# This is what the sample would look like if the first two components are used and plotted on a 2d scatter plot

# In[6]:


# dimensionality reduction
pca = PCA(n_components=2)

# Fit the data
pca.fit(data_filled.iloc[:,4:])
transformed = pca.transform(data_filled.iloc[:,4:])
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys)
plt.show()


# ## Manual Calculation of PCA and eigenvalues
# For completeness the eigenvalues and eigenvectors were computed using only numpy  
# We can see that the values closely resemble sklearn's returned results. Sklearn uses SVD in the background to compute the PCA

# In[7]:


X = data_filled.iloc[:,4:]
n_samples = X.shape[0]
# We center the data and compute the sample covariance matrix.
X -= np.mean(X, axis=0)
cov_matrix = np.dot(X.T, X) / n_samples

eigenValues, eigenVectors = LA.eig(cov_matrix)

idx = eigenValues.argsort()[::-1]   
eigenValues = pd.Series(eigenValues[idx])
eigenVectors = pd.DataFrame(eigenVectors[:,idx])
display(eigenValues[:20])

