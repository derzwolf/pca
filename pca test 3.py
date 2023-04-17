#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[8]:


## Insert your code here
df = pd.read_csv('kidney_disease.csv', sep = ';' ,index_col='id')
df.head()


# In[9]:


df.describe()


# In[10]:


## Insert your code here
df.drop_duplicates()
df.value_counts()


# In[11]:


df.describe()
df.info()


# In[12]:


cat_df=df.select_dtypes(include='O')
cat_df


# In[13]:


num_df=df.select_dtypes(include=['int64','float64'])
num_df


# In[38]:


## Insert your code here
num_df.fillna(num_df.mode(),inplace=True)
num_df


# In[15]:


num_df.sample(10)


# In[16]:


## Insert your code here
cat_df.mode()


# In[17]:


## Insert your code here
cat_df_encoded =cat_df.replace({'yes': 1, 'no': 0, 'normal':1, 'abnormal':0, 'present':1, 'notpresent':0, 'good':1,'poor':0})


# In[18]:


cat_df_encoded


# In[19]:


cat_df_encoded.info()


# In[39]:


## Insert your code here
cat_df_encoded.fillna(cat_df_encoded.mode(),inplace=True)
cat_df_encoded


# In[21]:


cat_df_encoded.info()


# In[22]:


cat_df_encoded.isna().sum()


# In[40]:


new_df = pd.concat([cat_df_encoded,num_df],axis=1)
new_df


# In[ ]:





# In[ ]:





# In[ ]:





# In[41]:


## Insert your code here
#new_df = num_df.join(cat_df_encoded)
#new_df.info()


# In[42]:


#new_df.head()


# In[43]:


new_df.isna().sum()


# In[ ]:


## nein.
#cat_df_encoded.isna().sum()
#num_df.isna().sum()


# In[31]:


## Insert your code here
#num_df.classification.value_counts()


# In[44]:


## Insert your code here
target = new_df.classification
target


# In[45]:


data = new_df.drop('classification', axis =1)
data


# In[47]:


## Insert your code here
sc = StandardScaler()
Z = sc.fit_transform(num_df)


# In[ ]:


## Insert your code here
acp = PCA()
acp_coord = acp.fit_transform(Z)


# In[ ]:


## Insert your code here
aev = acp.explained_variance_


# In[ ]:


plt.plot(np.arange(1, 24), aev)
plt.xlabel('Factor number')
plt.ylabel('Eigenvalues')
plt.show()


# In[ ]:


#n=3


# In[ ]:


df_acp1 = pca.components_


# In[ ]:


## Insert your code here
df_acp = pd.DataFrame(
    {'PC1': pca.components_[:, 0], 'PC2': pca.components_[:, 1]})


# In[ ]:


plt.plot(df_acp1, aev)
plt.xlabel('Factor number')
plt.ylabel('Eigenvalues')
plt.show()


# In[ ]:


PCA_mat = pd.DataFrame(
    {'AXIS 1': Coord[:, 0], 'AXIS 2': Coord[:, 1], 'target': target})


# In[ ]:


## Insert your code here
sns.scatterplot(x='PC1', y='PC2', hue=target, data=df_acp)
plt.show()


# In[ ]:


racine_valeurs_propres = np.sqrt(acp.explained_variance_)
corvar = np.zeros((23,23))
for k in range(23):
    corvar[:,k] = acp.components_[:,k] * racine_valeurs_propres[k]

#Figure delimitation
fig, axes = plt.subplots(figsize=(10,10))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)

#Displaying variables
for j in range(24):
    plt.annotate(df.columns[j],(corvar[j,0],corvar[j,1]),color='#091158')
    plt.arrow(0,0,corvar[j,0]*0.6,corvar[j,1]*0.6, alpha=0.5, head_width=0.03,color='b' )

#Adding axes
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)

#Circle & Legends
cercle = plt.Circle((0,0),1,color='#16E4CA',fill=False)
axes.add_artist(cercle)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()


# In[ ]:


df_kmeans=pd.DataFrame(df_norm[:,2:4],columns=['Annual Income (k$)', 'Spending Score (1-100)'])


kmeans=KMeans(n_clusters = 2)

## Adjustment 
kmeans.fit(df_kmeans)

## Predictions
y_kmeans = kmeans.predict(df_kmeans)

y_kmeans


# In[ ]:


plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1]);


# In[ ]:



plt.figure(figsize=(12,12))
plt.scatter(df_kmeans[y_kmeans == 0].iloc[:,0],df_kmeans[y_kmeans == 0].iloc[:,1],
            s = 50, c = 'red', label = 'Cluster 1')

plt.scatter(df_kmeans[y_kmeans == 1].iloc[:,0], df_kmeans[y_kmeans == 1].iloc[:,1], 
            s = 50, c = 'blue', label = 'Cluster 2')

plt.scatter(df_kmeans[y_kmeans == 2].iloc[:,0], df_kmeans[y_kmeans == 2].iloc[:,1], 
            s = 50, c = 'green', label = 'Cluster 3')

plt.scatter(df_kmeans[y_kmeans == 3].iloc[:,0], df_kmeans[y_kmeans == 3].iloc[:,1],
            s = 50, c = 'cyan', label = 'Cluster 4')

plt.scatter(df_kmeans[y_kmeans == 4].iloc[:,0], df_kmeans[y_kmeans == 4].iloc[:,1], 
            s = 50, c = 'magenta', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s = 100, c = 'yellow', label ='Centroids')

plt.title('Clusters of customers')
plt.ylabel('Spending Score (1-100)')
plt.legend(loc='best')
plt.show()


# In[49]:


#plt.scatter(y_kmeans[:,0],y_kmeans[:,1]);
#sns.scatterplot(x='y_kmeans[:,0]', y='y_kmeans[:,1])', hue=target, data=y_kmeans)
#plt.show()


# In[ ]:


from collections import Counter

ckm = Counter(y_kmeans)
ckm


# In[ ]:


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
tsne_coord = tsne.fit_transform(Z)


# In[ ]:


tsne_df = pd.DataFrame(
    {'AXIS 1': Coord_TSNE[:, 0], 'AXIS 2': Coord_TSNE[:, 1], 'Target': target})


# In[ ]:


sns.scatterplot(x = 'AXIS 1', y = 'AXIS 2',hue='Target', data =tsne_df )
plt.show()


# In[ ]:


#df_kmeans=pd.DataFrame(df_norm[:,2:4],columns=['Annual Income (k$)', 'Spending Score (1-100)'])


kmeans=KMeans(n_clusters = 2)

## Adjustment 
kmeans.fit(df_kmeans)

## Predictions
y_kmeans2 = kmeans.predict(df_kmeans)

y_kmeans2


# In[ ]:


plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1]);


# In[ ]:



plt.figure(figsize=(12,12))
plt.scatter(df_kmeans[y_kmeans == 0].iloc[:,0],df_kmeans[y_kmeans == 0].iloc[:,1],
            s = 50, c = 'red', label = 'Cluster 1')

plt.scatter(df_kmeans[y_kmeans == 1].iloc[:,0], df_kmeans[y_kmeans == 1].iloc[:,1], 
            s = 50, c = 'blue', label = 'Cluster 2')

#plt.scatter(df_kmeans[y_kmeans == 2].iloc[:,0], df_kmeans[y_kmeans == 2].iloc[:,1], 
            s = 50, c = 'green', label = 'Cluster 3')

#plt.scatter(df_kmeans[y_kmeans == 3].iloc[:,0], df_kmeans[y_kmeans == 3].iloc[:,1],
            s = 50, c = 'cyan', label = 'Cluster 4')

#plt.scatter(df_kmeans[y_kmeans == 4].iloc[:,0], df_kmeans[y_kmeans == 4].iloc[:,1], 
            s = 50, c = 'magenta', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s = 100, c = 'yellow', label ='Centroids')

plt.title('Clusters of customers')
plt.ylabel('Spending Score (1-100)')
plt.legend(loc='best')
plt.show()


# In[ ]:


ckm2 = Counter(y_kmeans)
ckm2


# In[ ]:


#blblalabbla


# In[ ]:


#####

