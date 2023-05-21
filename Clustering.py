#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_excel("Mall_Customers.xlsx")


# In[3]:


df.head()


# # Univariate Analysis
# 
# 

# In[4]:


df.describe()


# In[5]:


sns.distplot(df['Annual Income (k$)']);


# In[6]:


df.columns


# In[7]:


columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.distplot(df[i])


# In[ ]:





# In[ ]:





# In[8]:


columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.boxplot(data=df, x='Gender', y=df[i])


# In[9]:


df['Gender'].value_counts(normalize=True)


# # Bivariate Analysis

# In[10]:


sns.scatterplot(data=df, x= 'Annual Income (k$)', y='Spending Score (1-100)')


# In[11]:


df=df.drop('CustomerID', axis=1)
sns.pairplot(df,hue='Gender')


# In[12]:


df.groupby(['Gender'])['Age', 'Annual Income (k$)', 'Spending Score (1-100)'].mean()


# In[13]:


df.corr()


# In[14]:


sns.heatmap(df.corr(),annot=True,cmap='coolwarm')


# # Clustering- Univariate, Bivariate, Mulivariate

# In[15]:


cluster = KMeans(n_clusters=3)


# In[16]:


cluster.fit(df[['Annual Income (k$)']])


# In[17]:


cluster.labels_


# In[18]:


df['Income Cluster'] = cluster.labels_
df.head()


# In[19]:


df['Income Cluster'].value_counts()


# In[20]:


cluster.inertia_


# In[21]:


inertia_scores=[]
for i in range (1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)']])
    inertia_scores.append(kmeans.inertia_)


# In[22]:


inertia_scores


# In[23]:


plt.plot(range(1,11),inertia_scores)


# In[24]:


df.groupby('Income Cluster')['Age', 'Annual Income (k$)', 'Spending Score (1-100)',].mean()


# In[25]:


df.columns


# In[26]:


#Bivariate Clustering


# In[30]:


cluster2 = KMeans(n_clusters=5)
cluster2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
df['Spending and Income Cluster']= cluster2.labels_
df.head()


# In[29]:


inertia_scores2=[]
for i in range(1,11):
    kmeans2=KMeans(n_clusters=i)
    kmeans2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
    inertia_scores2.append(kmeans2.inertia_)
plt.plot(range(1,11), inertia_scores2)                   


# In[38]:


centers = pd.DataFrame(cluster2.cluster_centers_)
centers.columns = ['x', 'y']


# In[63]:


plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'], y=centers['y'], s=100,c='black', marker='*')
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue = 'Spending and Income Cluster', palette='viridis' )
plt.savefig('clustering_bivariate.png')


# In[41]:


pd.crosstab(df['Spending and Income Cluster'],df['Gender'], normalize='index')


# In[47]:


df.groupby('Spending and Income Cluster')['Age', 'Annual Income (k$)', 'Spending Score (1-100)',].mean()


# In[48]:


#mulivariate clustering

from sklearn.preprocessing import StandardScaler


# In[49]:


scale = StandardScaler()


# In[52]:


dff = pd.get_dummies(df)
dff.head()


# In[53]:


dff.columns


# In[56]:


dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)',"Gender_Male"]]


# In[57]:


dff.head()


# In[58]:


dff = scale.fit_transform(dff)


# In[59]:


dff = pd.DataFrame(scale.fit_transform(dff))


# In[60]:


inertia_scores3=[]
for i in range(1,11):
    kmeans3=KMeans(n_clusters=i)
    kmeans3.fit(dff)
    inertia_scores3.append(kmeans3.inertia_)
plt.plot(range(1,11), inertia_scores3)     


# In[61]:


df


# In[62]:


df.to_csv('Clustering.csv')


# In[ ]:




