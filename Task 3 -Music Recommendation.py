#!/usr/bin/env python
# coding: utf-8

# ![Screenshot%20%2847%29.png](attachment:Screenshot%20%2847%29.png)

# # Virtual Internship Program

# ##  <span style="color:red">  Author :- Ankit Sharma </span>
# ## <span style="color:red">Beginer Level Tasks</span>

# # Task-3 Music Recommendation

# Music recommender systems can suggest songs to users based on their listening patterns.

# Datasetlink :- https://www.kaggle.com/c/kkbox-music-recommendation-challenge/data

# ##  <span style="color:blue"> 1. Importing The Libraries</span>

# pip install recommender-system

# In[15]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ##  <span style="color:blue"> 2. Load Music Data</span>

# In[38]:


train = pd.read_csv('train.csv')
songs = pd.read_csv('songs.csv')
members = pd.read_csv('members.csv')


# In[39]:


df_train.head()


# In[40]:


df_songs.head()


# In[41]:


df_songs_extra.head()


# In[42]:


df_test.head()


# ##  <span style="color:blue"> 3. Create New data</span>

# In[43]:


res = df_train.merge(df_songs[['song_id','song_length','genre_ids','artist_name','language']], on=['song_id'], how='left')
res.head()


# In[44]:


train = res.merge(df_songs_extra,on=['song_id'],how = 'left')
train.head()


# In[45]:


song_id = train.loc[:,["name","target"]]
song1 = song_id.groupby(["name"],as_index=False).count().rename(columns = {"target":"listen_count"})
song1.head()


# In[46]:


dataset=train.merge(song1,on=['name'],how= 'left')


# In[47]:


df=pd.DataFrame(dataset)


# In[48]:


df.drop(columns=['source_system_tab','source_screen_name','source_type','target','isrc'],axis=1,inplace=True)
df=df.rename(columns={'msno':'user_id'})


# ##  <span style="color:blue"> 4. Loading New data</span> 

# In[49]:


df.head()


# ##  <span style="color:blue"> 5. Preprocessing of data</span> 

# In[50]:


df.shape


# In[51]:


#checking null values
df.isnull().sum()


# In[52]:


#fill the null values
df['song_length'].fillna('0',inplace=True)
df['genre_ids'].fillna('0',inplace=True)
df['artist_name'].fillna('none',inplace=True)
df['language'].fillna('0',inplace=True)
df['name'].fillna('none',inplace=True)
df['listen_count'].fillna('0',inplace=True)


# In[53]:


#Recheck null values
df.isnull().sum()


# In[32]:


print("Total no of songs:",len(df))


# ##  <span style="color:blue"> 6. Subset of The Dataset</span>

# In[54]:


df = df.head(10000)
df['song'] = df['name'].map(str) + " - " + df['artist_name']


# ##  <span style="color:blue"> 7. Most Popular Songs in The Dataset</span>

# In[55]:


song_gr = df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_gr['listen_count'].sum()
song_gr['percentage']  = song_gr['listen_count'].div(grouped_sum)*100
song_gr.sort_values(['listen_count', 'song'], ascending = [0,1])


# ##  <span style="color:blue"> 8. Unique Users in The Dataset </span>

# In[56]:


users = df['user_id'].unique()
print("The no. of unique users:", len(users))


# ##  <span style="color:blue"> 8. Number of Unique Songs in The Dataset</span>

# In[57]:


songs = df['song'].unique()
len(songs)


# ##  <span style="color:blue"> 8. Create a Song Recommender</span>

# In[58]:


train_data, test_data = train_test_split(df, test_size = 0.20, random_state=0)
print(train.head(5))


# ## <span style="color:blue"> 9. Data Visualization</span>

# In[59]:


plt.figure(figsize=(10,10))
sns.countplot(x='source_system_tab', hue='source_system_tab', data=train)


# In[60]:


plt.figure(figsize=(10,10))
sns.countplot(x='source_system_tab', hue='target', data=train)


# In[61]:


plt.figure(figsize=(10,10))
plt.xticks(rotation=90)
sns.countplot(x='source_screen_name', hue='target',data=train)


# In[62]:


plt.figure(figsize=(10,10))
plt.xticks(rotation=90)
sns.countplot(x='source_type', hue='source_type',data=train)


# In[63]:


plt.figure(figsize=(10,10))
plt.xticks(rotation=90)
sns.countplot(x='source_type', hue='target',data=train)


# In[65]:


plt.figure(figsize=(10,10))
sns.countplot(x='registered_via', hue='registered_via',data=members)


# ## <span style="color:blue"> 10. data Cleaning</span> 

# In[66]:


ntr = 7000
nts = 3000
names=['msno','song_id','source_system_tab','source_screen_name','source_type','target']
test1 = pd.read_csv('train.csv',names=names,skiprows=ntr,nrows=nts)


# In[67]:


test = test1.drop(['target'],axis=1)
ytr = np.array(test1['target'])


# In[68]:


test_name = ['id','msno','song_id','source_system_tab','source_screen_name','source_type']
test['id']=np.arange(nts)
test = test[test_name]


# In[71]:


members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))


# In[72]:


members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
members = members.drop(['registration_init_time'], axis=1)


# In[73]:


members_cols = members.columns
train = train.merge(members[members_cols], on='msno', how='left')
test = test.merge(members[members_cols], on='msno', how='left')


# In[74]:


train = train.fillna(-1)
test = test.fillna(-1)


# In[75]:


import gc
del members, songs; gc.collect();
import warnings
warnings.filterwarnings('ignore')


# In[76]:


cols = list(train.columns)
cols.remove('target')


# In[81]:


from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder   


# In[83]:


unique_songs = range(max(train['song_id'].max(), test['song_id'].max()))
song_popularity = pd.DataFrame({'song_id': unique_songs, 'popularity':0})
train_sorted = train.sort_values('song_id')
train_sorted.reset_index(drop=True, inplace=True)
test_sorted = test.sort_values('song_id')
test_sorted.reset_index(drop=True, inplace=True)


# # <span style="color: Green">  Thank You </span>
