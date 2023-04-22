#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[48]:


fab=pd.read_csv('Data for price of houses at Offa Garage.csv')
fab


# In[49]:


fab.dtypes


# In[50]:


fab.shape


# In[51]:


fab.head()


# In[52]:


fab.tail()


# In[53]:


fab.columns


# In[54]:


columns = (['Unnamed: 2',
        'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7',
       'Unnamed: 8', 'Unnamed: 9' , 'AGBA AKIN & ASSOCIATES',])

fab=fab.drop(columns, axis=1)
fab


# In[55]:


fab.isnull().sum()


# In[56]:


fab.loc[0]
fab.drop([0],axis=0,inplace=True)


# In[57]:


fab


# In[58]:


fab.isnull().sum()


# In[59]:


fab['Unnamed: 3']


# In[60]:


#fab.astype(int)
# for items in fab['Unnamed: 3']:
    
#     items.str.replace('#',"")
#     print(items)

fab['Unnamed: 3']=fab['Unnamed: 3'].str.replace('#','')
fab['Unnamed: 3']=fab['Unnamed: 3'].str.replace(',','')

fab


# In[61]:


fab['Unnamed: 3'].dtypes
fab['Unnamed: 3']=fab['Unnamed: 3'].astype(int)


# In[62]:


fab.describe()


# In[63]:


fab


# In[64]:


fab.dtypes


# In[65]:


fab.shape


# In[66]:


fab.head()


# In[ ]:





# In[67]:


# fab


# In[68]:


fab['COST_OF_HOUSE']=fab['Unnamed: 3']
fab


# In[69]:


col=['Unnamed: 3']
fab=fab.drop(col,axis=1)


# In[70]:


fab


# In[71]:


fab['COMMISSION']=fab['COST_OF_HOUSE']*0.1
fab


# In[72]:


fab


# In[73]:


# plt.xlabel('COST OF HOUSE')
# plt.ylabel('TYPE OF APARTMENT')
# plt.scatter(fab.COMMISSION,fab.COMMISSION,color="orange",marker="o");


# In[74]:


fab.columns


# In[75]:


fab["TYPE OF APARTMENT."].hist()
plt.title("COST_OF_HOUSE");


# In[76]:


import seaborn as sns
sns.heatmap(fab.corr(),annot=True)


# In[77]:


# plt.xlabel('2 Bedroom flat')
# plt.ylabel('3 Bedroom flat')
# plt.scatter(fab.2 Bedroom flat,fab.3 Bedroom flat,color="orange",marker="o")


# In[78]:


# sns.countplot(x="COST OF HOUSE",hue="Pclass",TYPE OF APARTMENT=fab)


# In[79]:


fab["COST_OF_HOUSE"].value_counts()


# In[80]:


plt.figure(figsize=(10,4))
sns.countplot(x='COST_OF_HOUSE',data=fab);


# In[81]:


sns.countplot(x="TYPE OF APARTMENT.",hue="COST_OF_HOUSE",data=fab);


# In[82]:


fab.plot.pie(title='Std Mark',y='COST_OF_HOUSE', 
            fontsize=20,startangle=40);


# In[83]:



X = fab.drop("TYPE OF APARTMENT.",axis=1)
y = fab['COST_OF_HOUSE']



# In[84]:


from sklearn.linear_model import LogisticRegression


# In[85]:


model = LogisticRegression()


# In[86]:


from sklearn.model_selection import train_test_split
X_test,X_train,y_test,y_train = train_test_split(X,y,test_size=0.3,random_state=42)


# In[87]:


model.fit(X_train,y_train)


# In[ ]:





# In[88]:


X_train_prediction=model.predict(X_train)


# In[89]:


from sklearn.metrics import accuracy_score


# In[94]:


training_data_accuracy=accuracy_score(y_train,X_train_prediction)


# In[95]:


print("Accuracy score of training data:",training_data_accuracy)


# In[ ]:





# In[93]:




