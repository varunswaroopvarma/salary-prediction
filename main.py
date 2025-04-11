#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

df=pd.read_csv("Salary Data.csv")
df


# In[2]:


df.isna().sum()


# In[3]:


df['Age'].fillna(df['Age'].mean() ,inplace=True)
df['Gender'].fillna(method='ffill' ,inplace=True)
df['Education Level'].fillna(method='ffill' ,inplace=True)
df['Job Title'].fillna(method='ffill' ,inplace=True)
df['Years of Experience'].fillna(df['Years of Experience'].mean() ,inplace=True)
df['Salary'].fillna(df['Salary'].mean() ,inplace=True)


# In[4]:


df.isna().sum()


# In[5]:


df.drop_duplicates(inplace=True)


# In[6]:


df.duplicated().sum()


# In[7]:


import matplotlib.pyplot as plt
data=df['Salary']
plt.boxplot(data)


# In[8]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Initialize and fit encoders
gender_le = LabelEncoder()
education_le = LabelEncoder()
job_le = LabelEncoder()

df['Gender'] = gender_le.fit_transform(df['Gender'])
df['Education Level'] = education_le.fit_transform(df['Education Level'])
df['Job Title'] = job_le.fit_transform(df['Job Title'])



# In[9]:


x=df.drop(columns='Salary',axis=1)


# In[30]:


y=df['Salary']
scaler=StandardScaler()
scaler.fit(x)


# In[31]:


y_reshaped = np.array(y).reshape(-1, 1)


# In[32]:


y_reshaped 


# In[33]:


standard_data=scaler.transform(x)


# In[35]:


scaled_y = scaler.fit_transform(y_reshaped)


# In[36]:


X=standard_data
Y=scaled_y 


# In[37]:



Y


# In[38]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=2)


# In[39]:


print(X.shape,x_train.shape,x_test.shape)


# In[40]:


Regressor=RandomForestRegressor(n_estimators=100, random_state=42)


# In[41]:


Regressor.fit(x_train,y_train)


# In[42]:


x_train_prediction = Regressor.predict(x_train)


mse = mean_squared_error(y_train, x_train_prediction)
mae = mean_absolute_error(y_train, x_train_prediction)
r2 = r2_score(y_train, x_train_prediction)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)


# In[43]:


y_test_prediction = Regressor.predict(x_test)


mse_test = mean_squared_error(y_test, y_test_prediction)
mae_test = mean_absolute_error(y_test, y_test_prediction)
r2_test = r2_score(y_test, y_test_prediction)

print("Test Data - Mean Squared Error:", mse_test)
print("Test Data - Mean Absolute Error:", mae_test)
print("Test Data - R-squared Score:", r2_test)


# In[ ]:

import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(Regressor, f)

with open('gender_encoder.pkl', 'wb') as f:
    pickle.dump(gender_le, f)

with open('education_encoder.pkl', 'wb') as f:
    pickle.dump(education_le, f)

with open('job_encoder.pkl', 'wb') as f:
    pickle.dump(job_le, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)