#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the python libraries to be used
from tensorflow import keras # Python library for developing and evaluating deep learning models
import pandas as pd # python library for data science/data analysis and machine learning tasks
from sklearn.metrics import mean_absolute_error #Used to measure the perfomance of a model
from keras.models import Sequential # used for analysis and comparison of simple neural network-oriented models
from keras.layers import Dense, LSTM, Dropout, GRU #developing and evaluating deep learning models
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


# Load the dataset
Data=pd.read_csv("D:\Ben Important\Master Data Analytics\MSC 2.1 Notes\Data Analytics and Knowledge Engineering\AirPassengers.csv")
#Renaming the column names since the hashtag would render it a comment
Data.rename(columns={'#Passengers':'Passengers'}, inplace=True)


# In[3]:


df=pd.DataFrame(Data)
df.sample(5)


# # 3i) LSTM

# In[4]:


from sklearn.model_selection import train_test_split
x=df.Month
y=df.Passengers
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[5]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train.values.reshape(-1,1))
x_test = scaler.transform(x_test.values.reshape(-1,1))
y_train = scaler.fit_transform(y_train.values.reshape(-1,1))
y_test = scaler.transform(y_test.values.reshape(-1,1))


# In[6]:


model = Sequential()
model.add(LSTM(80, activation="tanh", return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(40, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss="mean_squared_error",optimizer="adam")
model.fit(x_train,y_train,epochs= 2000,batch_size=32)


# In[7]:


predicted_data=model.predict(x_test)
predicted_data=scaler.inverse_transform(predicted_data)


# In[15]:


y_test1=scaler.inverse_transform(y_test)
x_train=scaler.inverse_transform(x_test)
x_test=scaler.inverse_transform(x_test)


# In[16]:


plt.figure(figsize=(8,6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(y_test1,color="r",label="True-result")
plt.plot(predicted_data,color="g",label="Predicted-result")
plt.legend()
plt.title("LSTM PLOT")
plt.xlabel("Time Step")
plt.ylabel("Number of Air Passengers")
plt.grid(True)
plt.show()


# # 3ii) GRU

# In[10]:


gru=Sequential()
gru.add(GRU(units=80, activation="tanh", return_sequences=True, input_shape=(x_train.shape[1],1)))
gru.add(Dropout(0.2))
gru.add(GRU(units=40, return_sequences=True))
gru.add(Dropout(0.2))
gru.add(Dense(1))
gru.compile(optimizer='adam', loss='mean_squared_error')
gru.fit(x_train,y_train,epochs= 2000,batch_size=32)


# In[11]:


predicted_data1=gru.predict(x_test)
predicted_data1=predicted_data1.reshape(44,-1)
predicted_data1=scaler.inverse_transform(predicted_data1)


# In[12]:


plt.figure(figsize=(8,6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(y_test1,color="b",label="True-result")
plt.plot(predicted_data1,color="y",label="Predicted-result")
plt.legend()
plt.title("GRU PLOT")
plt.xlabel("Time Step")
plt.ylabel("Number of Air Passengers")
plt.grid(True)
plt.show()


# In[13]:


gru.save('gru-modell.h5')


# In[ ]:




