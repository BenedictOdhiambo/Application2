#!/usr/bin/env python
# coding: utf-8

# ## Name:Benedict Odhiambo
# ## Reg no:21/08516
# ## Unit:MDA(Data Analytics and Knowledge Engineering)
# ## Assignment 4

# In[14]:


import streamlit as st#library for creating web apps
import numpy as np#library for working with arrays
from keras.models import load_model#library used to load the saved model
savedModel=load_model('gru-modell.h5')#loading the saved model to be used in the creation of the web app
savedModel.summary()#Displaying to see the summay of the model

def main():#defining the main function
    st.title("Air Passengers Predictor")#Title of the streamlit app
    st.header('Enter the timestamp:')#header instructing the user on what to do
    timestamp = st.number_input('timestamp:', min_value=0, max_value=1000, value=1)#defining the timestamp which is the input
    if st.button('Predict Passengers'):#creting the button to be clicked in order to run thr predicting command
        st.code(scaler.inverse_transform(savedModel.predict(timestamp)))#Defining how the prediction will be performed
        st.success('The predicted number of passengers is;')#output to be displayed when the prediction is successful
if __name__=='__main__': #running the script
        main() #starting point of the execution of the program


# In[ ]:





# In[ ]:




