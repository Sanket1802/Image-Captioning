#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
from time import time
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50,preprocess_input,decode_predictions
from keras.preprocessing import image as im
from keras.models import Model,load_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers import Input,Dense,Dropout,Embedding,LSTM
from keras.layers.merge import add
import re
import json


# In[2]:


model=load_model("model_weights/model_9.h5")


# In[3]:


model_temp=ResNet50(weights="imagenet",input_shape=(224,224,3))


# In[4]:


model.summary()


# In[5]:


model_resnet=Model(model_temp.input,model_temp.layers[-2].output)


# In[14]:


def preprocess_img(img):
    img=im.load_img(img,target_size=(224,224))
    img=im.img_to_array(img)
    img=np.expand_dims(img,axis=0)#original image size is(224,224,3) and when we feed it to a model it goes in a certain batch size (b,224,224,3) which is 4d tensor so to convert this image we use expand_dims(img,axis=0) so now image looks like (1,224,224,3), we can also use reshape
    #normalization
    img= preprocess_input(img)
    return img


# In[11]:


def encode_image(img):
    img=preprocess_img(img)
    feature_vector=model_resnet.predict(img)
    #print(feature_vector.shape)
    feature_vector=feature_vector.reshape(1,2048)
    return feature_vector


# In[12]:


enc=encode_image("1149851.jpg")


# In[13]:


enc.shape


# In[18]:


def predict_captions(photo):
    max_len=35
    in_text="startseq"
    for i in range(max_len):
        sequence=[word2index[w] for w in in_text.split() if w in word2index]
        sequence=pad_sequences([sequence],maxlen=max_len,padding='post')
        
        ypred=model.predict([photo,sequence])
        ypred=ypred.argmax()
        word=index2word[ypred]
        in_text+=(' '+word)
        
        if word=='endseq':
            break
            
    final_caption=in_text.split()[1:-1]
    final_caption=' '.join(final_caption) 
    return final_caption
        


# In[37]:


with open("word2index.pkl",'rb') as w2i:
    word2index=pickle.load(w2i)
with open("index2word.pkl",'rb') as i2w:
    index2word=pickle.load(i2w)    


# In[38]:

def caption_this_image(image):
    enc=encode_image(image)
    caption=predict_captions(enc)
    return caption

# #### 

# In[ ]:





# In[ ]:




