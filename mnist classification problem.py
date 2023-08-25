#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.datasets import mnist


# In[3]:


(train_image,train_label),(test_image,test_label) = mnist.load_data()


# In[4]:


train_image.shape


# In[5]:


test_image.shape


# In[6]:


train_label.shape


# In[7]:


test_label.shape


# In[8]:


from keras.models import Sequential
from keras.layers import Dense


# In[9]:


model = Sequential()
model.add(Dense(512,activation='relu',input_shape=(28*28,)))
model.add(Dense(10,activation='softmax'))


# In[10]:


model.summary()


# In[11]:


model.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[12]:


#preparing the image data
train_image = train_image.reshape((60000,28*28))
train_image =train_image.astype('float32')/255


# In[13]:


test_image = test_image.reshape((10000,28*28))
test_image =test_image.astype('float32')/255


# In[14]:


model.fit(train_image,train_label,epochs=10, batch_size=128)


# In[15]:


loss,acc = model.evaluate(test_image,test_label)


# In[16]:


print(acc)


# In[17]:


# there is a little bit gap in between training accuracy and test accuracy is an exa

