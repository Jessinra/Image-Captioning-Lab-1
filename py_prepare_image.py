#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
tf.enable_eager_execution()


# In[3]:


import numpy as np
import os
import time
import json
import pickle
from glob import glob
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# In[4]:


import keras as K


# In[5]:


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)


# In[6]:


annotation_folder = '../Dataset/MSCOCO/annotations/'
image_folder = '../Dataset/MSCOCO/train2014/'


# In[7]:


import os
os.environ['http_proxy']="http://jessin:77332066@cache.itb.ac.id:8080"
os.environ['https_proxy']="https://jessin:77332066@cache.itb.ac.id:8080"


# In[8]:


annotation_file = annotation_folder + 'captions_train2014.json'

# Read the json file
with open(annotation_file, 'r') as f:
    annotations = json.load(f)


# In[12]:


# Store captions and image names
all_captions = []
all_img_paths = []

for annot in annotations['annotations']:
    caption = "START " + annot['caption'] + " END"
    image_id = annot['image_id']
    img_path = image_folder + 'COCO_train2014_' + '%012d.jpg' % (image_id)

    all_img_paths.append(img_path)
    all_captions.append(caption)
    
all_captions, all_img_paths = shuffle(all_captions, all_img_paths, random_state=1)


# In[22]:


NUM_SAMPLES = len(all_captions)


# In[ ]:


# # Shuffle captions and image_names together
# all_captions, all_img_paths = shuffle(all_captions, all_img_paths, random_state=1)
# train_captions = all_captions[:NUM_SAMPLES]
# train_img_paths = all_img_paths[:NUM_SAMPLES]


# In[15]:


train_captions = all_captions
train_img_paths = all_img_paths


# ## Image feature extractor

# In[17]:


def get_image_feature_extractor(model_type="xception"):

    if model_type == "xception":
        cnn_preprocessor = tf.keras.applications.xception
        cnn_model = tf.keras.applications.Xception(include_top=False, weights='imagenet')

    elif model_type == "inception_v3":
        cnn_preprocessor = tf.keras.applications.inception_v3
        cnn_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        
    else:
        raise Exception("CNN encoder model not supported yet")

    input_layer = cnn_model.input
    output_layer = cnn_model.layers[-1].output # use last hidden layer as output
    
    encoder = tf.keras.Model(input_layer, output_layer)
    encoder_preprocessor = cnn_preprocessor
    
    return encoder, encoder_preprocessor


# In[18]:


MODEL_TYPE = "xception"
# Shape of the vector extracted from xception is (100, 2048)
# Shape of the vector extracted from InceptionV3 is (64, 2048)

extractor, extractor_preprocessor = get_image_feature_extractor(MODEL_TYPE)


# In[19]:


IMAGE_SIZE = (299, 299)


def load_image(image_path):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = extractor_preprocessor.preprocess_input(image)
    
    return image, image_path


# ## Prepare Image dataset

# In[20]:


BATCH_SIZE = 16


# Get unique images
unique_train_img_paths = sorted(set(train_img_paths))

# Prepare dataset
image_dataset = tf.data.Dataset.from_tensor_slices(unique_train_img_paths)
image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) # use max num of CPU
image_dataset = image_dataset.batch(BATCH_SIZE)


# In[23]:


estimated_batch_count = NUM_SAMPLES / BATCH_SIZE + 1
print("estimated_batch_count", estimated_batch_count)


# In[24]:


# Preprocessed image (batch)

for batch_imgs, batch_img_paths in tqdm(image_dataset):
    
    # get context vector of batch images
    batch_features = extractor(batch_imgs)
    
    # flatten 2D cnn result into 1D for RNN decoder input
    # (batch_size, 10, 10, 2048)  => (batch_size, 100, 2048)
    # image_feature = 100 (Xception)
    # image_feature = 64 (Inception V3)
    batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
    
    # Cache preprocessed image
    for image_feature, image_path in zip(batch_features, batch_img_paths):
        image_path = image_path.numpy().decode("utf-8")
        np.save(image_path, image_feature.numpy())
