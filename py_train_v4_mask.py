#!/usr/bin/env python
# coding: utf-8

# !git add "Keras.ipynb"
# !git commit -m "test pred using pplm"

# In[ ]:


PARAMS = {
    "rnn_units": 512,
    "rnn_type": "LSTM",
    "tokenizer": "BERT",
    "word_embedding": "BERT",
    "vocab_size": 3000,
    "combine_strategy": "merge",
    "combine_layer": "concat",
    "image_context_size": 256,
    "word_embedding_dim": 256,
    "batch_size": 32,
    "data_size": 30000,
    "use_mapping": True,
    "learning_rate": 0.001,
    "max_caption_length": 25, # use <int> or None
    "image_feature_extractor": "xception",
    "use_sequence": True,
    "epoch": 20,
    "version": "v4 : masking",
    "note": "using softmax",
}


# ## Import

# In[ ]:


import tensorflow as tf
tf.__version__


# In[ ]:


import keras.backend as K


# In[ ]:


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


# In[ ]:


import warnings
warnings.filterwarnings(action='ignore')


# ## Setup

# In[ ]:


# tf.compat.v1.disable_eager_execution()


# In[ ]:


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)


# In[ ]:


import os
os.environ['http_proxy']="http://jessin:77332066@cache.itb.ac.id:8080"
os.environ['https_proxy']="https://jessin:77332066@cache.itb.ac.id:8080"

# for TFBertModel
PROXIES = {
  "http": "http://jessin:77332066@cache.itb.ac.id:8080",
  "https": "https://jessin:77332066@cache.itb.ac.id:8080",
}


# ## Load dataset

# In[ ]:


annotation_folder = '../Dataset/MSCOCO/annotations/'
image_folder = '../Dataset/MSCOCO/train2014/'


# In[ ]:


annotation_file = annotation_folder + 'captions_train2014.json'

# Read the json file
with open(annotation_file, 'r') as f:
    annotations = json.load(f)


# In[ ]:


# Store captions and image names
all_captions = []
all_img_paths = []

for annot in annotations['annotations']:
    caption = "[CLS] " + annot['caption'] + " [SEP]"
    image_id = annot['image_id']
    img_path = image_folder + 'COCO_train2014_' + '%012d.jpg' % (image_id)

    all_img_paths.append(img_path)
    all_captions.append(caption)

# Shuffle captions and image_names together
all_captions, all_img_paths = shuffle(all_captions, all_img_paths, random_state=1)

stopper = -1 if PARAMS["data_size"] == "all" else PARAMS["data_size"]
train_captions = all_captions[:stopper]
train_img_paths = all_img_paths[:stopper]


# In[ ]:


print("len train_captions :", len(train_img_paths))
print("len all_captions :", len(all_img_paths))


# In[ ]:


DATA_SIZE = len(train_captions) if PARAMS["data_size"] == "all" else PARAMS["data_size"]


# ## Preprocess image dataset

# ### Image Feature Extractor

# In[ ]:


def get_image_feature_extractor(model_type="xception"):

    if model_type == "xception":
        cnn_preprocessor = tf.compat.v1.keras.applications.xception
        cnn_model = tf.compat.v1.keras.applications.Xception(include_top=False, weights='imagenet')

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


def get_image_feature_shape(model_type):
    
    if model_type == "xception":
        return (100, 2048)
    elif model_type == "inception_v3":
        return (64, 2048)
    else:
        raise Exception ("model unknown")


# In[ ]:


def load_image(image_path):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (299, 299))
    image = extractor_preprocessor.preprocess_input(image)
    
    return image, image_path


# In[ ]:


IMAGE_FEATURE_SHAPE = get_image_feature_shape(PARAMS["image_feature_extractor"])


# In[ ]:


extractor, extractor_preprocessor = get_image_feature_extractor(PARAMS["image_feature_extractor"])


# ### Extract image feature
# 
# this step mostly skipped since the results are cached already

# In[ ]:


BATCH_SIZE = PARAMS["batch_size"]

estimated_batch_count = DATA_SIZE / BATCH_SIZE + 1
print("estimated_batch_count", estimated_batch_count)


# In[ ]:


# # Get unique images
# unique_train_img_paths = sorted(set(train_img_paths))

# # Prepare dataset
# image_dataset = tf.data.Dataset.from_tensor_slices(unique_train_img_paths)
# image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) # use max num of CPU
# image_dataset = image_dataset.batch(BATCH_SIZE)


# In[ ]:


# # Preprocessed image (batch)

# for batch_imgs, batch_img_paths in tqdm(image_dataset):
    
#     # get context vector of batch images
#     batch_features = extractor(batch_imgs)
    
#     # flatten 2D cnn result into 1D for RNN decoder input
#     # (batch_size, 10, 10, 2048)  => (batch_size, 100, 2048)
#     # image_feature = 100 (Xception)
#     # image_feature = 64 (Inception V3)
#     batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
    
#     # Cache preprocessed image
#     for image_feature, image_path in zip(batch_features, batch_img_paths):
#         image_path = image_path.numpy().decode("utf-8")
#         np.save(image_path, image_feature.numpy())


# ## Preprocess Caption dataset

# ### Prepare Tokenizer

# In[ ]:


from transformers import BertTokenizer
from keras.preprocessing.text import Tokenizer


# In[ ]:


TOKENIZER = PARAMS["tokenizer"]
VOCAB_SIZE = PARAMS["vocab_size"]  # Choose the top-n words from the vocabulary


# In[ ]:


class BertTokenizerWrapper(BertTokenizer):
    
    def use_custom_mapping(self, use_mapping=True, vocab_size=3000):
        
        self.use_mapping = use_mapping
        self.cust_vocab_size = vocab_size
        self.mapping_initialized = False

        
    def texts_to_sequences(self, texts):
        """
        convert batch texts into custom indexed version
        eg: ['an apple', 'two person']
        output: [[1037,17260], [2083, 2711]] 
        """
        
        bert_ids = [self.convert_tokens_to_ids(self.tokenize(x)) for x in tqdm(texts)]
        
        if not self.use_mapping:
            return bert_ids
        
        if not self.mapping_initialized:
            self._initialize_custom_mapping(bert_ids)
            return [self._convert_bert_id_to_custom_id(x) for x in tqdm(bert_ids)]
        
        return bert_ids
    
        
    def convert_tokens_to_ids(self, tokens):
        
        bert_ids = super().convert_tokens_to_ids(tokens)
        
        if self.use_mapping and self.mapping_initialized:
            return self._convert_bert_id_to_custom_id(bert_ids)
        else:
            return bert_ids
        
        
    def convert_ids_to_tokens(self, token_ids):
        
        if self.use_mapping and self.mapping_initialized:
            bert_ids = self._convert_custom_id_to_bert_id(token_ids)
        else:
            bert_ids = token_ids
            
        bert_tokens = super().convert_ids_to_tokens(bert_ids)
        return bert_tokens
    
    
    def _initialize_custom_mapping(self, corpus_bert_ids):
        
        print("    > constructing custom mapping < \n")
        self._build_occurence_table(corpus_bert_ids)
        self._build_custom_mapping_table()
        self.mapping_initialized = True
        
        
    def _build_occurence_table(self, tokenized_captions):
        """
        build dict of token frequency
        """
        
        self.occurence_table = {}
        for caption in tqdm(tokenized_captions):
            for token in caption:
                if token not in self.occurence_table:
                    self.occurence_table[token] = 0
                self.occurence_table[token] += 1
                
    
    def _build_custom_mapping_table(self):
        
        _special_token = ['[UNK]', '[PAD]']
        _actual_vocab_size = self.cust_vocab_size - len(_special_token)
        
        sorted_occurence = {k: v for k, v in sorted(
            self.occurence_table.items(), reverse=True, key=lambda item: item[1]
        )}
        
        used_tokens = sorted(list(sorted_occurence)[:_actual_vocab_size])
        mapping_size = min(len(used_tokens), _actual_vocab_size)
        
        _bert_pad = 0
        _bert_oov = 100
        self._custom_pad = 0
        self._custom_oov = mapping_size + 1
        
        self.bert_id_to_custom_id = {
            _bert_pad: self._custom_pad, 
            _bert_oov: self._custom_oov
        }
        self.custom_id_to_bert_id = {
            self._custom_pad: _bert_pad, 
            self._custom_oov: _bert_oov
        }
        
        for i in range(0, mapping_size):
            bert_token = used_tokens[i]
            self.bert_id_to_custom_id[bert_token] = i + 1    
            self.custom_id_to_bert_id[i + 1] = bert_token
            
        print("Vocab contains {0} / {1} unique tokens ({2:.2f} %)".format(
            len(used_tokens) + 2,\
            len(sorted_occurence),\
            (len(used_tokens) / len(sorted_occurence) * 100)
        ))
        
        sorted_occurence_count = list(sorted_occurence.values())
        used_tokens_count = sum(sorted_occurence_count[:_actual_vocab_size])
        total_tokens_count = sum(sorted_occurence_count)
        
        print("Using {0} / {1} tokens available ({2:.2f} %)".format(
            used_tokens_count,\
            total_tokens_count,\
            (used_tokens_count / total_tokens_count * 100)
        ))        
        
    def _convert_bert_id_to_custom_id(self, token_ids):
        return [self.bert_id_to_custom_id[x] if x in self.bert_id_to_custom_id else self._custom_oov for x in token_ids]
    
    def _convert_custom_id_to_bert_id(self, token_ids):
        return [self.custom_id_to_bert_id[x] for x in token_ids]


# In[ ]:


class TokenizerWrapper(Tokenizer):
    
    def convert_tokens_to_ids(self, tokens):
        return [self.word_index[x] for x in tokens]


# In[ ]:


def get_tokenizer(tokenizer_type, use_mapping, vocab_size):
    
    # Load pre-trained BERT tokenizer (vocabulary)
    if tokenizer_type == "BERT" :
        tokenizer = BertTokenizerWrapper.from_pretrained('bert-base-uncased')
        tokenizer.use_custom_mapping(use_mapping, vocab_size)

    # use default keras tokenizer
    else : 
        tokenizer = TokenizerWrapper(num_words=vocab_size, oov_token="[UNK]")
        tokenizer.fit_on_texts(train_captions)    
        tokenizer.word_index['[PAD]'] = 0
        tokenizer.index_word[0] = '[PAD]'
        
    return tokenizer


# In[ ]:


caption_tokenizer = get_tokenizer(
    tokenizer_type=PARAMS["tokenizer"],
    use_mapping=False,
    vocab_size=0
)

target_tokenizer = get_tokenizer(
    tokenizer_type=PARAMS["tokenizer"],
    use_mapping=PARAMS["use_mapping"],
    vocab_size=PARAMS["vocab_size"],
)


# ### Tokenize

# In[ ]:


caption_tokens = caption_tokenizer.texts_to_sequences(train_captions)
target_tokens = target_tokenizer.texts_to_sequences(train_captions)


# ## Create parallel dataset
# 
# prepare captions and target to support parallel training

# In[ ]:


from keras.preprocessing.sequence import pad_sequences


# In[ ]:


def preprocess_parallel_dataset(img_paths, caption_tokens, target_tokens):
    
    dataset_img_paths = []
    dataset_captions = []
    dataset_target = []

    for i in tqdm(range(0, len(train_img_paths))):
        img = img_paths[i]
        cap = caption_tokens[i]
        tar = target_tokens[i]

        for j in range(1, len(cap)):
            dataset_img_paths.append(img)
            dataset_captions.append(cap[:j])
            dataset_target.append(tar[j])

    dataset_captions = pad_sequences(dataset_captions, maxlen=PARAMS["max_caption_length"], padding='post')
    
    return dataset_img_paths, dataset_captions, dataset_target


# In[ ]:


parallel_img_paths, parallel_captions, parallel_target = preprocess_parallel_dataset(train_img_paths, caption_tokens, target_tokens)


# In[ ]:


DATA_SIZE = parallel_captions.shape[0]
MAX_CAPTION_LENGTH = parallel_captions.shape[1]


# ## Create dataset object

# In[ ]:


# Load the numpy files

def load_dataset(img_name, caption, target):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, caption, target


# In[ ]:


# Create dataset object

dataset = tf.data.Dataset.from_tensor_slices((parallel_img_paths, parallel_captions, parallel_target))


# In[ ]:


# Use map to load the numpy files in parallel
# wrap function into numpy function

dataset = dataset.map(lambda item1, item2, item3: tf.numpy_function(
          load_dataset, [item1, item2, item3], [tf.float32, tf.int32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)


# In[ ]:


# Shuffle and batch

dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# ## Split train eval test

# In[ ]:


# Split dataset 

TRAIN_SPLIT = 0.7
EVAL_SPLIT = 0.15
TEST_SPLIT = 0.15  # approx

n_batch = int(DATA_SIZE / BATCH_SIZE) + 1
n_train = int(n_batch * 0.7)
n_eval = int(n_batch * 0.15)
n_test = n_batch - (n_train + n_eval)

train_dataset = dataset.take(n_train)
eval_dataset = dataset.skip(n_train).take(n_eval)
test_dataset = dataset.skip(n_train + n_eval)


# """
# dataset => tuple of (image, captions)
# image   => (batch_size = 16, image_feature = 100, 2048)
# caption => (batch_size = 16, max_length)
# """


# In[ ]:


print("train: {} batches, (total : {})".format(n_train, n_train * BATCH_SIZE))
print("eval : {} batches, (total : {})".format(n_eval, n_eval * BATCH_SIZE))
print("test : {} batches, (total : {} (aprx))".format(n_test, n_test * BATCH_SIZE))


# -------

# # Model

# ### Captioning Loss

# In[ ]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy


optimizer = Adam(learning_rate=PARAMS["learning_rate"])
loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    """
    real  => (batch_size,)
    pred  => (batch_size, vocab_size)
    """
    
    # calculate loss
    loss_ = loss_object(real, pred)
    
    # create mask to filter out padding token 
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=loss_.dtype)
    
    # Ignore loss_ if real token is padding
    loss_ *= mask
    
    # Get mean of curren batch's loss (somewhat batch norm)
    result_loss = tf.reduce_mean(loss_)
    
    return result_loss


    """
    loss_  => (batch_size, 1)
    mask   => (batch_size, 1)  : indicate is padding or not

    return => (1)
    """


# ### Encoder

# In[ ]:


from tensorflow.keras.layers import Dense


class CNN_Encoder(tf.keras.Model):
    
    # Image features are extracted and saved already
    # This encoder passes those features through a Fully connected layer

    def __init__(self, output_dim=256):
        super(CNN_Encoder, self).__init__()
        self.fc = Dense(output_dim, activation="relu")

    def call(self, x):
        
        # x => (batch_size, 100, 2048)
        x = self.fc(x)
        
        return x
    
        """
        return => (batch_size, image_feature_size, image_context_size)
        """


# ### CNN Attention

# In[ ]:


from tensorflow.keras.layers import Dense


class BahdanauAttention(tf.keras.Model):
    
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        """
        features (CNN_encoder output) => (batch_size, img_feature_size, image_context_size)
        hidden                        => (batch_size, embedding_size)
        
        note : 
        img_feature_size ==  64 for Inception V3,
        img_feature_size == 100 for Xception,
        """
        
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        
        _w1 = self.W1(features)
        _w2 = self.W2(hidden_with_time_axis)
        score = tf.nn.tanh(_w1 + _w2)

        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


        """
        hidden_with_time_axis      => (batch_size, 1, rnn_units)
        score                      => (batch_size, img_feature_size, rnn_units)
        attention_weights          => (batch_size, img_feature_size, 1)
        context_vector (after sum) => (batch_size, img_context_size)
        """


# ### Decoder

# In[ ]:


from tensorflow.keras.layers import Dense, Embedding, Masking, LSTM, GRU
from transformers import TFBertModel


class RNN_Decoder(tf.keras.Model):
    
    def __init__(self, rnn_type="LSTM", rnn_units=256, 
                 embedding_type="BERT", embedding_dim=256, 
                 combine_strategy="merge", combine_layer="concat",
                 vocab_size=3000, batch_size=32):
        
        super(RNN_Decoder, self).__init__()
        self.rnn_units = rnn_units
        self.rnn_type = rnn_type
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        
        # when to use context_vector ["inject_init", "inject_pre", "inject_par", "merge"]
        self.combine_strategy = combine_strategy
        
        # how to use context_vector ["add", "concat"]
        self.combine_layer = combine_layer
        
        # =====================================================
        
        self._init_embedding()
        self._init_rnn()
        
        self.mask = Masking(mask_value=0, dtype="int32")
        
        # dense layer to choose word to generate
        self.fc1 = Dense(self.rnn_units)
        self.fc2 = Dense(self.vocab_size) # same size as vocab
        
        
    def reset(self):
        
        super(RNN_Decoder, self).__init__()
        self._init_embedding()
        self._init_rnn()
           
        # dense layer to choose word to generate
        self.fc1 = Dense(self.rnn_units)
        self.fc2 = Dense(self.vocab_size) # same size as vocab
        
        
    def _init_embedding(self):
        
        # embedding layer (process tokenized caption into vector)
        if self.embedding_type == "BERT":
            self.bert_embedding = TFBertModel.from_pretrained('bert-base-uncased', proxies=PROXIES)
            self.bert_embedding.trainable = False
            self.embedding_dim = self.bert_embedding.config.hidden_size
            
        else:
            self.default_embedding = Embedding(
                input_dim=self.vocab_size, 
                output_dim=self.embedding_dim, 
                mask_zero=True,
            )
        
        
    def _init_rnn(self):
        
        # rnn layer for captions sequence and/or image's context vector'
        
        if self.rnn_type == "LSTM":
            self.lstm = LSTM(self.rnn_units,
                             return_sequences=True,
                             return_state=True,
                             recurrent_initializer='glorot_uniform')
        
        elif self.rnn_type == "GRU":
            self.gru = GRU(self.rnn_units,
                           return_sequences=True,
                           return_state=True,
                           recurrent_initializer='glorot_uniform')
        else:
            raise Exception('RNN type not supported yet (LSTM / GRU only)')
        
        
    def embedding(self, tokens, as_sentence=False):
        """
        Get BERT's embedding for text tokens
        """
        
        if self.embedding_type == "BERT": 
            embedding, sentence_embedding = self._bert_embedding(tokens)
        
        else:
            embedding = self._default_embedding(tokens)
            sentence_embedding = tf.reduce_mean(embedding, 1)
            

        if as_sentence:
            # embedding => (batch_size, embedding_dim)
            return sentence_embedding
        
        else:
            # embedding => (batch_size, tokens_length, embedding_dim)
            return embedding
            
        
        """
        embedding (tokens)   => (batch_size, tokens_length, embedding_dim)
        embedding (sentence) => (batch_size, embedding_dim)
        """
        
        
    def _bert_embedding(self, x):
        
        # mask out attention from padding
        # mask => (batch_size, sequence_length)
        attention_mask = x == 0
        
        # hidden_states => (batch_size, sequence_length, embedding_size)
        # sentence_embedding => (batch_size, embedding_size)
        hidden_states, sentence_embedding = self.bert_embedding(inputs=x, attention_mask=attention_mask)
        return hidden_states, sentence_embedding
    
        """
        hidden states contains hidden state for each word
        sentence_embedding is general embedding for whole sentences
        """
    
    
    def _default_embedding(self, x):
        
        # embedding => (batch_size, sequence_length, embedding_size)
        embedding = self.default_embedding(x)
        return embedding
    
    
    def apply_strategy(self, x, context_vector, curr_iter=-1):
        """
        context_vector : image's vector
        x              : rnn input (word embedding)
        strategy       : 
        curr_iter      : current iteration number
        
        No longer support inject_init & inject_pre, since training become fully parallel
        """
        
#         if self.combine_strategy == "inject_init":
#             initial_state = tf.squeeze(context_vector) if curr_iter == 1 else None
#             output, state = self.rnn_model(x, initial_state=initial_state)  
            
#         elif self.combine_strategy == "inject_pre":
#             x = context_vector if curr_iter == 1 else x
#             output, state = self.rnn_model(x)  
            
        if self.combine_strategy == "inject_par":
            
            context_vector = tf.expand_dims(context_vector, 1)
            context_vector = tf.tile(context_vector, [1, MAX_CAPTION_LENGTH, 1])
                        
            x = self.custom_combine_layer(context_vector, x)
            
            # output => (batch_size, sequence_len, rnn_unit)
            output, state = self.rnn_model(x)              
            
        else: # merge (as default)
            
            # output => (batch_size, sequence_len, rnn_unit)
            output, state = self.rnn_model(x)           
            
            context_vector = tf.expand_dims(context_vector, 1)
            context_vector = tf.tile(context_vector, [1, MAX_CAPTION_LENGTH, 1])
            
            # output => (batch_size, rnn_unit)
            output = self.custom_combine_layer(context_vector, output)
            
        return output, state
    
    
    def rnn_model(self, x, initial_state=None):
        
        if self.rnn_type == "LSTM":
            
            # adjust initial state, LSTM has 2 hidden states (h and c)
            if initial_state is not None:
                init_h = initial_state
                init_c = tf.zeros(initial_state.shape)
                initial_state = [init_h, init_c]
            
            output, h_state, c_state = self.lstm(x, initial_state=initial_state)
            
        elif self.rnn_type == "GRU":
            output, h_state = self.gru(x, initial_state=initial_state)
            
        else:
            raise Exception('RNN type not supported yet (LSTM / GRU only)')
        
        return output, h_state
    
        """
        output => (batch_size, rnn_size)
        """
    
    
    def custom_combine_layer(self, x, y):
        if self.combine_layer == "add":
            return self._add_layer(x, y)
        else:
            return self._concat_layer(x, y)

        
    def _add_layer(self, x, y):
        
        if x.shape[1] != y.shape[1] :
            exception = "Cannot combine using 'add' strategy, both tensor has different shape {} & {}"
            raise Exception(exception.format(x.shape, y.shape))
            
        return tf.keras.layers.add([x, y])
            
        
    def _concat_layer(self, x, y):
        return tf.concat([x, y], axis=-1)
        
    
    def call(self, decoder_input, context_vector):
        """ 
        decoder_input  : last predicted word => (batch_size, 1)
        context_vector : image's vector      => (batch_size, img_context_size)
        """
        
        # Mask out padding (0)
        decoder_input = self.mask(decoder_input)
        
        # x1 => (batch_size, input_sentence_len, embedding_dim)
        x1 = self.embedding(decoder_input)
        
        # x2 (concat) => (batch_size, embedding_dim + image_context_size)
        # x2 (add) => (batch_size, embedding_dim)
        x2, rnn_state = self.apply_strategy(x1, context_vector)
        
        ## ============================================
        ## TODO: add another attention layer ? 
        ## ============================================
        
        # x3 => (batch_size, sequence_len, rnn_units)
        x3 = self.fc1(x2)   # how important is every sequence 
        
        # x4 => (batch_size, sequence_len * rnn_units) || was (batch_size * sequence_len=1, rnn_units)
        x4 = tf.reshape(x3, (x3.shape[0], -1))

        # word_predictions => (batch_size, vocab)
        word_predictions = self.fc2(x4)
        
        return word_predictions, rnn_state
    

    def reset_state(self, batch_size=None):
        
        if batch_size is not None:
            return tf.zeros((batch_size, self.rnn_units))
        
        return tf.zeros((self.batch_size, self.rnn_units))        


# ## Define model

# In[ ]:


PARAMS


# In[ ]:


encoder = CNN_Encoder(
    output_dim=PARAMS["image_context_size"]
)

attention = BahdanauAttention(
    units=PARAMS["rnn_units"]
)

decoder = RNN_Decoder(
    rnn_type=PARAMS["rnn_type"], 
    rnn_units=PARAMS["rnn_units"],
    embedding_type=PARAMS["word_embedding"], 
    embedding_dim=PARAMS["word_embedding_dim"],  
    combine_strategy=PARAMS["combine_strategy"], 
    combine_layer=PARAMS["combine_layer"],
    vocab_size=PARAMS["vocab_size"],
    batch_size=PARAMS["batch_size"]
)


# """
# Requirements

# combine_strategy = "inject_init" : IMAGE_CONTEXT_SIZE == UNITS
# combine_strategy = "inject_pre"  : IMAGE_CONTEXT_SIZE == WORD_EMBEDDING_DIM

# """


# In[ ]:


# Default feed forward function

@tf.function
def feed_forward(img_tensor, caption):
    """
    img_tensor => (batch_size, image_feature_size, 2048)
    caption => (batch_size, max_caption_length)
    """

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    # note : used to be decoder hidden state, 
    # but changed into current word embedding for paralel training
    
    # hidden => (batch_size, embedding_size)
    hidden = decoder.embedding(caption, as_sentence=True)

    # features => (batch_size, img_feature_size, image_context_size)
    features = encoder(img_tensor)

    # context_vector => (batch_size, image_context_size)
    context_vector, _ = attention(features, hidden)

    # predictions => (batch_size, vocab_size)
    predictions, _ = decoder(caption, context_vector)

    return predictions


    """
    predictions => (batch_size, vocab_size)
    decoder_input => tf.Tensor: id=11841, shape=(batch_size, 1), dtype=int32
    """


# In[ ]:


@tf.function
def train_step(img_tensor, caption, target):
    """
    target => (batch_size, 1)
    """
    
    # Training model
    with tf.GradientTape() as gradient_tape:
        
        # predictions => (batch_size, vocab_size)
        predictions = feed_forward(img_tensor, caption)
        
        # loss => Tensor("add:0", shape=(), dtype=float32)
        loss = loss_function(target, predictions)

        
    # Apply gradient
    trainable_variables = encoder.trainable_variables +                           decoder.trainable_variables +                           attention.trainable_variables
    
    gradients = gradient_tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss



@tf.function
def eval_step(img_tensor, caption, target):
    
    # predictions => (batch_size, vocab_size)
    predictions = feed_forward(img_tensor, caption)
    
    # loss => Tensor("add:0", shape=(), dtype=float32)
    loss = loss_function(target, predictions)
    
    return loss


# In[ ]:


@tf.function
def choose_predicted_id(predictions, strategy="max", sampling_k=10):
    """
    predictions : encoder word prediction => (batch_size, vocab_size)
    strategy    : how to choose word ["sample", "max"]
    """
    
    # Sampling method (categorical dist)
    if strategy == "sample":
        
        # sampled_proba & sampled_ids => (batch_size, sampling_k)
        sampled_proba, sampled_ids = tf.math.top_k(predictions, sampling_k)
        
        # chosen_sampled_col => (batch_size, )
        chosen_sampled_col = tf.squeeze(tf.random.categorical(sampled_proba, 1))
        
        # create row idx to zip with chosen_sampled_col
        row_idx = tf.range(predictions.shape[0], dtype=chosen_sampled_col.dtype)
        row_col_idx = tf.stack([row_idx, chosen_sampled_col], axis=1)
        
        # predicted_ids => (batch_size, )
        predicted_ids = tf.gather_nd(sampled_ids, row_col_idx)

    # Max index method
    else:
        predicted_ids = tf.argmax(predictions, 1)
    
    # predicted_ids => (batch_size, )
    return predicted_ids


# ### Support using image

# In[ ]:


def get_image_features(images_paths):
    """
    images_paths => (batch_size, 1)
    """
    
    # Extract images features
    images = [load_image(x)[0] for x in images_paths]
    
    # x => (batch_size, 299, 299, 3)
    x = tf.convert_to_tensor(images)
    
    # x => (batch_size, 10, 10, 2048)
    x = extractor(x)
    
    # x  => (batch_size, img_feature_size, 2048)
    x = tf.reshape(x, (x.shape[0], -1, x.shape[3]))
    
    # features => (batch_size, img_feature_size, image_context_size)
    features = encoder(x)
    
    return features


def get_supporting_features(images_paths, strategy="mean"):
    """
    images_paths => (batch_size, img_count, 1)
    strategy : strategy to aggregate multiple supporting image ["logsumexp", "mean", "min", "max"]
    """
    
    # Extract images features
    images = [[load_image(x)[0] for x in images_set] for images_set in images_paths]
    
    # x => (batch_size, img_count, 299, 299, 3)
    x = tf.convert_to_tensor(images)
    
    # x => (batch_size, img_count, 10, 10, 2048)
    x = [extractor(image_set) for image_set in x]
    
    # features => (batch_size, img_count, img_feature_size, image_context_size)
    features = encoder(x)
    
    # features => (batch_size, img_feature_size, image_context_size)
    if strategy == "logsumexp":
        features = tf.reduce_logsumexp(features, 1)
    elif strategy == "max":
        features = tf.reduce_max(features, 1)
    elif strategy == "min":
        features = tf.reduce_min(features, 1)
    else:
        features = tf.reduce_mean(features, 1)

    return features


# ### Support using text

# In[ ]:


def get_one_hot_indices(support_text):
    
    indices = []
    for i in range(0, len(support_text)):
        context_token = target_tokenizer.tokenize(support_text[i])
        context_token_id = target_tokenizer.convert_tokens_to_ids(context_token)
        context_token_id = set(context_token_id)
        context_token_id.discard(0)
        for x in sorted(context_token_id):
            indices.append([i, x])

    # return => (word_count, 2)
    return indices


def get_supporting_text_vector(support_text, vocab_size):
    """
    support_text : list of text describing main image context => (batch_size)
    """
    
    batch_size = len(support_text)
    
    # indices => ( sum(batch_size * ?word_count), 2)
    indices = get_one_hot_indices(support_text)
    values = tf.ones(len(indices))
    sparse_one_hot = tf.sparse.SparseTensor(indices, values, dense_shape=[batch_size, vocab_size])
    
    # sparse_one_hot => (batch_size, vocab_size)
    return sparse_one_hot


# In[ ]:


def custom_evaluate(images_paths,
                    support_text=None,
                    support_imgs=None, 
                    support_aggregate_strategy="mean",
                    pplm_iteration=3,
                    pplm_weight=0.03,
                    pplm_gm_weight=0.8,
                    choose_word_strategy="sample",
                   ):
    
    """
    images_paths : list of image_path                           => (batch_size, 1)
    support_text : list of text describing main image context   => (batch_size)
    support_imgs : list of list of image_path                   => (batch_size, image_count, 1)
    support_aggregate_strategy : how to aggregate support image ["logsumexp", "mean", "min", "max"]
    pplm_iteration : number of pplm step done for every decoding step
    pplm_weight    : weight of pplm loss
    pplm_gm_weight : geometric mean fusion weight (0 means use only original prediction, 1 means use only pplm prediction)
    choose_word_strategy : how to choose word from prediction distribution ["sample", "max"]
    """
    
    batch_size = len(images_paths)
    
    # initialize captions placeholder
    start_token = caption_tokenizer.convert_tokens_to_ids(['[CLS]']) # use bert id's not custom id
    result_captions = tf.tile(tf.expand_dims(start_token, 1), [batch_size, 1])
    attention_plot = tf.reshape([], shape=(batch_size, 0, IMAGE_FEATURE_SHAPE[0]))
    
    # Extract features from main images
    # features => (batch_size, img_feature_size, image_context_size)
    features = get_image_features(images_paths)
    
    
    if support_text is not None:
        # support_text_vector => (batch_size, vocab_size)
        support_text_vector = get_supporting_text_vector(support_text, decoder.vocab_size) 
    else:
        # set all pplm related variable to 0
        pplm_iteration = 0
        pplm_weight = 0
        pplm_gm_weight = 0
        
        
    for i in tqdm(range(MAX_CAPTION_LENGTH)):
        
        # decoder_input => (batch_size, ~MAX_CAPTION_LENGTH)
        caption = pad_sequences(result_captions, maxlen=MAX_CAPTION_LENGTH, padding="post")
        
        # hidden => (batch_size, embedding_size)
        hidden = decoder.embedding(caption, as_sentence=True)  ## BOTTLE

        # context_vector => (batch_size, image_context_size)
        # attention_weights => (batch_size, img_feature_size, 1)
        context_vector, attention_weights = attention(features, hidden)

        # predictions => (batch_size, vocab_size)
        predictions, _ = decoder(caption, context_vector)  
        
        # ======================== PPLM section ========================
        ori_prediction = predictions
        
        curr_pertubation = tf.Variable(tf.zeros((batch_size, decoder.embedding_dim)), name="curr_pertubation", trainable=True)
        
        
        for j in range(pplm_iteration):
            
            with tf.GradientTape() as pplm_tape: 
                
                hidden += curr_pertubation
                
                context_vector, attention_weights = attention(features, hidden)
                predictions, _ = decoder(caption, context_vector)
                pplm_loss = pplm_loss_function(support_text_vector, predictions, pplm_weight=pplm_weight)
                
            """
            most impactfull layer to train = last dense layer
            """
            
            trainable_variables = [curr_pertubation]
            gradients = pplm_tape.gradient(pplm_loss, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))
            
            predictions, _ = decoder(caption, context_vector)
        
        # fuse final pplm_prediction and original prediction
        fused_predictions = (predictions * pplm_gm_weight) + (ori_prediction * (1 - pplm_gm_weight)) 
        
        # predicted_ids => (batch_size,)
        predicted_ids = choose_predicted_id(fused_predictions, strategy=choose_word_strategy)
        
        # convert custom id mapping to bert's id
        # predicted_bert_ids => (batch_size, 1)
        predicted_bert_ids = target_tokenizer._convert_custom_id_to_bert_id(predicted_ids.numpy())
        predicted_bert_ids = tf.expand_dims(predicted_bert_ids, 1)
        
        # store result
        result_captions = tf.concat([result_captions, predicted_bert_ids], axis=1)
        
        # attention_weights => (batch_size, 1, img_feature_size)
        attention_weights = tf.reshape(attention_weights, shape=(batch_size, 1, -1))
        
        # assign attention weights to respective generated word
        # attention_plot => (batch_size, ~max_caption_len, feature_size)
        attention_plot = tf.concat([attention_plot, attention_weights], axis=1)
        
    # remove start token & revert to tokens
    result_captions = [caption_tokenizer.convert_ids_to_tokens(x[1:]) for x in result_captions]
    
    return result_captions, attention_plot


# In[ ]:


from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy, MeanSquaredError


@tf.function
def pplm_loss_function(real, pred, pplm_weight=0.03):
    """
    real  => (batch_size, vocab_size)
    pred  => (batch_size, vocab_size)
    """

#     mm = tf.sparse.sparse_dense_matmul(real, tf.transpose(pred))
#     mm = tf.reduce_sum(tf.abs(mm), 1)
#     loss = tf.reduce_sum(mm, 0)
    
    real = tf.sparse.to_dense(real, default_value=0)
    pplm_loss = CategoricalCrossentropy(from_logits=True)
    loss = pplm_loss(real, pred, pplm_weight)
    
#     print(loss)
    return loss

    """
    return => (1)
    """


# ## Setup training

# In[ ]:


# ! rm -rf checkpoints


# In[ ]:


from datetime import datetime

checkpoint_path = "./checkpoints/train/{}".format(str(datetime.now())[:-10])
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           attention=attention,
                           optimizer=optimizer
                          )
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)


# In[ ]:


print("----------------------")
print(checkpoint_path)
print("----------------------")


# In[ ]:


start_epoch = 0
loss_plot = []


# In[ ]:


import os

os.mkdir(checkpoint_path)
with open(checkpoint_path + "/config.txt", "w") as f:
    f.write(str(PARAMS))
    
log_file = open(checkpoint_path + "/log.txt", "a")


# In[ ]:


if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)


# ## Actual training

# In[ ]:


for epoch in range(start_epoch, PARAMS["epoch"]):
    
    start = time.time()
    loss = 0
    batch = 1

    for img_tensor, captions, target in tqdm(train_dataset):
        
        batch_loss = train_step(img_tensor, captions, target)
        loss += batch_loss

        if batch % 100 == 0:
            
            with open(checkpoint_path + "/log.txt", "a") as f:
                log_message = ' {} Epoch {} Batch {} Loss {:.4f}'.format(
                    checkpoint_path, epoch + 1, batch, batch_loss.numpy()
                )
                f.write(str(log_message + "\n"))
                print(log_message)   

        batch += 1
        
        # storing the epoch end loss value to plot later
        loss_plot.append(batch_loss.numpy())

    ckpt_manager.save()

    print ('Epoch {} Loss {:.6f}'.format(epoch + 1, loss))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


    print ('Epoch {} Loss {:.6f}'.format(epoch + 1, loss))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
