#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.enable_eager_execution()


# In[2]:


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
}


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


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)


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


# In[9]:


# Store captions and image names
all_captions = []
all_img_paths = []

for annot in annotations['annotations']:
    caption = "START " + annot['caption'] + " END"
    image_id = annot['image_id']
    img_path = image_folder + 'COCO_train2014_' + '%012d.jpg' % (image_id)

    all_img_paths.append(img_path)
    all_captions.append(caption)

# Shuffle captions and image_names together
all_captions, all_img_paths = shuffle(all_captions, all_img_paths, random_state=1)

stopper = -1 if PARAMS["data_size"] == "all" else PARAMS["data_size"]
train_captions = all_captions[:stopper]
train_img_paths = all_img_paths[:stopper]


# In[10]:


print("len train_captions :", len(train_img_paths))
print("len all_captions :", len(all_img_paths))


# In[11]:


DATA_SIZE = len(train_captions) if PARAMS["data_size"] == "all" else PARAMS["data_size"]


# ## Image feature extractor

# In[12]:


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


# In[13]:


MODEL_TYPE = "xception"
# Shape of the vector extracted from xception is (100, 2048)
# Shape of the vector extracted from InceptionV3 is (64, 2048)

extractor, extractor_preprocessor = get_image_feature_extractor(MODEL_TYPE)


# In[14]:


IMAGE_SIZE = (299, 299)


def load_image(image_path):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = extractor_preprocessor.preprocess_input(image)
    
    return image, image_path


# ## Prepare Image dataset

# In[15]:


BATCH_SIZE = PARAMS["batch_size"]


# Get unique images
unique_train_img_paths = sorted(set(train_img_paths))

# Prepare dataset
image_dataset = tf.data.Dataset.from_tensor_slices(unique_train_img_paths)
image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) # use max num of CPU
image_dataset = image_dataset.batch(BATCH_SIZE)


# In[16]:


estimated_batch_count = DATA_SIZE / BATCH_SIZE + 1
print("estimated_batch_count", estimated_batch_count)


# In[17]:


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


# ## Prepare Tokenizer

# In[18]:


import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from keras.preprocessing.text import Tokenizer


# In[19]:


TOKENIZER = PARAMS["tokenizer"]
VOCAB_SIZE = PARAMS["vocab_size"]  # Choose the top-n words from the vocabulary


# In[20]:


class BertTokenizerWrapper(BertTokenizer):
    
    def use_custom_mapping(self, use_mapping=True, vocab_size=3000):
        self.use_mapping = use_mapping
        self.vocab_size = vocab_size
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
            return [self._convert_bert_id_to_custom_id(x) for x in bert_ids]
    
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
    
    
    def _set_vocab_size(self, vocab_size):
        self.vocab_size = vocab_size
    
    
    def _build_custom_mapping_table(self):
            
        self.bert_id_to_custom_id = {0:0}
        self.custom_id_to_bert_id = {0:0}
        
        actual_vocab_size = self.vocab_size - 1 # idx 0 for padding & unknown
        
        sorted_occurence = {k: v for k, v in sorted(
            self.occurence_table.items(), reverse=True, key=lambda item: item[1]
        )}
        
        used_tokens = sorted(list(sorted_occurence)[:actual_vocab_size])
        
        for i in range(0, min(len(used_tokens), actual_vocab_size)):
            bert_token = used_tokens[i]
            self.bert_id_to_custom_id[bert_token] = i + 1    # 0 for padding
            self.custom_id_to_bert_id[i + 1] = bert_token    # 0 for padding
            
        print("Vocab contains {0} / {1} unique tokens ({2:.2f} %)".format(
            len(used_tokens),\
            len(sorted_occurence),\
            (len(used_tokens) / len(sorted_occurence) * 100)
        ))
        
        sorted_occurence_count = list(sorted_occurence.values())
        used_tokens_count = sum(sorted_occurence_count[:actual_vocab_size])
        total_tokens_count = sum(sorted_occurence_count)
        
        print("Using {0} / {1} tokens available ({2:.2f} %)".format(
            used_tokens_count,\
            total_tokens_count,\
            (used_tokens_count / total_tokens_count * 100)
        ))
        
    def _convert_bert_id_to_custom_id(self, token_ids):
        
        token_ids = [self.bert_id_to_custom_id[x] if x in self.bert_id_to_custom_id else 0 for x in token_ids]
        return token_ids
                    
    def _convert_custom_id_to_bert_id(self, token_ids):  
        
        token_ids = [self.custom_id_to_bert_id[x] for x in token_ids]
        return token_ids


# In[21]:


class TokenizerWrapper(Tokenizer):
    
    def convert_tokens_to_ids(self, tokens):
        return [self.word_index[x] for x in tokens]


# In[22]:


def get_tokenizer():
    
    if PARAMS["tokenizer"] == "BERT" :

        # Load pre-trained BERT tokenizer (vocabulary)
        tokenizer = BertTokenizerWrapper.from_pretrained('bert-base-uncased')
        tokenizer.use_custom_mapping(
            use_mapping=PARAMS["use_mapping"],
            vocab_size=PARAMS["vocab_size"]
        )

    else : 

        # use default keras tokenizer
        tokenizer = TokenizerWrapper(num_words=PARAMS["vocab_size"], oov_token="[UNK]")
        tokenizer.fit_on_texts(train_captions)    
        tokenizer.word_index['[PAD]'] = 0
        tokenizer.index_word[0] = '[PAD]'
        
    return tokenizer


# In[23]:


tokenizer = get_tokenizer()
train_captions = tokenizer.texts_to_sequences(train_captions)


# ## Pad sequence

# In[24]:


from keras.preprocessing.sequence import pad_sequences


# In[25]:


MAX_LENGTH = None  # use <int> or None


# If you do not provide a max_length value, pad_sequences calculates it automatically
train_captions = pad_sequences(train_captions, maxlen=MAX_LENGTH, padding='post', truncating="post")


# ## Create dataset object

# In[26]:


# Load the numpy files

def load_image_npy(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap


# In[27]:


# Create dataset object

dataset = tf.data.Dataset.from_tensor_slices((train_img_paths, train_captions))


# In[28]:


# Use map to load the numpy files in parallel
# wrap function into numpy function

dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          load_image_npy, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)


# In[29]:


# Shuffle and batch

dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# ## Split train eval test

# In[30]:


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


# In[31]:


print("train: {} batches, (total : {})".format(n_train, n_train * BATCH_SIZE))
print("eval : {} batches, (total : {})".format(n_eval, n_eval * BATCH_SIZE))
print("test : {} batches, (total : {} (aprx))".format(n_test, n_test * BATCH_SIZE))


# -------

# # Model

# ## Encoder

# In[32]:


from tensorflow.keras.layers import Dense


class CNN_Encoder(tf.keras.Model):
    
    # Image features are extracted and saved already
    # This encoder passes those features through a Fully connected layer

    def __init__(self, output_dim=256):
        super(CNN_Encoder, self).__init__()
        self.fc = Dense(output_dim, activation="relu")

    def call(self, x):
        
        x = self.fc(x)
        return x
    
        """
        return => (batch_size, image_feature_size, image_context_size)
        """


# ## CNN Attention

# In[33]:


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
        hidden                        => (batch_size, rnn_units)
        
        note : 
        img_feature_size ==  64 for Inception V3,
        img_feature_size == 100 for Xception,
        """
        
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

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


# In[34]:


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


# ## Decoder

# In[35]:


from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU
from pytorch_pretrained_bert import BertModel


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
            
            self.bert_embedding = BertModel.from_pretrained('bert-base-uncased')
            self.bert_embedding.to('cuda')
            
            self.embedding_dim = self.bert_embedding.config.hidden_size
            # self.vocab_size = self.bert_embedding.config.vocab_size
            
        else:
            self.default_embedding = Embedding(self.vocab_size, self.embedding_dim)
        
        
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
        
        
    def embedding(self, x):
        """
        Get BERT's embedding for text tokens
        
        x (Text tokens) => (batch_size, 1)
        """
        
        if self.embedding_type == "BERT": 
            return self._bert_embedding(x)
        else:
            return self._default_embedding(x)
        
        
    def _bert_embedding(self, x, output_layer=11):

        # Format as torch Tensor
        x = torch.as_tensor(x.numpy())
        x = x.type(torch.LongTensor).to('cuda')
        
        # BERT's embedding
        with torch.no_grad():
            embedding , _ = self.bert_embedding(x)

        # Revert back to tf.Tensor
        x = embedding[output_layer].cpu().numpy()
        x = tf.convert_to_tensor(x)
            
        return x
    
    
    def _default_embedding(self, x):
        return self.default_embedding(x)
    
    
    def apply_strategy(self, x, context_vector, curr_iter=0):
        """
        context_vector : image's vector
        x              : rnn input (word embedding)
        strategy       : 
        curr_iter      : current iteration number
        """
        
        if self.combine_strategy == "inject_init":
            initial_state = tf.squeeze(context_vector) if curr_iter == 1 else None
            output, state = self.rnn_model(x, initial_state=initial_state)  
            
        elif self.combine_strategy == "inject_pre":
            x = context_vector if curr_iter == 1 else x
            output, state = self.rnn_model(x)  
            
        elif self.combine_strategy == "inject_par":
            x = self.custom_combine_layer(context_vector, x)
            output, state = self.rnn_model(x)              

        else: # merge (as default)
            output, state = self.rnn_model(x)           
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
        
    
    def call(self, decoder_input, context_vector, iteration):
        """ 
        decoder_input  : last predicted word => (batch_size, 1)
        context_vector : image's vector      => (batch_size, 1, img_context_size)
        """

        # x1 => (batch_size, 1, embedding_dim)
        x1 = self.embedding(decoder_input)
        
        x2, rnn_state = self.apply_strategy(x1, context_vector, iteration)
            
        ## ============================================
        ## TODO: add another attention layer ? 
        ## ============================================
            
        # x3 shape => (batch_size, 1, rnn_units = 32)
        x3 = self.fc1(x2)

        # x4 => (batch_size, rnn_units = 32)
        x4 = tf.reshape(x3, (-1, x3.shape[2]))

        # word_predictions => (batch_size, vocab)
        word_predictions = self.fc2(x4)
        
        return word_predictions, rnn_state

    def reset_state(self, batch_size=None):
        
        if batch_size is not None:
            return tf.zeros((batch_size, self.rnn_units))
        
        return tf.zeros((self.batch_size, self.rnn_units))        


# ## Define model

# In[36]:


PARAMS


# In[37]:


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


# In[38]:


# Default feed forward function


def feed_forward(img_tensor, target):
    
    loss = 0
    
    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state()

    ## decoder_input == last word generated
    decoder_input = tf.expand_dims(target[:, 0], 1)

    # Training model
    with tf.GradientTape() as gradient_tape:
        
        ## Get image context vector
        features = encoder(img_tensor)
        
        for i in range(1, target.shape[1]):

            # Getting image feature / context_vector from encoder -> attention model
            # context_vector => (batch_size, image_context_size)
            context_vector, attention_weights = attention(features, hidden)

            # context_vector => (batch_size, 1, image_context_size)
            context_vector = tf.expand_dims(context_vector, 1)     
            
            # Passing the features through the decoder
            predictions, hidden = decoder(decoder_input, context_vector, iteration=i)
            
            # loss => Tensor("add:0", shape=(), dtype=float32)
            loss += loss_function(target[:, i], predictions)
            
            # Using teacher forcing
            decoder_input = tf.expand_dims(target[:, i], 1)
        
    total_loss = (loss / int(target.shape[1]))
    
    return loss, total_loss, gradient_tape


    """
    predictions => (batch_size, vocab_size)
    decoder_input => tf.Tensor: id=11841, shape=(batch_size, 1), dtype=int32
    """


# In[39]:


# @tf.function


def train_step(img_tensor, target):
    
    loss, total_loss, gradient_tape = feed_forward(img_tensor, target)

    # Apply gradient
    trainable_variables = encoder.trainable_variables +                           decoder.trainable_variables +                           attention.trainable_variables
    
    gradients = gradient_tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


def eval_step(img_tensor, target):
    
    loss, total_loss, _ = feed_forward(img_tensor, target)
    return loss, total_loss


# In[40]:


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
        predicted_ids = tf.expand_dims(predicted_ids, 1)

    # Max index method
    else:
        predicted_ids = tf.expand_dims(tf.argmax(predictions, 1), 1)
    
    # predicted_ids => (batch_size, 1)
    return predicted_ids


# ### Support using image

# In[41]:


MAX_CAPTION_LENGTH = 25
ATTENTION_SHAPE = 10 * 10 # 100 for xception, 64 for Inception


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

# In[42]:


def get_one_hot_indices(support_text):
    
    indices = []
    for i in range(0, len(support_text)):
        context_token = tokenizer.tokenize(support_text[i])
        context_token_id = tokenizer.convert_tokens_to_ids(context_token)
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


# In[43]:


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
    
    attention_plot = np.zeros((batch_size, MAX_CAPTION_LENGTH, ATTENTION_SHAPE))

    # Extract features from main images
    # features => (batch_size, img_feature_size, img_context_size)
    features = get_image_features(images_paths)
    
    if support_text is not None:
        # support_text_vector => (batch_size, vocab_size)
        support_text_vector = get_supporting_text_vector(support_text, decoder.vocab_size) 
    else:
        # set all pplm related variable to 0
        pplm_iteration = 0
        pplm_weight = 0
        pplm_gm_weight = 0

    # initialize the hidden state for decoder
    hidden = decoder.reset_state(batch_size=batch_size)
    
    # initialize start token for decoder input
    start_token = tokenizer.convert_tokens_to_ids(['start'])
    decoder_input = tf.tile(tf.expand_dims(start_token, 1), [batch_size, 1])
    
    # initialize result container
    result = [[]] * batch_size
    

    for i in range(MAX_CAPTION_LENGTH):
        
        # Getting image feature / context_vector from attention model
        # context_vector => (batch_size, image_context_size)
        context_vector, attention_weights = attention(features, hidden)

        # context_vector => (batch_size, 1, image_context_size)
        context_vector = tf.expand_dims(context_vector, 1)     

        # Passing the features through the decoder
        predictions, hidden = decoder(decoder_input, context_vector, iteration=i)
        ori_prediction = predictions
        
        # ======================== PPLM section ========================
        curr_pertubation = tf.Variable(tf.zeros(decoder_input.shape), name="curr_pertubation", trainable=True)
        
        for j in range(0, pplm_iteration):
            
            with tf.GradientTape() as pplm_tape: 
                
                hidden += curr_pertubation
                
                context_vector, attention_weights = attention(features, hidden)
                context_vector = tf.expand_dims(context_vector, 1)  
                
                predictions, hidden = decoder(decoder_input, context_vector, iteration=i)
                pplm_loss = pplm_loss_function(support_text_vector, predictions, pplm_weight=pplm_weight)

                
            """
            most impactfull layer to train = last dense layer
            """
            
#             trainable_variables = [curr_pertubation] + decoder.trainable_variables
            trainable_variables = [curr_pertubation]
            gradients = pplm_tape.gradient(pplm_loss, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))


            predictions, hidden = decoder(decoder_input, context_vector, iteration=i)
        

        # fuse final pplm_prediction and original prediction
        fused_predictions = (predictions ** pplm_gm_weight) * (ori_prediction ** (1 - pplm_gm_weight)) 
        
        # predicted_ids => (batch_size, 1)
        predicted_ids = choose_predicted_id(fused_predictions, strategy=choose_word_strategy)
        
        # revert back id to token, and append into result
        predicted_tokens = [tokenizer.convert_ids_to_tokens(x) for x in predicted_ids.numpy()]
        result = np.hstack([result, predicted_tokens])

        # assign attention weights to respective generated word
        # attention_plot => (batch_size, max_caption_len, feature_size)
        attention_plot[:, i] = tf.squeeze(attention_weights)

        # use last generated word as next decoder input
        # decoder_input => (batch_size, 1)
        decoder_input = predicted_ids
        
        
    # slice attention to match result len
    attention_plot = attention_plot[:, :len(result[0]), :]
    
    return result, attention_plot


# In[44]:


from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy, MeanSquaredError


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
    
    print(loss)
    return loss

    """
    return => (1)
    """


# !git add "Keras.ipynb"
# !git commit -m "add top k vocab"

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
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)


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


EPOCHS = 20

for epoch in range(start_epoch, EPOCHS):
    
    start = time.time()
    total_loss = 0
    batch = 1

    for img_tensor, target in tqdm(train_dataset):
        
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 50 == 0:
            log_message = '{} Epoch {} Batch {} Loss {:.4f}'.format(
                checkpoint_path, epoch + 1, batch, batch_loss.numpy())
            
            print (log_message)   
            log_file.write(str(log_message + "\n"))
                
        batch += 1
        
        # storing the epoch end loss value to plot later
        loss_plot.append(batch_loss)

    ckpt_manager.save()

    print ('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# In[ ]:


plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.show()


# ## Test Train

# In[ ]:


# for (batch, (img_tensor, target)) in tqdm(enumerate(train_dataset)):

#     batch_loss, t_loss = train_step(img_tensor, target)


# ## Test predict

# In[ ]:


# images = all_img_paths[:4]

# result, attention_plot = custom_evaluate(images)
# result


# ## Test PPLM

# In[ ]:


# images = all_img_paths[:4]
# text = [
#     "butter cream cheese blue cheese cottage cheese goats cheese crème fraîche eggs free range eggs margarine milk full-fat milk semi-skimmed milk skimmed milk sour cream yoghurt",
#     "butter cream cheese blue cheese cottage cheese goats cheese crème fraîche eggs free range eggs margarine milk full-fat milk semi-skimmed milk skimmed milk sour cream yoghurt",
#     "butter cream cheese blue cheese cottage cheese goats cheese crème fraîche eggs free range eggs margarine milk full-fat milk semi-skimmed milk skimmed milk sour cream yoghurt",
#     "butter cream cheese blue cheese cottage cheese goats cheese crème fraîche eggs free range eggs margarine milk full-fat milk semi-skimmed milk skimmed milk sour cream yoghurt",
# ]

# result, attention_plot = custom_evaluate(images, support_text=text, pplm_iteration=5, pplm_weight=1)
# result


# ## Frozen

# In[ ]:


# MAX_CAPTION_LENGTH = 25
# ATTENTION_SHAPE = 10 * 10 # 100 for xception, 64 for Inception


# def get_image_features(images_paths):
#     """
#     images_paths => (batch_size, 1)
#     """
    
#     # Extract images features
#     images = [load_image(x)[0] for x in images_paths]
    
#     # x => (batch_size, 299, 299, 3)
#     x = tf.convert_to_tensor(images)
    
#     # x => (batch_size, 10, 10, 2048)
#     x = extractor(x)
    
#     # x  => (batch_size, img_feature_size, 2048)
#     x = tf.reshape(x, (x.shape[0], -1, x.shape[3]))
    
#     # features => (batch_size, img_feature_size, image_context_size)
#     features = encoder(x)
    
#     return features


# def get_supporting_features(images_paths, strategy="mean"):
#     """
#     images_paths => (batch_size, img_count, 1)
#     strategy : strategy to aggregate multiple supporting image ["logsumexp", "mean", "min", "max"]
#     """
    
#     # Extract images features
#     images = [[load_image(x)[0] for x in images_set] for images_set in images_paths]
    
#     # x => (batch_size, img_count, 299, 299, 3)
#     x = tf.convert_to_tensor(images)
    
#     # x => (batch_size, img_count, 10, 10, 2048)
#     x = [extractor(image_set) for image_set in x]
    
#     # features => (batch_size, img_count, img_feature_size, image_context_size)
#     features = encoder(x)
    
#     # features => (batch_size, img_feature_size, image_context_size)
#     if strategy == "logsumexp":
#         features = tf.reduce_logsumexp(features, 1)
#     elif strategy == "max":
#         features = tf.reduce_max(features, 1)
#     elif strategy == "min":
#         features = tf.reduce_min(features, 1)
#     else:
#         features = tf.reduce_mean(features, 1)

#     return features
    

# def custom_evaluate(images_paths,
#                     support_imgs=None, 
#                     support_aggregate_strategy="mean"):
#     """
#     images_paths : list of image_path         => (batch_size, 1)
#     support_imgs : list of list of image_path => (batch_size, image_count, 1)
#     support_aggregate_strategy : how to aggregate support image ["logsumexp", "mean", "min", "max"]
#     """
    
#     batch_size = len(images_paths)
#     attention_plot = np.zeros((batch_size, MAX_CAPTION_LENGTH, ATTENTION_SHAPE))

#     # Extract features from main images
#     # features => (batch_size, img_feature_size, img_context_size)
#     features = get_image_features(images_paths)

    
#     # Extract aggregated features from support images
#     if support_imgs is not None:
#         # supporting_features => (batch_size, img_feature_size, img_context_size)
#         supporting_features = get_supporting_features(support_imgs, strategy=support_aggregate_strategy)

    
#     # initialize the hidden state for decoder
#     hidden = decoder.reset_state(batch_size=batch_size)
    
#     # initialize start token for decoder input
#     start_token = tokenizer.convert_tokens_to_ids(['start'])
#     decoder_input = tf.tile(tf.expand_dims(start_token, 1), [batch_size, 1])
    
#     # initialize result container
#     result = [[]] * batch_size
    

#     for i in range(MAX_CAPTION_LENGTH):
        
#         # Getting image feature / context_vector from attention model
#         # context_vector => (batch_size, image_context_size)
#         context_vector, attention_weights = attention(features, hidden)
        
#         # context_vector => (batch_size, 1, image_context_size)
#         context_vector = tf.expand_dims(context_vector, 1)     
        

#         if support_imgs is not None:
#             # support_context_vector => (batch_size, 1, image_context_size)
#             support_context_vector, support_attention_weights = attention(supporting_features, hidden)
        
#         """
#         HOW TO USE CONTEXT VECTOR ? 
#         """
        

#         # Passing the features through the decoder
#         predictions, hidden = decoder(decoder_input, context_vector, iteration=i)
        
#         """
#         1. ADJUST CONTEXT VECTOR, (ORIGINAL + SOMETHING(SUPPORT VECTOR))
        
#         OR
        
#         ADJUST HIDDEN STATE USED TO GENERATE TEXT ???
        
#         """
        
        
#         """
#         ## TODO : apply PPLM here
#         ## check loss (prediction - context vector of supporting images)
#         ## apply gradient : hidden_state += diffrence(pred, supporting img vectors) (after n-iteration)
#         ## re compute predictions
        
#         ## not the right loss (?) use PPLM ??
#         # loss => Tensor("add:0", shape=(), dtype=float32)
#         # loss += loss_function(target[:, i], predictions)
#         # use supporting_features here
#         ## =========================================
#         """
        

        
#         # predicted_ids => (batch_size, 1)
#         predicted_ids = choose_predicted_id(predictions, strategy="sampling")
        
#         # revert back id to token, and append into result
#         predicted_tokens = [tokenizer.convert_ids_to_tokens(x) for x in predicted_ids]
#         result = np.hstack([result, predicted_tokens])

#         # assign attention weights to respective generated word
#         # attention_plot => (batch_size, max_caption_len, feature_size)
#         attention_plot[:, i] = tf.squeeze(attention_weights)

#         # use last generated word as next decoder input
#         # decoder_input => (batch_size, 1)
#         decoder_input = predicted_ids
        

#     # slice attention to match result len
#     attention_plot = attention_plot[:, :len(result[0]), :]
    
#     return result, attention_plot
