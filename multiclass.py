# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 10:20:12 2021

@author: fanyak
@title: multi-class classification
@url:https://colab.research.google.com/drive/1Aw98z6xEIHFj6O7ZgMIXnhSPTTYs9c1O#scrollTo=eSSuci_6nCEG
"""

import tensorflow as tf;
import os;
import re;
import string;
import matplotlib.pyplot as plt;
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization;
import numpy as np;

url = "http://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz";
# dataset = tf.keras.utils.get_file("stack_overflow_16k.tar.gz", 
#                                   url, untar=True, 
#                                   cache_dir=".", cache_subdir="");
dataset = '.\\DATA\\stack_overflow_16k';
dataset_dir = os.path.join(os.path.dirname(dataset), 'stack_overflow_16k');
os.listdir(dataset_dir);
train_dir = os.path.join(dataset_dir, 'train');
os.listdir(train_dir);
test_dir = os.path.join(dataset_dir, 'test');


# with open(os.path.join(train_wd,os.listdir(train_wd)[1]), 'r') as fs:
#     print(fs.read());

batch_size = 32;
seed = 42;

# CREATE A DATABASE from the train directory using the existing folders ('python','javascript'..)
# as classes for the classification
# use a validation 80:20 split
train_raw_ds = tf.keras.preprocessing.text_dataset_from_directory(train_dir, batch_size=batch_size,
                                                                  seed=seed,
                                                                  validation_split = 0.2,
                                                                  subset='training');
#print(train_raw_ds.class_names);

val_raw_ds = tf.keras.preprocessing.text_dataset_from_directory(train_dir, 
                                                                batch_size = batch_size, 
                                                                seed = seed,
                                                                validation_split=0.2,
                                                                subset='validation');
test_raw_ds = tf.keras.preprocessing.text_dataset_from_directory(test_dir, 
                                                                 batch_size = batch_size);


########## NORMALIZE, TOKENIZE, VECTORIZE THE TEXTS ############
def standardize_data(data):
    lowercase = tf.strings.lower(data);
    return tf.strings.regex_replace(lowercase,'[%s]' % re.escape(string.punctuation), " ");


max_features = 10000;#vocabulary length
sequence_length = 250;

vectorize_layer = TextVectorization(
    standardize=standardize_data,
    max_tokens = max_features, #tokenize
    output_mode = 'int', # vectorize,
    output_sequence_length = sequence_length);

#get only the texts from the iterable tuples
fit_texts = train_raw_ds.map(lambda x,y: x);
fit_labels = train_raw_ds.map(lambda x,y: y);

############ FIT THE TEXTVECTORIZATION ###########
#use the texts to FIT the TextVectorization
vectorize_layer.adapt(fit_texts);

def vectorize_text(text, label): # a Tuple as input since that is the form of the dataset
    text = tf.expand_dims(text, -1);
    return vectorize_layer(text), label;

# try out the TextVectorization    
batch = next(iter(train_raw_ds)); #batch of size 32
# each batch is a TUPLE of 2 lists (texts<32>, labels<32>)
texts, labels = batch;
#texts.numpy() # list of length = batch_size = 32;
#labels.numpy() # list of length = batch_size = 32;

train_ds = train_raw_ds.map(vectorize_text);
val_ds = val_raw_ds.map(vectorize_text);
test_ds = test_raw_ds.map(vectorize_text);

######### DATASET PERFMORMANCE #############
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size = AUTOTUNE);
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE);
test_ds = test_ds.cache().prefetch(buffer_size = AUTOTUNE);


##### CREATE THE MODEL ##########
embedding_dim = 16;
category_num = len(train_raw_ds.class_names);

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_features+1, embedding_dim),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(category_num),
    ]);
model.summary();

###### TRAIN THE MODEL ########

##### METHOD 1: Manually train

# def train_step(x,y_true):
#     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3);
#     with tf.GradientTape() as tape:        
#         y_pred = model(x)
#         loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
#     grads = tape.gradient(loss, model.trainable_variables);
#     optimizer.apply_gradients(zip(grads, model.trainable_variables));
#     return loss;

# epochs = 5;
# history = [];
# vhistory = [];

# for i in range(epochs):
#     iterator = iter(train_ds);
#     viterator = iter(val_ds);
#     epochHistory = [];
#     for train_batch in range(tf.data.experimental.cardinality(train_ds).numpy()):
#         x,y=iterator.get_next()#next(iterator);
#         loss = train_step(x,y);
#         epochHistory.extend(loss.numpy());
#     history.append(np.array(epochHistory).mean());

# #check validation
#     epochVHistory = [];
#     for val_batch in range(tf.data.experimental.cardinality(val_ds).numpy()):
#         vx, vy = viterator.get_next()#next(viterator);
#         val_pred = model(vx);
#         vloss = tf.keras.losses.sparse_categorical_crossentropy(vy, val_pred);
#         epochVHistory.extend(vloss.numpy());
#     vhistory.append(np.array(epochVHistory).mean())

# plt.plot(np.arange(epochs),history, color="blue");
# plt.plot(np.arange(epochs),vhistory, color="red");
# plt.legend(['train', 'validation'])
# plt.show();


#########  METHOD 2: Use TF
model.compile(optimizer='adam', 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']);

epochs = 5;
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs);
#evaluate the model
loss, accuracy = model.evaluate(test_ds); 

history_dict = history.history;
acc = history_dict['accuracy'];
val_acc = history_dict['val_accuracy'];
loss = history_dict['loss'];
val_loss = history_dict['val_loss'];

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show();

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()