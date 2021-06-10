# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 14:42:21 2020

@author: fanyak
@source: https://colab.research.google.com/drive/1EvnVxZm_FsyTzAoHk9zdIIXswej_Zkbehttps://colab.research.google.com/drive/1EvnVxZm_FsyTzAoHk9zdIIXswej_Zkbe
from youtube tensorflow
"""

import tensorflow as tf;
import json;
from tensorflow.keras.preprocessing.text import Tokenizer;
from tensorflow.keras.preprocessing.sequence import pad_sequences;
import os;
import numpy as np;
import matplotlib.pyplot as plt;

vocab_size= 10000;
embedding_dim = 16;
max_length  = 100;
trunc_type = padding_type = 'post';
oov_tok = "<OOV>";
training_size = 20000;

wd = os.getcwd();
path_to_data = ".\\sarcasm.json";

with open(path_to_data, "r") as fp:
    datastore = json.load(fp);
 
sentences = [];
labels = [];    

for item in datastore:
    sentences.append(item["headline"]);
    labels.append(item["is_sarcastic"]);
    

training_sentences = sentences[0:training_size];
training_labels = labels[0:training_size];
test_sentences = sentences[training_size:];test_labels = labels[training_size:];

print(training_labels)

tokenizer = Tokenizer(num_words=vocab_size, oov_token = oov_tok);
tokenizer.fit_on_texts(training_sentences);

word_index = tokenizer.word_index;

training_sequences = tokenizer.texts_to_sequences(training_sentences);
training_padded = pad_sequences(training_sequences);

testing_sequences = tokenizer.texts_to_sequences(test_sentences);
testing_padded = pad_sequences(testing_sequences);


training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(test_labels)

def build_model():
    model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ]);
    return model;

model = build_model();    
learning_rate = 1e-3;
optimizer = tf.keras.optimizers.Adam(learning_rate)

def train_step(model, optimizer, x, y_true):
    with tf.GradientTape() as tape:
        y_pred = model(x);
        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred);
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables));

#train_step(model, optimizer, training_padded, training_labels.reshape((training_size,1)));
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary();

num_epochs = 30;
history = model.fit(training_padded, 
                    training_labels,
                    epochs=num_epochs, 
                    validation_data=(testing_padded, testing_labels), verbose=2);

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss");

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)


sentence = ["granny starting to fear spiders in the garden might be real", "game of thrones season finale showing this sunday night"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(padded))