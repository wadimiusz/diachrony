import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Dot, Lambda
from keras import backend as K
from keras.layers.embeddings import Embedding
from keras.datasets import imdb
from keras.preprocessing.sequence import skipgrams
import glob, random

random.seed(1234)
np.random.seed(1234)

batch_size = 10000
n_epochs = 5

vocabulary = open("vocabulary.txt.gz", "r").read().split("\n")
vocab_size = len(vocabulary)+2 # OOV_CHAR is 1 and 0 index is excluded

m, n, k = 100, 100, 300 # Tensor Dimensions as defined in the paper
t_dim_a, t_dim_b = 100, 100
final_dim = 100

#Time Model
time = Input((1,), name="time")
time_a = Dense(t_dim_a, activation="tanh")(time)
time_b = Dense(t_dim_b, activation="tanh", name = "")(time_a)

#Word/Context model
x = Input((1,))
embed_x = Embedding(vocab_size, k)(x)
transform_x = Dense(m*n)(embed_x)
reshape_x = Reshape((m,n))(transform_x)

shared_model = Model(inputs=x, outputs = reshape_x)
shared_dense = Dense(final_dim)

word = Input((1,), name = "word")
context = Input((1,), name = "context")

trans_w = shared_model(word)
trans_c = shared_model(context)

multiply_layer = Lambda(lambda x: K.batch_dot(x[0], x[1])) #A custom layer for matrix multiplication

mat_vec_word = multiply_layer([trans_w, time_b])
mat_vec_context = multiply_layer([trans_c, time_b])

linear_prod_w = shared_dense(mat_vec_word)
linear_prod_c = shared_dense(mat_vec_context)

dot_prod = Dot(axes = 1)([linear_prod_w, linear_prod_c])
final_output = Dense(1, activation="sigmoid")(dot_prod)

final_model = Model(inputs=[word, context, time], outputs=[final_output])

print(shared_model.summary())

print(final_model.summary())

final_model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#final_model.fit([w_train, c_train, t_train], y_train, epochs = 10)
word_time_model = Model(inputs = [word, time], outputs =linear_prod_w)

for epoch in range(1, n_epochs):
    loss, accuracy = 0.0, 0.0
#    with open("train_data_shuffle.txt", "r") as fw:
    with open("sample_data.txt", "r") as fw:
        w_train, c_train, t_train, y_train = [], [], [], []
        for i, line in enumerate(fw):
            w, c, t, y = line.strip().split("\t")
#            print("Lines ", w, c, t, y)
            w_train.append([int(w)])
            c_train.append([int(c)])
            t_train.append([float(t)])
            y_train.append(int(y))

            if i % batch_size == 0:
                w_train = np.array(w_train)
                c_train = np.array(c_train)
                t_train = np.array(t_train)
                y_train = np.array(y_train)
#                print(w_train[:5])
                temp = final_model.train_on_batch([w_train, c_train, t_train], y_train) 
                loss += temp[0]
                accuracy += temp[1]    
                w_train, c_train, t_train, y_train = [], [], [], []
                nr_batch = (i/batch_size)+1
                print('Batch:', nr_batch, 'Loss:', loss/nr_batch, 'Accuracy:', accuracy/nr_batch, sep=" ")

    print('Epoch:', epoch, 'Loss:', loss/nr_batch, 'Accuracy:', accuracy/nr_batch)

final_model.save('final_model.h5')

from keras.models import load_model

##Change here to extract word embeddings
for n in range(10):
    rand_vocab_num = np.random.randint(2,vocab_size)
    rand_time = np.random.random()
    rand_vocab = vocabulary[rand_vocab_num]
    print(n, rand_vocab, rand_time, word_time_model.predict([[rand_vocab_num], [rand_time]])) #This line predicts the word embedding by taking vocab number and random time as the input.




