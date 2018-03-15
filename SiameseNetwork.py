import numpy as np
np.random.seed(1337) # for reproducibility
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, Input
from keras.optimizers import SGD, RMSprop ,Adam
from keras import backend as K
from keras.callbacks import History
from keras.utils import plot_model
import tensorflow as T
import os,random
import collections
import matplotlib.pyplot as plt
import pandas as pd
n = 1556
feature_size = 50
def contrastive_loss(y_true, Dw):
    margin = 2
    a = T.multiply( y_true , T.square(T.maximum(0.0,margin - Dw)))
    b = T.multiply( (1 - y_true) , T.square(Dw) )
    return T.add(a,b)
def siamese_euclidean_distance(pair):
    feature1,feature2 = pair[0],pair[1]
    print("Euclidean Input Shape : " , feature1.get_shape().as_list())
    diff = T.subtract(feature1,feature2)
    sqr  = T.square(diff)
    print("Squared Shape : " , sqr.get_shape().as_list())
    sum_ = T.reduce_sum(sqr,1,keep_dims=True)
    print("Mean Shape : ",sum_.get_shape().as_list())
    dist = T.sqrt(sum_)
    print("Dist Shape : ",dist.get_shape().as_list())
    return dist

def normalized(v):
    mx = np.max(v)
    mn = np.min(v)
    return np.divide(v,float(mx-mn))
# For images instead of appending features you will append imnages
def read_features():
    features = np.genfromtxt('features.csv',delimiter = ',')
    train_path = 'Images/'
    training_names = os.listdir(train_path)
    dataset = [[] for i in  range(n)]
    writers = []
    curr_index = -1
    i = 0
    for name in training_names:
        writer_name = name[:4]
        #print(writer_name)
        if writer_name not in writers:
            writers.append(writer_name)
            curr_index += 1
        dataset[curr_index].append(normalized(features[i]))
        i += 1
    return np.array(dataset),writers

def create_pairs(data,writers,pair_index):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs =  []
    labels = []
    for writer in range(len(writers)):
        for i in range(len(data[writer])-1):
            inc = random.randrange(1, 1556)
            other_writer =  (writer + inc )%1556
            pos_feature1, pos_feature2 = data[writer][i], data[writer][i+1]
            neg_feature1, neg_feature2 = data[writer][i], data[other_writer][ random.randrange(0,len(data[other_writer])) ]

            pairs += [[pos_feature1, pos_feature2]]
            pairs += [[neg_feature1, neg_feature2]]

            pair_index += [[writer,writer]]
            pair_index += [[writer,other_writer]]
            labels += [0, 1]

    summary = pd.DataFrame( np.column_stack((np.array(pair_index),np.array(labels))) , columns = ['writer1' ,'writer2','label'])
    return np.array(pairs), np.array(labels)

def NeuralNet(pairs,num_epochs):
    # Model
    input_shape = (feature_size,)
    left_input  = Input(shape = input_shape)
    right_input = Input(shape = input_shape)

    L2_layer = Lambda(siamese_euclidean_distance)
    L2_distance = L2_layer([left_input,right_input])
    prediction = L2_distance
    #prediction = Dense(1,activation='sigmoid')(L2_distance)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    # train
    adam = Adam(lr=0.0005)
    siamese_net.compile(loss=contrastive_loss , optimizer=adam, metrics = ['binary_accuracy'])
    history = History()
    summary = siamese_net.fit([pairs[6000:,0,:] , pairs[6000:,1,:] ], labels[6000:], batch_size=128, nb_epoch=num_epochs, callbacks= [history] , validation_data=([pairs[:6000,0,:], pairs[:6000,1,:]], labels[:6000]))
    siamese_net.summary()
    history_df = pd.DataFrame(summary.history)
    history_df.to_csv("history.csv")
    return siamese_net, history_df


data,writers =  read_features()
pair_index = []
pairs , labels =  create_pairs(data,writers,pair_index)
print(np.array(pairs).shape)
model,history =  NeuralNet(pairs,num_epochs = 20)

plot_model(model, to_file='model.png' ,show_shapes = True)

plt.figure(1)
axes = plt.gca()
axes.set_ylim([0,1])
history['acc'].plot()

plt.figure(2)
history['loss'].plot()
plt.show()
