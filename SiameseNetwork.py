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
    margin = 1
    a = T.multiply( y_true , T.pow(T.maximum(0.0,margin - Dw), 2 ))
    b = T.multiply( T.subtract ( T.constant([1.0],) , y_true ), T.pow(Dw,2) )
    return T.divide(T.add(a,b) , 2 )

def siamese_euclidean_distance(pair):
    feature1,feature2 = pair[0],pair[1]
    diff = T.subtract(T.convert_to_tensor(feature1),T.convert_to_tensor(feature2))
    sqr  = T.pow(diff,2)
    dist = T.pow(sqr,0.5)
    return dist

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
        else:
            dataset[curr_index].append(features[i])
            #print(dataset[curr_index])
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
            visited = [1 for w in writers]
            while True:
                inc = random.randrange(1, 1556)
                other_writer =  (writer + inc )%1556
                visited[other_writer] = 0
                if sum(visited) > 1:   # If not tried every other writer
                    if len(data[other_writer]) > i :
                        pos_feature1, pos_feature2 = data[writer][i], data[writer][i+1]
                        neg_feature1, neg_feature2 = data[writer][i], data[other_writer][i]

                        pairs += [[pos_feature1, pos_feature2]]
                        pairs += [[neg_feature1, neg_feature2]]

                        pair_index += [[writer,writer]]
                        pair_index += [[writer,other_writer]]
                        labels += [1, 0]
                        break
                else:
                    break;
    for writer in range(len(writers)):
        for i in range(len(data[writer])-2):
            visited = [1 for w in writers]
            while True:
                inc = random.randrange(1, 1556)
                other_writer =  (writer + inc )%1556
                visited[other_writer] = 0
                if sum(visited) > 1:   # If not tried every other writer
                    if len(data[other_writer]) > i :
                        pos_feature1, pos_feature2 = data[writer][i], data[writer][i+2]
                        neg_feature1, neg_feature2 = data[writer][i], data[other_writer][i]

                        pairs += [[pos_feature1, pos_feature2]]
                        pairs += [[neg_feature1, neg_feature2]]

                        pair_index += [[writer,writer]]
                        pair_index += [[writer,other_writer]]
                        labels += [1, 0]
                        break
                else:
                    break;
    summary = pd.DataFrame( np.column_stack((np.array(pair_index),np.array(labels))) , columns = ['writer1' ,'writer2','label'])
    return np.array(pairs), np.array(labels)
def check_model_predictions(pairs,label,model,pair_index):
    n = 150
    predictions = model.predict([pairs[:n,0,:],pairs[:n,1,:]])
    predictions = np.round(predictions)
    print(np.array(pair_index))
    a = np.column_stack(pair_index[0:n,0:],predictions)
    table = np.column_stack(a,label[:n])
    for i in range(n):
        if predictions[p] == label[p]:
            count += 1
    table_df = pd.DataFrame(table, columns=['writer 1','writer 2','pred','known'])
    print(table_df)
    print("Custom accuracy : " , count/n)
def NeuralNet(pairs,num_epochs):
    # Model
    input_shape = (feature_size,)
    left_input  = Input(shape = input_shape)
    right_input = Input(shape = input_shape)

    L2_layer = Lambda(siamese_euclidean_distance)
    L2_distance = L2_layer([left_input,right_input])

    prediction = Dense(1,activation='sigmoid')(L2_distance)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    # train
    adam = Adam(lr=0.01)
    siamese_net.compile(loss=contrastive_loss , optimizer=adam, metrics = ['accuracy'])
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
model,history =  NeuralNet(pairs,num_epochs = 50)

plot_model(model, to_file='model.png' ,show_shapes = True)

plt.figure()
history['loss'].plot()
plt.show()
