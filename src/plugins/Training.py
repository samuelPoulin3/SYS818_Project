import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from keras import callbacks

DEBUG = False

def training_MLPs(train_data, number_epochs, batch_size, i):
    model = []

    # mlp for binary classification
    # determine the number of input features
    data_train_clusters = [train_data[key][train_data[key]['kmeans_label'] == i].loc[:, ~train_data[key].columns.isin(['image_no', 
                                                                                                                       'keypoint_no', 
                                                                                                                       'x', 
                                                                                                                       'y', 
                                                                                                                       'sigma', 
                                                                                                                       'magnitude', 
                                                                                                                       'orientation',
                                                                                                                       'image_label', 
                                                                                                                       'kmeans_label'])] for key in list(train_data.keys())]
    data_train_clusters = pd.concat(data_train_clusters).to_numpy().astype(float)
    label_train_clusters = [train_data[key][train_data[key]['kmeans_label'] == i]['image_label'].values for key in list(train_data.keys())]
    label_train_clusters = np.reshape(np.hstack((label_train_clusters)), (len(data_train_clusters),1))
    n_features = data_train_clusters.shape[1]

    # define model
    model = Sequential()
    model.add(Dense(40, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    earlystopping = callbacks.EarlyStopping(monitor ="accuracy", 
                                            mode ="max", patience = 5, 
                                            restore_best_weights = True)

    # fit the model
    hist = model.fit(data_train_clusters, 
                     label_train_clusters, 
                     epochs=number_epochs, 
                     batch_size=batch_size, 
                     verbose=1, 
                     callbacks =[earlystopping])
    
    return hist