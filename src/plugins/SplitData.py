import numpy as np

DEBUG = False

def Split_Data(data, labels, training_r, test_r):
    # Get list of images for the training and test set
    image_list = list(data.keys())
    nb_train = np.floor(len(image_list)*training_r).astype(int)
    nb_test = np.floor(len(image_list)*test_r).astype(int)
    train_img = image_list[0:nb_train]
    test_img = image_list[nb_train:nb_train+nb_test]

    # Train set
    train_data = {key: data[key] for key in train_img}
    labels = labels.fillna(0)
    train_label = {key: labels.loc[labels.ID == key[0:13], 'CDR'].values[0] for key in train_img}
    train_pos = [np.vstack((data[key]['x'], data[key]['y'])).T.astype(int) for key in train_img]
    train_pos = np.vstack((train_pos))

    # Test set
    test_data = {key: data[key] for key in test_img}
    test_label = {key: labels.loc[labels.ID == key[0:13], 'CDR'].values[0] for key in test_img}
    test_pos = [np.vstack((data[key]['x'], data[key]['y'])).T.astype(int) for key in test_img]
    test_pos = np.vstack((test_pos))

    return train_data, train_label, train_pos, test_data, test_label, test_pos