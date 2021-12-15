# Main window
from plugins.ImportSubjects import Import_Subjects
from plugins.SIFT import SIFT_Implementation
from plugins.SplitData import Split_Data
from plugins.Training import training_MLPs
from plugins.Plotting import plot_confusion_matrix, plot_roc
from tqdm import tqdm
import concurrent.futures
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from PIL import Image

DEBUG = True

def worker(image, key, size):
    m = SIFT_Implementation(image, key, size).keypoints
    return m

if __name__ == '__main__':   

    # -----------------  SET THE PATH TO THE DATA HERE -------------------------
    DATA_PATH = "C:/Users/sampo/Python/PycharmProjects/SYS818_Project/Data/subjects"
    LABEL_PATH = "C:/Users/sampo/Python/PycharmProjects/SYS818_Project/Data/oasis_cross-sectional.csv"
    
    # ------------------------- IMPORT SUBJECT DATA ----------------------------
    subjects = Import_Subjects(DATA_PATH, LABEL_PATH)
    print("\nImages found: {}".format(str(len(subjects['data']))))

    # ------------------------ FIND KEYPOINTS PER IMAGE ------------------------
    #Set progress bar
    total_event = len(list(subjects['data'].keys()))
    desc = 'Extracting keypoints...'

    keypoints = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        with tqdm(total=total_event, desc=desc) as progress:
            futureList = []
            for key in list(subjects['data'].keys()):
                future = executor.submit(worker,subjects['data'][key], key, (256,256))
                future.add_done_callback(lambda p: progress.update())
                futureList.append(future)

            for f in futureList:
                df = f.result()
                key = df['image_no'][0].astype(str)
                keypoints[key] = f.result()
    
    # ------------------------ PREPARE DATA FOR TRAINING -----------------------
    training_ratio = .8
    test_ratio = .2
    number_epochs = 15
    batch_size = 30
    if DEBUG:
        print('\nPreparing data...')

    train_data, train_label, train_pos, test_data, test_label, test_pos = Split_Data(keypoints, subjects['labels'], training_ratio, test_ratio)
    
    # ------------------------------- TRAINING ---------------------------------
    # Clusters
    if DEBUG:
        print('\nComputing clusters...')

    kmeans = KMeans(n_clusters=13, init='random', n_init=10, max_iter=300, tol=1e-4, verbose=0, random_state=0, algorithm="auto").fit(train_pos)
    train_data_k_labels = kmeans.labels_
    idx = 0

    # Adding clusters value
    for img in list(train_data.keys()):
        nb_key = len(train_data[img])
        first_idx = idx
        end_idx = first_idx + nb_key
        label_use = 0 if train_label[img] == 0 else 1
        label_col = np.full((nb_key, 1), label_use)
        train_data[img]['image_label'] = label_col
        train_data[img]['kmeans_label'] = train_data_k_labels[first_idx:end_idx]
        idx = end_idx

    if DEBUG:
        print('\nStart training...')
    clusters = {}
    for i in range(kmeans.n_clusters):
        hist = training_MLPs(train_data, number_epochs, batch_size, i)
        key_name = 'clusters_' + str(i)
        clusters[key_name] = hist

    weights_acc = np.array([clusters[list(clusters.keys())[i]].history.get('accuracy')[-1] for i in range(kmeans.n_clusters)])
    weights_acc = weights_acc ** 2
    weights_acc = weights_acc / np.max(weights_acc)
    weights_acc = np.reshape(weights_acc,(len(weights_acc),1))

    # -------------------------------- TEST ------------------------------------
    # Clusters
    if DEBUG:
        print('\nPredicting clusters...')
    test_data_k_labels = kmeans.predict(test_pos)
    idx = 0
    
    # Adding clusters value
    for img in list(test_data.keys()):
        nb_key = len(test_data[img])
        first_idx = idx
        end_idx = first_idx + nb_key
        label_use = 0 if test_label[img] == 0 else 1
        label_col = np.full((nb_key, 1), label_use)
        test_data[img]['image_label'] = label_col
        test_data[img]['kmeans_label'] = test_data_k_labels[first_idx:end_idx]
        idx = end_idx

    if DEBUG:
        print('\nStart testing...')

    clusters_test_labels = {}
    clusters_test = {}
    for i in range(kmeans.n_clusters):
        # mlp for binary classification
        # determine the number of input features
        test_data_clusters = [test_data[key][test_data[key]['kmeans_label'] == i].loc[:, ~test_data[key].columns.isin(['image_no', 
                                                                                                                       'keypoint_no', 
                                                                                                                       'x', 
                                                                                                                       'y', 
                                                                                                                       'sigma', 
                                                                                                                       'magnitude', 
                                                                                                                       'orientation',
                                                                                                                       'image_label', 
                                                                                                                       'kmeans_label'])] for key in list(test_data.keys())]
        test_data_clusters = pd.concat(test_data_clusters).to_numpy().astype(float)
        label_test_clusters = [test_data[key][test_data[key]['kmeans_label'] == i]['image_label'].values for key in list(test_data.keys())]
        label_test_clusters = np.reshape(np.hstack((label_test_clusters)), (len(test_data_clusters),1))
        n_features = test_data_clusters.shape[1]
        key_name = 'clusters_' + str(i)
        clusters_test_labels[key_name] = label_test_clusters
        clusters_test[key_name] = clusters[key_name].model.predict(test_data_clusters, verbose=1)
        # cm = (confusion_matrix(np.argmax(clusters_test[key_name],axis=1), np.argmax(clusters_test[key_name],axis=1)))
        # self.plot_confusion_matrix(cm, None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)
    text_label='ROC CNN'
    plot_roc(clusters_test_labels, clusters_test, text_label)
    index = {}
    for key in list(test_data.keys()):
        test_data[key]['predicted_label'] = 0
        index[key] = test_data[key].index
    for i in range(kmeans.n_clusters):
        condition = [test_data[key]['kmeans_label'] == i for key in list(test_data.keys())]
        kmeans_label_indices = index[condition]
        kmeans_indices_list = kmeans_label_indices.tolist()
        test_data.loc[kmeans_indices_list,'predicted_label'] = clusters_test[list(clusters_test.keys())[i]]
    
    image_cluster = np.zeros((kmeans.n_clusters,1))
    vec_pred_label = []
    for i in range(np.max(test_data['image_no'].to_numpy())):
        image_all_clusters = test_data[test_data['image_no'] == i]
        for j in range(kmeans.n_clusters):
            image_cluster[j] = np.sum(image_all_clusters[image_all_clusters['kmeans_label'] == j]['predicted_label'].values)
        vec_pred_label.append(image_cluster)

    new_label = np.zeros((len(vec_pred_label),1))
    for i in range(len(vec_pred_label)):
        new_label[i] = np.dot(np.reshape(vec_pred_label[i],(1,kmeans.n_clusters)),weights_acc)


    #model_clustering = Models_Clustering(kmeans)
    # import numpy as np
    # import nibabel as nib

    # converted_array = numpy.array(normal_array, dtype=numpy.float32) # You need to replace normal array by yours
    # affine = numpy.eye(4)
    # nifti_file = nibabel.Nifti1Image(converted_array, affine)

    # nibabel.save(nifti_file, path_to_save) # Here you put the path + the extionsion 'nii' or 'nii.gz'