

def training_MLPs(test_data, clusters, i):
    # mlp for binary classification
    # determine the number of input features
    test_data_clusters = test_data[test_data['kmeans_label'] == i].loc[:, ~test_data.columns.isin(['image_no', 'keypoint_no', 'image_label', 'kmeans_label'])]
    #self.show_images(test_data_clusters['x'].values, test_data_clusters['y'].values)
    test_data_clusters = test_data_clusters.to_numpy()
    label_test_clusters = test_data[test_data['kmeans_label'] == i]['image_label'].values
    n_features = test_data_clusters.shape[1]
    
    