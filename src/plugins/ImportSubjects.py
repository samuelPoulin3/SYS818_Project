#! /usr/bin/env python3

import numpy as np
import pandas as pd
import os
from scipy.sparse import csr_matrix
from tqdm import tqdm
from PIL import Image

DEBUG = False

def Import_Subjects(data_path, label_path):
    try:
        # Get labels
        labels = pd.read_csv(label_path)

        # Get subjects in path
        subjects_folder = os.listdir(data_path)

        #Set progress bar
        total_event = len(subjects_folder)
        desc = 'Uploading data...'
        pbar = tqdm(total=total_event, desc=desc)

        data = {}
        for subject in subjects_folder:
            # Loop subjects
            imgs_folder_path = data_path + "/" + subject + "/RAW"
            imgs_folder = os.listdir(imgs_folder_path)
            images = [img for img in imgs_folder if '.gif' in img]
            i = 0

            # Loop scans per subject
            for image in images:
                img_folder_path = imgs_folder_path + '/' + image
                image = Image.open(img_folder_path)
                key_name = subject + '_mpr' + str(i)
                data[key_name] = csr_matrix(np.array(image).flatten()/255)
                i += 1

            pbar.update(1)
        pbar.close
        return {
            'data': data,
            'labels': labels
        }

    # Catch if path not valid
    except Exception as e:
        err_message = "ERROR Import_Subject: {}".format(e)
        print(err_message)
        return {'data': [],
                'labels': []
        }
