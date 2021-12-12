# Main window
from plugins.ImportSubjects import Import_Subjects
from plugins.SIFT import SIFT_Implementation
from plugins.SplitData import Split_Data
from tqdm import tqdm
import concurrent.futures
import numpy as np
import pickle

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

    # a_file = open("data.pkl", "wb")
    # pickle.dump(keypoints, a_file)
    # a_file.close()

    a_file = open("data.pkl", "rb")
    keypoints = pickle.load(a_file)
    a_file.close()
    
    # ------------------------ PREPARE DATA FOR TRAINING ------------------------
    training_ratio = .8
    test_ratio = .2
    train_data, train_label, test_data, test_label = Split_Data(subjects['data'], keypoints, training_ratio, test_ratio)
    pos_train = np.vstack((train_data['x'], train_data['y'])).T.astype(int)

    # import numpy as np
    # import nibabel as nib

    # converted_array = numpy.array(normal_array, dtype=numpy.float32) # You need to replace normal array by yours
    # affine = numpy.eye(4)
    # nifti_file = nibabel.Nifti1Image(converted_array, affine)

    # nibabel.save(nifti_file, path_to_save) # Here you put the path + the extionsion 'nii' or 'nii.gz'