#! /usr/bin/env python3

import numpy as np
import pandas as pd
import os
import subprocess
from PyQt5 import QtWidgets
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from commons.SubjectModel import SubjectModel
from commons.ScanModel import ScanModel

DEBUG = False

class ImportSubjects():
    """
        Import subjects from folder

        Parameters
        -----------
            path: string 
                The path to get subjects informations.

        Returns
        -----------    
            subjects: Dict
                Dict of all subjects and their scans info
            nb_images: int
    """
    def __init__(self, subject_path, **kwargs):
        super().__init__(**kwargs)
        if DEBUG: print('ImportSubject.__init__')
        self._subject_path = subject_path
        self.subjects = defaultdict()
        self.nb_images = 0
        self.compute(self._subject_path)

    def compute(self, path):
        """
            Import subjects from folder

            Parameters
            -----------
                path: string 
                    The path to get subjects informations.

            Returns
            -----------    
                subjects: List
                    List of all subjects and their scans info
        """

        if DEBUG: print('ImportSubject.compute')         

        if path is not None:
            try :
                subjects_folder = os.listdir(path)
                subjects = self.subjects
                
                # Set progress bar
                total_event = len(subjects_folder)
                desc = 'Uploading data...'
                pbar = tqdm(total=total_event, desc=desc)

                for subject in subjects_folder:
                    subject_model = SubjectModel()
                    filename = path + "/" + subject + "/" + subject + ".txt"
                    try:
                        f = open(filename, "r")
                    except Exception as e:
                        err_message = "ERROR in Import: {} does not exist. {}".format(filename, e)
                        print(err_message)
                        continue
                    
                    index_scan = ''
                    for line in f:
                        # Get informations from subjects
                        if line.startswith("SESSION ID:"):
                            sid = line.split(':')
                            if sid[1].strip() == '':
                                subject_model.sid = subject
                            else:
                                subject_model.sid = sid[1].strip()

                        if line.startswith("AGE:"):
                            age = line.split(':')
                            if not age[1].strip() == '':
                                subject_model.age = int(age[1].strip())

                        if line.startswith("M/F:"):
                            gender = line.split(':')
                            if not gender[1].strip() == '':
                                subject_model.gender = gender[1].strip()

                        if line.startswith("HAND:"):
                            hand = line.split(':')
                            if not hand[1].strip() == '':
                                subject_model.hand = hand[1].strip()

                        if line.startswith("EDUC:"):
                            educ = line.split(':')
                            if not educ[1].strip() == '':
                                subject_model.educ = int(educ[1].strip())

                        if line.startswith("SES:"):
                            ses = line.split(':')
                            if not ses[1].strip() == '':
                                subject_model.ses = int(ses[1].strip())

                        if line.startswith("CDR:"):
                            cdr = line.split(':')
                            if not cdr[1].strip() == '':
                                subject_model.cdr = float(cdr[1].strip())

                        if line.startswith("MMSE:"):
                            mmse = line.split(':')
                            if not mmse[1].strip() == '':
                                subject_model.mmse = int(mmse[1].strip())

                        if line.startswith("eTIV:"):
                            e_tiv = line.split(':')
                            if not e_tiv[1].strip() == '':
                                subject_model.e_tiv = float(e_tiv[1].strip())

                        if line.startswith("ASF:"):
                            asf = line.split(':')
                            if not asf[1].strip() == '':
                                subject_model.asf = float(asf[1].strip())

                        if line.startswith("nWBV:"):
                            n_wbv = line.split(':')
                            if not n_wbv[1].strip() == '':
                                subject_model.n_wbv = float(n_wbv[1].strip())
                        
                        # Get informations from scans
                        if line.startswith("SCAN NUMBER:"):
                            scan_number = line.split(':')
                            scan_model = ScanModel()
                            if not scan_number[1].strip() == '':
                                index_scan = scan_number[1].strip()
                                subject_model.scans[index_scan] = scan_model
                                subject_model.scans[index_scan].scan_number = scan_number[1].strip()

                        if line.startswith("TYPE:"):
                            type = line.split(':')
                            if not type[1].strip() == '':
                                subject_model.scans[index_scan].scan_type = type[1].strip()

                        if line.startswith("Vox res (mm):"):
                            vox_res = line.split(':')
                            if not vox_res[1].strip() == '':
                                subject_model.scans[index_scan].vox_res = vox_res[1].strip()

                        if line.startswith("Rect. Fov:"):
                            rect_fov = line.split(':')
                            if not rect_fov[1].strip() == '':
                                subject_model.scans[index_scan].rect_fov = rect_fov[1].strip()

                        if line.startswith("Orientation:"):
                            orientation = line.split(':')
                            if not orientation[1].strip() == '':
                                subject_model.scans[index_scan].orientation = orientation[1].strip()

                        if line.startswith("TR (ms):"):
                            tr = line.split(':')
                            if not tr[1].strip() == '':
                                subject_model.scans[index_scan].tr = float(tr[1].strip())

                        if line.startswith("TE (ms):"):
                            te = line.split(':')
                            if not te[1].strip() == '':
                                subject_model.scans[index_scan].te = float(te[1].strip())

                        if line.startswith("TI (ms):"):
                            ti = line.split(':')
                            if not ti[1].strip() == '':
                                subject_model.scans[index_scan].ti = float(ti[1].strip())

                        if line.startswith("Flip:"):
                            flip = line.split(':')
                            if not flip[1].strip() == '':
                                subject_model.scans[index_scan].flip = int(flip[1].strip())

                    f.close()

                    # Import images
                    imgs_folder_path = path + "/" + subject + "/RAW"
                    imgs_folder = os.listdir(imgs_folder_path)
                    images = [img for img in imgs_folder if '.gif' in img]
                    if not len(images) == len(subject_model.scans):
                        err_message = "Warning ImportSubjects {}: there is less images than scans done".format(subject)
                        print(err_message)
                    for scan in subject_model.scans:
                        self.nb_images += 1
                        image = [img for img in images if scan in img]
                        img_folder_path = imgs_folder_path + '/' + image[0]
                        subject_model.scans[scan].image = Image.open(img_folder_path)
                        subject_model.scans[scan].pixels = subject_model.scans[scan].img2pix(subject_model.scans[scan].image)
                    subjects[subject] = subject_model

                    pbar.update(1)
                pbar.close
                return {
                    'subjects': subjects,
                    'nb_images': self.nb_images
                }

            except Exception as e:
                err_message = "ERROR: {}".format(e)
                print(err_message)
                return {
                    'subjects': '',
                    'nb_images': ''
                }                 
        else:
            err_message = "ERROR: path not initialized"
            print(err_message)    

        return {
            'subjects': '',
            'nb_images': ''
        }          
