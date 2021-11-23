#! /usr/bin/env python3
from plugins import layers
from plugins.ScanShow.ScanShow import ScanShow
import concurrent.futures

DEBUG = False

class Models():
    """
        The model of our algorithm     
    """
    __slots__ = ('steps', 'new_mat', 'keypoints')
    def __init__(self, *args, **kwargs):
        self.steps = []
        self.new_mat = {}
        self.keypoints = [] 

    def add(self, layers):
        self.steps.append(layers)

    def compile(self):
        pass

    def fit(self, data, labels, epochs, batch_size):
        step_number = 1
        futureList = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for step in self.steps:
                step_str = 'step_' + str(step_number)
                if step_number == 1:
                    future = executor.submit(layers.Gauss2D_compile, data, step)
                    self.new_mat[step_str] = future.result()
                # First octave
                elif step_number > 1 and step_number <= 5:
                    futureList.append([executor.submit(layers.Gauss2D_compile, self.new_mat['step_1'], step)])
                elif step_number == 6:
                    self.new_mat['step_2'] = futureList[0][0].result()
                    self.new_mat['step_3'] = futureList[1][0].result()
                    self.new_mat['step_4'] = futureList[2][0].result()
                    self.new_mat['step_5'] = futureList[3][0].result()
                    
                    # self.new_mat[step_str], keypoints = executor.submit(layers.Gauss2D_compile, self.new_mat['step_1'], step)
                    # self.keypoints.append(keypoints)
                else:
                    pass
                    #self.new_mat[step_str] = executor.submit(layers.Gauss2D_compile, data, step)
                previous_step = 'step_' + str(step_number) 
                step_number += 1
