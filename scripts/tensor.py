import numpy as np
import pandas as pd
import dtaidistance
from data_class import DAL

dataset = "amie-kinect-data.hdf"


def get_people_exercises_sensors():
    dal = DAL(dataset)
    overview = dal.overview()
    people = overview["person"]
    exercises = overview["exercise"]
    skeletons = overview["df_key"]
    skeleton0 = dal.get(skeletons[0])
    sensors = skeleton0.columns
    return people, exercises, sensors


def build_tensor(max_slices=None, slice_size=None):
    '''
    builds distance tensor from the dataset located at ./data/dataset (should be the AMIE dataset)
    :param max_slices: limits amount of frontal slices in tensor.
    :param slice_size: determines the size (n) of the symmetrical n x n frontal slices.
    :return: a numpy array containing the distance tensor tensor
    '''
    dal = DAL(dataset)
    overview = dal.overview()
    skeletons = overview["df_key"]
    
    # get all sensors
    skeleton0 = dal.get(skeletons[0])
    sensor_names = skeleton0.columns
    
    # limit dimensions of tensor
    if max_slices is not None:
        k = max_slices
    else:
        k = len(sensor_names) # 75 sensors for AMIE dataset
    if slice_size is not None:
        n = slice_size
    else:
        n = len(skeletons) # 186 skeletons for AMIE dataset

    # preload sensor data from skeletons
    sensors = []
    for i in range(n):
        sensors.append(dal.get(skeletons[i]))

    tensor = []
    # build a slice for each sensor
    for sensor in sensor_names:
        matrix = np.zeros((n,n))
        print(f"processing {sensor}: {np.where(sensor_names.values == sensor)[0][0] + 1}/{max_slices}")

        for i in range(n):
            # get sensor data from skeleton[i]
            time_series1 = sensors[i].loc[:, sensor].values

            for j in range(i+1, n):
                # get sensor data from skeleton[j]
                time_series2 = sensors[j].loc[:, sensor].values

                # calculate DTW distance (using pure c compiled function for speed:) )
                distance = dtaidistance.dtw.distance(time_series1, time_series2, use_c=True)
                # symmetrical slice
                matrix[i, j] = distance
                matrix[j, i] = distance
            print("<" + (2*(sum(range(n)) - sum(range(n-i)))//(n))*"#" + (2*(sum(range(n-i)))//(n))*"-" + ">", end='\r')

        tensor.append(matrix)

        # limited amount of slices to include
        if len(tensor) >= k:
            break
    
    return np.array(tensor)


def save_overview():
    dal = DAL(dataset)
    overview = dal.overview()
    df = pd.DataFrame(overview)
    df.to_csv("overview.csv")