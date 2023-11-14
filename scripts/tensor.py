import numpy as np
import pandas as pd
import dtaidistance
from scripts.data_class import DAL

dataset = "amie-kinect-data.hdf"

# builds distancetensor from the dataset located at ./data/dataset (should be the AMIE dataset)
#  > max_slices: limits amount of frontal slices in tensor.
#  > sensors_per_slice: limits amount of sensors that should be included in 1 slice,
#                       determines the size of a frontal slice.
def build_tensor(max_slices=None, sensors_per_slice=None):
    dal = DAL(dataset)
    overview = dal.overview()
    skeletons = overview["df_key"]
    
    # get all sensors
    skeleton0 = dal.get(skeletons[0])
    sensor_names = skeleton0.columns
    
    # limit amount of sensors to include
    if sensors_per_slice is not None:
        n = sensors_per_slice
    else:
        n = len(skeletons) # 186 different skeletons for AMIE dataset

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
        if max_slices is not None and len(tensor) == max_slices:
            break
    
    return tensor

def same_tensor_test(amount_of_slices, sensors_per_slice):
    path = "saved_tensors/full_tensor.npy"
    big_boi = np.load(path)
    myTensor = build_tensor(amount_of_slices, sensors_per_slice)
    for k in range(amount_of_slices):
        matrix = []
        for i in range(sensors_per_slice):
            row = []
            for j in range(sensors_per_slice):
                assert big_boi[k, j, i] == myTensor[k][j][i]
    print("---Tensor Test Passed---")

def save_overview():
    dal = DAL(dataset)
    overview = dal.overview()
    df = pd.DataFrame(overview)
    df.to_csv("overview.csv")

#tensor = build_tensor(20, 20)
#print(tensor)
same_tensor_test(20, 5)