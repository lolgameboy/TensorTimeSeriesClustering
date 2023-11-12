import numpy as np
import pandas as pd
import dtaidistance
from DataTest import DAL

dataset = "amie-kinect-data.hdf"

def build_tensor(amount_of_slices=None, amount_of_sensors=None):
    dal = DAL(dataset)
    overview = dal.overview()
    skeletons = overview["df_key"]
    
    # get all sensors
    skeleton0 = dal.get(skeletons[0])
    sensor_names = skeleton0.columns
    
    # limit amount of sensors to include
    if amount_of_sensors is not None:
        n = amount_of_sensors
    else:
        n = len(skeletons) # 186 different skeletons for AMIE dataset

    tensor = []
    # build a slice for each sensor
    for sensor in sensor_names[:n]:
        matrix = np.zeros((n,n))
        print(f"at sensor {sensor}: {np.where(sensor_names.values == sensor)[0][0] + 1}/{n}")

        for i in range(n):
            # get sensor data from skeleton[i] (i.e. return data of the sensor column)
            time_series1 = dal.get(skeletons[i]).loc[:, sensor].values

            for j in range(i+1, n):
                # get sensor data from skeleton[j]
                time_series2 = dal.get(skeletons[j]).loc[:, sensor].values

                # calculate DTW distance (using pure c compiled function for speed:) )
                distance = dtaidistance.dtw.distance(time_series1, time_series2, use_c=True)

                # symmetrical slice
                matrix[i, j] = distance
                matrix[j, i] = distance
            print("<" + (i+1)*"#" + (n-i-1)*"-" + ">", end='\r')

        tensor.append(matrix)

        # limited amount of slices to include
        if amount_of_slices is not None and len(tensor) == amount_of_slices:
            break
    
    return tensor

def calc_element():
    pass

def save_overview():
    dal = DAL(dataset)
    overview = dal.overview()
    df = pd.DataFrame(overview)
    df.to_csv("overview.csv")

tensor = build_tensor(20, 20)
print(tensor)
