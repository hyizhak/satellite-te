import pickle
import numpy as np

dataset_path = '/data/projects/11003765/sate/input/starlink/DataSetForSaTE100/ISL/StarLink_DataSetForAgent100_5000_B.pkl'

with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)

def calculate_average_length(dictionary):
    total_length = 0
    num_sublists = 0
    
    for key, lists in dictionary.items():
        for sublist in lists:
            total_length += len(sublist)
            num_sublists += 1
    
    average_length = total_length / num_sublists
    return average_length

path_len = 0
for i in range(10):
    path_len += calculate_average_length(dataset[i]['path'])

print(path_len / 10)



