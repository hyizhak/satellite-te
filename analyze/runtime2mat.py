file_path_template = lambda mode, intensity: f'/data/projects/11003765/sate/satte/satellite-te/output/supervised/DataSetForSaTE{intensity}_{mode}_spaceTE/test_logs/spaceTE_obj-teal_total_flow_RL_lr-0.0001_ep-1_sample-200_layers-0_decoder-linear_step-10_failures-0_admm-test-False.csv'

import pandas as pd
import numpy as np
from scipy.io import savemat

for mode in ['ISL', 'GrdStation']:
    for intensity in [25, 50, 75, 100]:
        file_path = file_path_template(mode, intensity)

        df = pd.read_csv(file_path)

        column_data = df['runtime'].values

        Q1 = np.percentile(column_data, 25)
        Q3 = np.percentile(column_data, 75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_data = column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]

        # mean = np.mean(column_data)
        # std_dev = np.std(column_data)

        # lower_bound = mean - 2 * std_dev
        # upper_bound = mean + 2 * std_dev

        # filtered_data = column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]


        print(filtered_data.mean())

        mat_file_path = f'./runtime/runtime_{intensity}_{mode}.mat'

        savemat(mat_file_path, {'runtime': filtered_data})