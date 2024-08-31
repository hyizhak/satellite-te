# file_path_template = lambda mode, intensity, step: f'/data/projects/11003765/sate/satte/satellite-te/output/comb_supervised/DataSetForSaTE{intensity}_{mode}_spaceTE/test_logs/spaceTE_supervised-kl_div_ep-1_dummy-path-False_flow-lambda-25_layers-0_base-None_step-{step}_failures-0_admm-test-False.csv'

# file_path_template = lambda mode, intensity, step: '/data/projects/11003765/sate/satte/satellite-te/output/comb_supervised/new_form_Intensity_15_spaceTE/test_logs/spaceTE_supervised-kl_div_ep-30_dummy-path-False_flow-lambda-25_layers-0_base-None_step-10_failures-0_admm-test-False.csv'

def file_path_template(size):
    match size:
        case 66:
            return '/data/projects/11003765/sate/satte/satellite-te/output/scalability_4000/new_form_Intensity_15_spaceTE/test_logs/spaceTE_supervised-kl_div_ep-1_dummy-path-False_flow-lambda-25_layers-0_base-None_step-0_failures-0_admm-test-False.csv'
        case 500:
            return '/data/projects/11003765/sate/satte/satellite-te/output/scalability_4000/starlink_500_ISL_spaceTE/test_logs/spaceTE_supervised-kl_div_ep-1_dummy-path-False_flow-lambda-25_layers-0_base-None_step-0_failures-0_admm-test-False.csv'
        case 1500:
            return '/data/projects/11003765/sate/satte/satellite-te/output/scalability_4000/starlink_1500_ISL_spaceTE/test_logs/spaceTE_supervised-kl_div_ep-1_dummy-path-False_flow-lambda-25_layers-0_base-None_step-0_failures-0_admm-test-False.csv'
        case 4000:
            return '/data/projects/11003765/sate/satte/satellite-te/output/scalability_4000/DataSetForSaTE100_ISL_spaceTE/test_logs/spaceTE_supervised-kl_div_ep-1_dummy-path-False_flow-lambda-25_layers-0_base-None_step-0_failures-0_admm-test-False.csv'

import pandas as pd
import numpy as np
from scipy.io import savemat

for size in [66, 500, 1500, 4000]:
    file_path = file_path_template(size)

    df = pd.read_csv(file_path)

    for runtime in ['pre_runtime', 'model_runtime', 'post_runtime']:

        column_data = df[runtime].values

        Q1 = np.percentile(column_data, 25)
        Q3 = np.percentile(column_data, 75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_data = column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]

        # filtered_data = column_data

        # mean = np.mean(column_data)
        # std_dev = np.std(column_data)

        # lower_bound = mean - 3 * std_dev
        # upper_bound = mean + 3 * std_dev

        # filtered_data = column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]


        print(size, runtime, filtered_data.mean())

        mat_file_path = f'/data/projects/11003765/sate/satte/satellite-te/output/runtime/runtime_{size}.mat'

        savemat(mat_file_path, {runtime: filtered_data})