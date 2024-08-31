import os

folder_gen = lambda intensity, mode: f'/data/projects/11003765/sate/satte/satellite-te/output/comb_supervised/DataSetForSaTE{intensity}_{mode}_spaceTE/models/spaceTE_supervised-kl_div_ep-60_dummy-path-False_flow-lambda-25_layers-0_base-curriculum_supervised_mixed_{mode}'

folder = folder_gen(50, 'ISL')

files = os.listdir(folder)

files = sorted(files)

for file_name in files:
    if file_name.endswith('.pt'):
        old_path = os.path.join(folder, file_name)

        new_name = 'epoch_' + file_name.split('_')[-1]

        new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)

    if file_name.endswith('.trainings') and not file_name.endswith('60.trainings'):

        os.remove(os.path.join(folder, file_name))
    

print(files)



