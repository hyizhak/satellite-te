import pickle

path_template = lambda internsity, mode, volume: f'/data/projects/11003765/sate/input/starlink/DataSetForSaTE{internsity}/{mode}/StarLink_DataSetForAgent100_{volume}_B.pkl'

dataset = []

for intensity in [25, 50, 75, 100]:
    for mode in ['GrdStation', 'ISL']:
        with open(path_template(intensity, mode, 5000), 'rb') as f:
            data = pickle.load(f)

        dataset.extend(data[:1250])

with open('/data/projects/11003765/sate/input/starlink/DataSetForSaTE/StarLink_DataSetForAgent_mixed_5000.pkl', 'wb') as file:
    pickle.dump(dataset, file)