import pickle

path_template = lambda intensity, mode, volume: f'/data/projects/11003765/sate/input/starlink/DataSetForSaTE{intensity}/{mode}/StarLink_DataSetForAgent{intensity}_{volume}_B.pkl'


for mode in ['GrdStation', 'ISL']:

    dataset = []

    for intensity in [25, 50, 75, 100]:
        with open(path_template(intensity, mode, 5000), 'rb') as f:
            data = pickle.load(f)

        dataset.extend(data[:1000])

    with open(f'/data/projects/11003765/sate/input/starlink/mixed/{mode}/StarLink_DataSetForAgent_mixed_4000.pkl', 'wb') as file:
        pickle.dump(dataset, file)