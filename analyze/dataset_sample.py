import pickle

path_template = lambda size, mode, volume: f'/data/projects/11003765/sate/input/starlink/starlink_{size}/{mode}/StarLink_DataSetForAgent100_{volume}_Size{size}.pkl'

for size in [176, 500]:
    for mode in ['GrdStation', 'ISL']:
        with open(path_template(size, mode, 5000), 'rb') as f:
            data = pickle.load(f)

        print(len(data))

        # with open(path_template(size, mode, 10), 'wb') as file:
        #     pickle.dump(data[:10], file)

        # print(f'{size} {mode} 10 samples saved')
