import pickle
import torch

sol_dir = '/data/projects/11003765/sate/input/lp_solutions/Gurobi_size-5000_mode-ISL_intensity-100_volume-1000_solutions.pkl'

problem_dir = '/data/projects/11003765/sate/input/starlink/DataSetForSaTE100/ISL/StarLink_DataSetForAgent100_5000_B.pkl'

def read_solutions(file_path):
    solutions = []
    with open(file_path, 'rb') as f:
        while True:
            try:
                # Load each solution sequentially
                sol = pickle.load(f)
                solutions.append(sol)
            except EOFError:
                # End of file reached
                break
    return solutions

solutions = read_solutions(sol_dir)

with open(problem_dir, 'rb') as f:
    problem = pickle.load(f)

def write_pickle_file(objects, file_path):
    """Writes all objects to a pickle file."""
    with open(file_path, 'wb') as f:
        for obj in objects:
            pickle.dump(obj, f)

flag = False
for i in range(len(solutions)):
    if solutions[i].shape[0] > len(problem[i]['tm']):
        print(f'{i}: {solutions[i].shape[0]} > {len(problem[i]["tm"])}')
        flag = True

if not flag:
    print('All Done!')