import pickle
import torch

sol_dir = '/data/projects/11003765/sate/input/lp_solutions/Gurobi_size-5000_mode-ISL_intensity-100_volume-1000_solutions.pkl'

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

def write_pickle_file(objects, file_path):
    """Writes all objects to a pickle file."""
    with open(file_path, 'wb') as f:
        for obj in objects:
            pickle.dump(obj, f)

print(len(solutions))