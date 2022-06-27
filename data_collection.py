import multiprocessing as mp
import numpy as np

from sandpile_model import Sandpile_model

def run_and_save_model(parameters):

    model = Sandpile_model(**parameters)
    model.load_or_run()

def get_treshold_research_arguments():
    
    arguments = []
    tresholds = np.arange(1, 11, 3)

    for i, grain_treshold1 in enumerate(tresholds):

        # Since the results are symmetric not all possibilies need to be tested
        # eg. the tresholds (1, 2) and (2, 1) should give the same results
        for j, grain_treshold2 in enumerate(tresholds[:i + 1]):

            # Paramaters for the model
            arguments.append({
                "grid_size": 32,
                "n_steps": 1000000,
                "crit_values": [grain_treshold1, grain_treshold2],
                "n_grain_types": 2,
                "init_method": "random",
                "add_method": "random"
            })

    return arguments

if __name__ == "__main__":

    arguments = get_treshold_research_arguments()

    # Creates a pool of processes, which will perform the run_and_save_model function for
    # each argument in arguments. The argument is the parameter settings for the model
    pool = mp.Pool()
    pool.map(run_and_save_model, arguments)
