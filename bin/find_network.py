import os
import tempfile
import shutil

import mlflow
import numpy as np

from gpu_manager import GPU_Manager
from individual import Individual


from delta.config import config

def log_artifacts(output_folder, fittest_index):
    output_folder = os.path.join(output_folder, str(fittest_index))
    model_name = "model.h5"

    assert os.path.exists(os.path.join(output_folder, model_name))
    assert os.path.exists(os.path.join(output_folder, "genotype.csv"))

    mlflow.log_artifact(os.path.join(output_folder, model_name))
    mlflow.log_artifact(os.path.join(output_folder, "genotype.csv"))

def log_params():
    mlflow.log_param("Grid Height", config.model_grid_height())
    mlflow.log_param("Grid Width", config.model_grid_width())
    mlflow.log_param("Level Back", config.model_level_back())
    mlflow.log_param("Shape", config.model_shape())
    mlflow.log_param("r", config.r())
    mlflow.log_param("Children", config.search_children())
    mlflow.log_param("Generations", config.search_generations())
    mlflow.log_param("Fitness Metric", config.search_fitness_metric())

def create_tmp_dirs(num_children):
    sysTemp = tempfile.gettempdir()

    temproot = tempfile.mkdtemp(dir=sysTemp)

    for child in range(num_children):
        dirname = os.path.join(temproot, str(child))

        if not os.path.exists(dirname):
            os.mkdir(dirname)

    return temproot

def cleanup_tmp_dirs(tmp_dir):
    shutil.rmtree(tmp_dir)

def find_network():
    log = config.log_search()

    if log:
        output_folder = create_tmp_dirs(config.search_children())
        print(output_folder)
        mlflow.set_tracking_uri(config.mlflow_uri())
        mlflow.set_experiment(config.training().experiment)
        mlflow.start_run()

        log_params()
        #for category in config_values.keys():
        #    for param in config_values[category].keys():
        #        mlflow.log_param(param, config_values[category][param])
    device_manager = GPU_Manager()
    ###
    # Train and evaluate generation 0
    ###
    print("Generation 0")
    parent = Individual(output_folder, device_manager)

    parent.start()
    parent.join()

    best_history = Individual.histories()[0]
    metric_best = min(best_history[config.search_fitness_metric()])

    if log:
        log_artifacts(output_folder, 0)
    #    # Log parent as baseline for best model

        epoch_index = np.argmin(best_history[config.search_fitness_metric()])
        for metric, value in best_history.items():
            mlflow.log_metric(metric, value[epoch_index].item(), step=0)

    for generation in range(1, int(config.search_generations()) + 1):
        print("\n\n\nGeneration ", generation)

        children = [parent.generate_child(i) for i in range(int(config.search_children()))]

        for child in children:
            child.start()
        for child in children:
            child.join()

        child_histories = Individual.histories()

        metric_vals = list(map(lambda history: \
                           min(history[config.search_fitness_metric()]), child_histories))
        idx_fittest = metric_vals.index(min(metric_vals))

        if metric_vals[idx_fittest] < metric_best:
            parent = children[idx_fittest]
            metric_best = metric_vals[idx_fittest]

            print("Child {} is more fit than parent.".format(idx_fittest + 1))
            print("Parent for next generation will be child {}".format(idx_fittest + 1))

            if log:
                best_history = child_histories[idx_fittest]
                epoch_index = np.argmin(best_history[config.search_fitness_metric()])

                for metric, value in best_history.items():
                    mlflow.log_metric(metric, value[epoch_index].item(), step=generation)
                    print("Logged {} to mlflow".format(metric))
                try:
                    log_artifacts(output_folder, idx_fittest)
                    print("Logged architecture")
                except AssertionError:
                    print("Artifact can't be logged!")
        else:
            if log:
                epoch_index = np.argmin(best_history[config.search_fitness_metric()])
                for metric, value in best_history.items():

                    mlflow.log_metric(metric, value[epoch_index].item(), step=generation)
            print("Children were less fit than parent model. Continuing with parent for next generation")
            parent.self_mutate()


    if log:
        mlflow.end_run()
        cleanup_tmp_dirs(output_folder)


if __name__ == "__main__":
    ###
    # Test find_network()
    ###

    find_network()
