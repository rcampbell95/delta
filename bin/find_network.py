import os
import tempfile
import shutil

import mlflow
import numpy as np

from gpu_manager import GPU_Manager
from individual import Individual

def log_artifacts(config_values, fittest_index):
    output_folder = os.path.join(config_values["ml"]["output_folder"], str(fittest_index))
    model_name = "model.h5"

    assert os.path.exists(os.path.join(output_folder, model_name))
    assert os.path.exists(os.path.join(output_folder, "genotype.csv"))

    mlflow.log_artifact(os.path.join(output_folder, model_name))
    mlflow.log_artifact(os.path.join(output_folder, "genotype.csv"))

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

def find_network(config_values):
    log = config_values["evolutionary_search"]["log"].lower() == "true"

    if log:
        output_folder = create_tmp_dirs(config_values["evolutionary_search"]["num_children"])
        print(output_folder)
        config_values["ml"]["output_folder"] = output_folder
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(config_values["ml"]["experiment_name"])
        mlflow.start_run()

        for category in config_values.keys():
            for param in config_values[category].keys():
                mlflow.log_param(param, config_values[category][param])

    device_manager = GPU_Manager()
    ###
    # Train and evaluate generation 0
    ###
    print("Generation 0")
    parent = Individual(config_values, device_manager)

    parent.start()
    parent.join()

    best_history = Individual.histories()[0]
    metric_best = min(best_history[config_values["evolutionary_search"]["metric"]])

    if log:
        log_artifacts(config_values, 0)
    #    # Log parent as baseline for best model

        epoch_index = np.argmin(best_history[config_values["evolutionary_search"]["metric"]])
        for metric, value in best_history.items():
            mlflow.log_metric(metric, value[epoch_index], step=0)

    for generation in range(1, int(config_values["evolutionary_search"]["generations"]) + 1):
        print("\n\n\nGeneration ", generation)

        children = [parent.generate_child(i) for i in range(int(config_values["evolutionary_search"]["num_children"]))]

        for child in children:
            child.start()
        for child in children:
            child.join()

        child_histories = Individual.histories()

        metric_vals = list(map(lambda history: \
                           min(history[config_values["evolutionary_search"]["metric"]]), child_histories))
        idx_fittest = metric_vals.index(min(metric_vals))

        if metric_vals[idx_fittest] < metric_best:
            parent = children[idx_fittest]
            metric_best = metric_vals[idx_fittest]

            print("Child {} is more fit than parent.".format(idx_fittest + 1))
            print("Parent for next generation will be child {}".format(idx_fittest + 1))

            if log:
                best_history = child_histories[idx_fittest]
                epoch_index = np.argmin(best_history[config_values["evolutionary_search"]["metric"]])

                for metric, value in best_history.items():
                    mlflow.log_metric(metric, value[epoch_index], step=generation)
                    print("Logged {} to mlflow".format(metric))
                try:
                    log_artifacts(config_values, idx_fittest)
                    print("Logged architecture")
                except AssertionError:
                    print("Artifact can't be logged!")
        else:
            if log:
                epoch_index = np.argmin(best_history[config_values["evolutionary_search"]["metric"]])
                for metric, value in best_history.items():

                    mlflow.log_metric(metric, value[epoch_index], step=generation)
            print("Children were less fit than parent model. Continuing with parent for next generation")
            parent.self_mutate()


    if log:
        mlflow.end_run()
        cleanup_tmp_dirs(output_folder)


if __name__ == "__main__":
    ###
    # Test find_network()
    ###
    evolution_config = {
        "evolutionary_search" : {
            "grid_height": 10,
            "grid_width": 3,
            "level_back": 2,
            "num_children": 1,
            "generations": 100,
            "r": .3,
            "metric": "val_loss",
            "log": "true",
            "shape": "symmetric",
        },
        "ml": {
            "channels": 7,
            "experiment_name": "test_r_val"
        }
    }

    find_network(evolution_config)
