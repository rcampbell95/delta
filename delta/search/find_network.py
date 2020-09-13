# Copyright © 2020, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All rights reserved.
#
# The DELTA (Deep Earth Learning, Tools, and Analysis) platform is
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
import shutil
import os

import mlflow
import numpy as np

import tensorflow.keras as keras

from delta.search.individual import Individual

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from delta.config import config

def log_artifacts(output_folder, fittest_index):
    output_folder = os.path.join(output_folder, str(fittest_index))
    model_name = "model.h5"

    assert os.path.exists(os.path.join(output_folder, model_name))
    assert os.path.exists(os.path.join(output_folder, "genotype.csv"))

    mlflow.log_artifact(os.path.join(output_folder, model_name))
    mlflow.log_artifact(os.path.join(output_folder, "genotype.csv"))

def log_params():
    mlflow.log_param("Grid Height", config.search.grid_height())
    mlflow.log_param("Grid Width", config.search.grid_width())
    mlflow.log_param("Level Back", config.search.level_back())
    mlflow.log_param("Shape", config.search.shape())
    mlflow.log_param("r", config.search.r())
    mlflow.log_param("Children", config.search.children())
    mlflow.log_param("Generations", config.search.generations())
    mlflow.log_param("Fitness Metric", config.search.fitness_metric())
    mlflow.log_param("Gamma", config.search.gamma())

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
    log = config.search.log()

    print(config.general.gpus())

    if log:
        output_folder = create_tmp_dirs(config.search.children())
        print(output_folder)
        mlflow.set_tracking_uri(config.mlflow.uri())
        mlflow.set_experiment(config.mlflow.experiment())
        mlflow.start_run()

        log_params()

    #device_manager = GPU_Manager()
    ###
    # Train and evaluate generation 0
    ###
    print("Generation 0")
    parent = Individual(output_folder)

    parent.start()
    parent.join()

    best_history = Individual.histories()[0]
    metric_best = min(best_history[config.search.fitness_metric()])

    if log:
        log_artifacts(output_folder, 0)
        # Log parent as baseline for best model

        epoch_index = np.argmin(best_history[config.search.fitness_metric()])
        for metric, value in best_history.items():
            #if "test" in metric:
            #    mlflow.log_metric(metric, value[0].item(), step=0)
            #else:
            print(value[epoch_index])
            mlflow.log_metric(metric, value[epoch_index], step=0)

    for generation in range(1, int(config.search.generations()) + 1):
        print("\n\n\nGeneration ", generation)

        children = [parent.generate_child(i) for i in range(int(config.search.children()))]

        for child in children:
            child.start()
        for child in children:
            child.join()

        child_histories = Individual.histories()

        metric_vals = list(map(lambda history: \
                           min(history[config.search.fitness_metric()]), child_histories))
        idx_fittest = metric_vals.index(min(metric_vals))

        if metric_vals[idx_fittest] < metric_best:
            parent = children[idx_fittest]
            metric_best = metric_vals[idx_fittest]

            print("Child {} is more fit than parent.".format(idx_fittest + 1))
            print("Parent for next generation will be child {}".format(idx_fittest + 1))

            if log:
                best_history = child_histories[idx_fittest]
                epoch_index = np.argmin(best_history[config.search.fitness_metric()])

                for metric, value in best_history.items():
                    #if "test" in metric:
                    #    mlflow.log_metric(metric, value[0].item(), step=generation)
                    #else:
                    print(value[epoch_index])
                    mlflow.log_metric(metric, value[epoch_index], step=generation)
                    print("Logged {} to mlflow".format(metric))
                try:
                    log_artifacts(output_folder, idx_fittest)
                    print("Logged architecture")
                except AssertionError:
                    print("Artifact can't be logged!")
        else:
            if log:
                epoch_index = np.argmin(best_history[config.search.fitness_metric()])
                for metric, value in best_history.items():
                    if "test" in metric:
                        mlflow.log_metric(metric, value[0], step=generation)
                    else:
                        mlflow.log_metric(metric, value[epoch_index], step=generation)
            print("Children were less fit than parent model. Continuing with parent for next generation")
            parent.self_mutate()


    if log:
        tracking_uri = mlflow.get_artifact_uri()
        fittest_model = keras.models.load_model(os.path.join(tracking_uri, "model.h5"), compile=False)

        mlflow.end_run()
        cleanup_tmp_dirs(output_folder)
    else:
        fittest_model = None

    return fittest_model


if __name__ == "__main__":
    ###
    # Test find_network()
    ###

    find_network()
