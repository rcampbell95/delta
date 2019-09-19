import tensorflow
import numpy as np
from individual import Individual
import mlflow
import os


def log_artifacts():
    #mlflow.log_artifact("./model.png")
    #mlflow.log_artifact("./model.h5")
    mlflow.log_artifact("./modelsummary.txt")
    mlflow.log_artifact("./genotype.csv")


def find_network(config, dataset):
    log = bool(config["evolutionary_search"]["log"])

    if log:
        #mlflow.set_tracking_uri("file:./mlruns")
        mlflow.start_run()

        for param in config.keys():
            mlflow.log_param(param, config[param])


    ###
    # Train and evaluate generation 0 
    ###
    print("Generation 0")
    parent = Individual(config)
    training_history = parent.evaluate_fitness(dataset)
    metric_best = min(training_history.history[config["evolutionary_search"]["metric"]])

    if log:
        log_artifacts()
        # Log parent as baseline for best model
        for metric in training_history.history.keys():
            mlflow.log_metric(metric, min(training_history.history[metric]))

    for i in range(1, int(config["evolutionary_search"]["generations"]) + 1):
        print("\n\n\nGeneration ", i)
        
        children = [parent.generate_child() for i in range(int(config["evolutionary_search"]["num_children"]))]

        child_histories = []

        for child in children:
            child = parent.generate_child()
            child_histories.append(child.evaluate_fitness(dataset))
            #tensorflow.keras.backend.clear_session()

        metric_vals = list(map(lambda history: min(history.history[config["evolutionary_search"]["metric"]]), child_histories))
        idx_fittest = metric_vals.index(min(metric_vals))

        if metric_vals[idx_fittest] < metric_best:
            parent = children[idx_fittest]
            metric_best = metric_vals[idx_fittest]

            print("Child {} is more fit than parent. Parent for next generation will be child {}".format(idx_fittest + 1, idx_fittest + 1))

            if log:
                for metric in child_histories[idx_fittest].history.keys():
                    mlflow.log_metric(metric, min(child_histories[idx_fittest].history[metric]))
                    print("Logged {} to mlflow".format(metric))
                mlflow.log_metric("generation", i)

                try:
                    log_artifacts()
                    print("Logged architecture")
                except:
                    print("Artifact can't be logged!")
        else:
            print("Children were less fit than parent model. Continuing with parent for next generation")
            parent.self_mutate()


    if log:
        mlflow.end_run()


if __name__ == "__main__":
    import os
    import sys
    import argparse
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from delta.imagery import imagery_dataset 
    from delta import config 

    usage  = "usage: train_autoencoder [options]"
    parser = argparse.ArgumentParser(usage=usage)

    parser = argparse.ArgumentParser(usage='train_autoencoder.py [options]')

    parser.add_argument("--config-file", dest="config_file", default=None,
                        help="Dataset configuration file.")
    parser.add_argument("--data-folder", dest="data_folder", default=None,
                        help="Specify data folder instead of supplying config file.")
    parser.add_argument("--image-type", dest="image_type", default=None,
                        help="Specify image type along with the data folder."
                        +"(landsat, landsat-simple, worldview, or rgba)")

    #try:
    options = parser.parse_args()
    #except argparse.ArgumentError:
    #``    print(usage)
        #return -1

    config_values = config.parse_config_file(options.config_file,
                                             options.data_folder, options.image_type)

    print('loading data from ' + config_values['input_dataset']['data_directory'])
    aeds = imagery_dataset.AutoencoderDataset(config_values)
    ds = aeds.dataset()

    ##num_bands = aeds.num_bands()

    batch_size = config_values['ml']['batch_size']
    num_epochs = config_values["ml"]["num_epochs"]

    ds = ds.repeat(num_epochs).batch(batch_size)

    print(aeds.num_bands())

    find_network(config_values, ds)