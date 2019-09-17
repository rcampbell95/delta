import tensorflow
import numpy as np
from individual import Individual
import mlflow
import os


def log_artifacts():
    #mlflow.log_artifact("./model.png")
    mlflow.log_artifact("./model.h5")
    mlflow.log_artifact("./modelsummary.txt")
    mlflow.log_artifact("./genotype.csv")




def find_network(config, dataset):
    if config["log"]:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.start_run()

        for param in config.keys():
            mlflow.log_param(param, config[param])


    ###
    # Train and evaluate generation 0 
    ###
    print("Generation 0")
    parent = Individual(config)
    training_history = parent.evaluate_fitness(dataset)
    metric_best = min(training_history.history[config["metric"]])

    if config["log"]:
        log_artifacts()
        # Log parent as baseline for best model
        for metric in training_history.history.keys():
            mlflow.log_metric(metric, min(training_history.history[metric]))

    for i in range(1, config["generations"] + 1):
        print("\n\n\nGeneration ", i)
        
        children = [parent.generate_child() for i in range(config["num_children"])]

        child_histories = []

        for child in children:
            child = parent.generate_child()
            child_histories.append(child.evaluate_fitness(dataset))
            tensorflow.keras.backend.clear_session()

        metric_vals = list(map(lambda history: min(history.history[config["metric"]]), child_histories))
        idx_fittest = metric_vals.index(min(metric_vals))

        if metric_vals[idx_fittest] < metric_best:
            parent = children[idx_fittest]
            metric_best = metric_vals[idx_fittest]

            print("Child {} is more fit than parent. Parent for next generation will be child {}".format(idx_fittest + 1, idx_fittest + 1))

            if config["log"]:
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


    if config["log"]:
        mlflow.end_run()
