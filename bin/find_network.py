from gpu_manager import GPU_Manager
from individual import Individual

# def log_artifacts(config_values, fittest_index):
#     #mlflow.log_artifact("./model.png")
#     output_folder = os.path.join(config.output_folder(), str(fittest_index))
#     model_name = config_values["ml"]["model_dest_name"]

#     assert(os.path.exists(os.path.join(output_folder, model_name)))
#     assert(os.path.exists(os.path.join(output_folder, "modelsummary.txt")))
#     assert(os.path.exists(os.path.join(output_folder, "genotype.csv")))

#     mlflow.log_artifact(os.path.join(output_folder, model_name))
#     mlflow.log_artifact(os.path.join(output_folder, "modelsummary.txt"))
#     mlflow.log_artifact(os.path.join(output_folder, "genotype.csv"))


def find_network(config_values):
    #log = True if config_values["evolutionary_search"]["log"].lower() == "true" else False

    #if log:
    #    #mlflow.set_tracking_uri("file:./mlruns")
    #    mlflow.start_run()

    #    for category in config_values.keys():
    #        for param in config_values[category].keys():
    #            mlflow.log_param(param, config_values[category][param])


    ###
    # Train and evaluate generation 0
    ###
    print("Generation 0")
    device_manager = GPU_Manager()
    parent = Individual(config_values, device_manager)

    parent.start()
    parent.join()

    best_history = Individual.histories()[0]
    metric_best = min(best_history[config_values["evolutionary_search"]["metric"]])

    #if log:
    #    log_artifacts(config_values, 0)
    #    # Log parent as baseline for best model
    #    for metric in best_history.keys():
    #        mlflow.log_metric(metric, min(best_history[metric]), step=0)

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

            print("Child {} is more fit than parent. \
                   Parent for next generation will be child {}".format(idx_fittest + 1, idx_fittest + 1))

            #if log:
            #    best_history = child_histories[idx_fittest]
            #    for metric in child_histories[idx_fittest].keys():
            #        mlflow.log_metric(metric, min(child_histories[idx_fittest][metric]), step=generation)
            #        print("Logged {} to mlflow".format(metric))
            #    try:
            #        log_artifacts(config_values, idx_fittest)
            #        print("Logged architecture")
            #    except:
            #        print("Artifact can't be logged!")
        else:
            #for metric in best_history.keys():
            #    mlflow.log_metric(metric, min(best_history[metric]), step=generation)
            print("Children were less fit than parent model. Continuing with parent for next generation")
            parent.self_mutate()


    #if log:
    #    mlflow.end_run()

if __name__ == "__main__":
    ###
    # Test find_network()
    ###
    evolution_config = {
        "evolutionary_search" : {
            "grid_height": 10,
            "grid_width": 1,
            "level_back": 1,
            "num_children": 1,
            "generations": 20,
            "r": .1,
            "metric": "val_loss",
            "log": True,
            "shape": "asymmetric"
        },
        "ml": {
            "channels": 7
        }
    }

    find_network(evolution_config)

