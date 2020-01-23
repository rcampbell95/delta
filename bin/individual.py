from conv_autoencoder import ConvAutoencoderGenotype
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import os 
import multiprocessing

def load_data(config_values):
    from delta.imagery import imagery_dataset


    batch_size = config_values['ml']['batch_size']
    num_epochs = config_values["ml"]["num_epochs"]

    print('loading data from ' + config_values['input_dataset']['data_directory'])
    aeds_train = imagery_dataset.AutoencoderDataset(config_values)
    train_ds = aeds_train.dataset(filter_zero=False)

    train_ds = train_ds.repeat(num_epochs).batch(batch_size)

    train_directory = config_values['input_dataset']['data_directory']

    print('loading validation data from ' + config_values['input_dataset']['val_directory'])
    config_values['input_dataset']['data_directory'] = config_values['input_dataset']['val_directory']
    aeds_val = imagery_dataset.AutoencoderDataset(config_values)
    val_ds = aeds_val.dataset(filter_zero=False)

    val_ds = val_ds.repeat(num_epochs).batch(batch_size)

    config_values['input_dataset']['data_directory'] = train_directory
 
    return train_ds, val_ds

class Individual(multiprocessing.Process):
    fitness_queue = multiprocessing.Queue(0)


    def __init__(self, config, new_genotype=False, child_index=0):
        multiprocessing.Process.__init__(self)



        self.history = None
        self.child_index = child_index
        self.batch_size = int(config["ml"]["batch_size"])
        self.epochs = int(config["ml"]["num_epochs"])
        self.steps_per_epoch = int(config["ml"]["steps_per_epoch"])
        self.config = config
        self.model_path = os.path.join(self.config["ml"]["model_folder"], self.config["ml"]["model_dest_name"])

        self.output_folder = os.path.join(self.config["ml"]["output_folder"], str(child_index))

        if new_genotype == False:
            self.genotype = ConvAutoencoderGenotype(config)
        else:
            self.genotype = new_genotype

    def self_mutate(self):
        self.genotype.mutate_hidden_genes()

    def generate_child(self, child_index):
        child_genotype = self.genotype.replicate(self.config)
        child = Individual(self.config, child_genotype, child_index)
        return child

    @classmethod
    def histories(cls):
        histories = []
        while cls.fitness_queue.qsize() > 0:
            print(cls.fitness_queue.qsize())
            msg = cls.fitness_queue.get(block=False)
            histories.append(msg[1])
            
        return histories


    def evaluate_fitness(self, trainset, valset):
        from tensorflow.keras.utils import plot_model
        from tensorflow.keras.callbacks import ModelCheckpoint
        from tensorflow.keras.callbacks import EarlyStopping
        import tensorflow as tf
        from contextlib import redirect_stdout
        import pandas as pd
        import numpy as np
        
        chunk_size = int(self.config["ml"]["chunk_size"])
        channels = int(self.config["ml"]["channels"])

        input_shape = (chunk_size, chunk_size, channels)

        model = self.genotype.build_model(self.config, input_shape)

        # Save model summary to text file
        # TODO -- make child directories if not available
        # TODO -- best option would be to make temp directory and delete after training
        # TODO -- not high priority though
        with open(os.path.join(self.config["ml"]["output_folder"], str(self.child_index), 'modelsummary.txt'), 'w') as f:
            with redirect_stdout(f):
                model.summary()

        # Save genetics to csv
        gene_attrs = {}
        for gene in self.genotype.genes:
            for attr, val in gene.attrs.items():
                if attr in gene_attrs:
                    gene_attrs[attr].append(val)
                else:
                    gene_attrs[attr] = [val]
            if "Connection id" in gene_attrs:
                gene_attrs["Connection id"].append(gene.conn)
            else:
                gene_attrs["Connection id"] = [gene.conn]

        pd.DataFrame(gene_attrs).to_csv(os.path.join(self.config["ml"]["output_folder"], str(self.child_index), "genotype.csv"))

        # Model callbacks
        early_stopping = EarlyStopping(monitor=self.config["evolutionary_search"]["metric"], 
                                       mode='min', 
                                       verbose=1, 
                                       patience=self.epochs // 10)

        filepath = os.path.join(self.config["ml"]["model_folder"], str(self.child_index), self.config["ml"]["model_dest_name"])
        checkpoint = ModelCheckpoint(filepath       = filepath, 
                                     monitor        = self.config["evolutionary_search"]["metric"], 
                                     mode           = 'min', 
                                     save_best_only = True, 
                                     verbose        = 1)
        
        # Remove psnr from callbacks, for now
        history = model.fit(x                = trainset, 
                            epochs           = self.epochs, 
                            batch_size       = None, 
                            steps_per_epoch  = self.steps_per_epoch, 
                            shuffle          = True, 
                            callbacks        = [checkpoint, early_stopping],
                            validation_data  = valset,
                            validation_steps = None if valset is None else self.steps_per_epoch)
        #self.history = history
        # history.history["test_psnr"] = psnr.test_psnr
        return history
        # start thread
        # load config values
        # get gpu 
        # set up context
        # load data
        # train model
        # return fitness value


    def run(self):
        from gpu_manager import GPU_Manager

        device_manager = GPU_Manager()
        device = device_manager.request_device()

        with tf.Graph().as_default() as g:
            with tf.device(device):
                trainset, valset = load_data(self.config)

                history = self.evaluate_fitness(trainset, valset)

                msg = (self.child_index, history.history)

                self.fitness_queue.put(msg)
                
                print(id(self.fitness_queue))
                print(self.fitness_queue.qsize())
