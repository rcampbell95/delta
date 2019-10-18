from conv_autoencoder import ConvAutoencoderGenotype
from tensorflow.keras.callbacks import Callback

class Individual():
    def __init__(self, config, new_genotype=False):
        self.batch_size = int(config["ml"]["batch_size"])
        self.epochs = int(config["ml"]["num_epochs"])
        self.steps_per_epoch = int(config["ml"]["steps_per_epoch"])
        self.config = config
        self.model_path = self.config["ml"]["model_folder"] + "/" + self.config["ml"]["model_dest_name"]

        if new_genotype == False:
            self.genotype = ConvAutoencoderGenotype(config)
        else:
            self.genotype = new_genotype

    def self_mutate(self):
        self.genotype.mutate_hidden_genes()

    def generate_child(self):
        child_genotype = self.genotype.replicate(self.config)
        child = Individual(self.config, child_genotype)
        return child

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
        with open(self.config["ml"]["output_folder"] + '/modelsummary.txt', 'w') as f:
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

        pd.DataFrame(gene_attrs).to_csv(self.config["ml"]["output_folder"] + "/genotype.csv")

        # Model callbacks
        early_stopping = EarlyStopping(monitor=self.config["evolutionary_search"]["metric"], 
                                       mode='min', 
                                       verbose=1, 
                                       patience=self.epochs // 10)

        filepath = self.config["ml"]["model_folder"] + "/" + self.config["ml"]["model_dest_name"]
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
        # history.history["test_psnr"] = psnr.test_psnr

        return history
