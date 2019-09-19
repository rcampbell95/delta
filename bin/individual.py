from genetics import Genotype
from tensorflow.keras.callbacks import Callback

class Individual():
    def __init__(self, config, new_genotype=False):
        self.batch_size = int(config["ml"]["batch_size"])
        self.epochs = int(config["ml"]["num_epochs"])
        self.config = config

        if new_genotype == False:
            self.genotype = Genotype(config)
        else:
            self.genotype = new_genotype

    def self_mutate(self):
        self.genotype.mutate_hidden_genes()

    def generate_child(self):
        child_genotype = self.genotype.replicate(self.config)
        child = Individual(self.config, child_genotype)
        return child

    def evaluate_fitness(self, dataset):
        from tensorflow.keras.utils import plot_model
        from tensorflow.keras.callbacks import ModelCheckpoint
        from tensorflow.keras.callbacks import EarlyStopping
        import tensorflow as tf
        from contextlib import redirect_stdout
        import pandas as pd
        import numpy as np
        
        chunk_size = int(self.config["ml"]["chunk_size"])
        channels = int(self.config["ml"]["channels"])

        input_shape = (channels, chunk_size, chunk_size)

        if "random_seed" not in self.config:
            self.config["random_seed"] = np.random.random()

        self.model = self.genotype.build_model(input_shape, shape=self.config["evolutionary_search"]["shape"], random_seed=self.config["random_seed"])

        # plot_model(self.model, to_file="./model.png", show_shapes=True)

        # Save model summary to text file
        with open('./modelsummary.txt', 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

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

        pd.DataFrame(gene_attrs).to_csv("./genotype.csv")

        # Model callbacks
        early_stopping = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=self.epochs // 10)
        checkpoint = ModelCheckpoint('./model.h5', monitor='loss', mode='min', save_best_only=True, verbose=1)
        
        # Remove psnr from callbacks, for now
        history = self.model.fit(dataset, epochs=self.epochs, batch_size = None, steps_per_epoch=50, shuffle=True, callbacks=[checkpoint, early_stopping])
        # history.history["test_psnr"] = psnr.test_psnr

        return history