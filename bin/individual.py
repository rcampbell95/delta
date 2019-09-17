from cgp import Genotype
from tensorflow.keras.callbacks import Callback

class PSNRCallback(Callback):
    def __init__(self, X, Y, batch_size):
        #super().__init__()
        self.X_train = X
        self.Y_train = Y
        self.batch_size = batch_size


    def on_train_begin(self, logs={}):
        self.test_psnr = []
        self.ssim = []

    def on_train_end(self, logs=None):
        from skimage.measure import compare_psnr
        from skimage.measure import compare_ssim
        import numpy as np 

        train_prediction = np.asarray(self.model.predict(self.X_train, batch_size=self.batch_size, verbose=0))
        psnr = compare_psnr(np.asarray(self.Y_train), train_prediction.astype(self.Y_train.dtype))
        #ssim = compare_ssim(self.X_train, train_prediction, multichannel=True)

        self.test_psnr.append(psnr)

        print("- test psnr: {}".format(psnr))
        #print(" - ssim: {}".format(ssim))

class Individual():
    def __init__(self, config, new_genotype=False):
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
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
                
        input_shape = config["data_shape"]

        if "random_seed" not in self.config:
            self.config["random_seed"] = np.random.random()

        self.model = self.genotype.build_model(input_shape, shape=self.config["shape"], random_seed=self.config["random_seed"])

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
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.epochs // 10)
        checkpoint = ModelCheckpoint('./model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

        psnr = PSNRCallback(x_test, y_test, self.batch_size)

        history = self.model.fit(dataset, epochs=self.epochs, validation_split=.3, batch_size = self.batch_size, shuffle=True, callbacks=[psnr, checkpoint, early_stopping])
        history.history["test_psnr"] = psnr.test_psnr

        return history