import multiprocessing
import os
from functools import reduce

import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, GlobalAveragePooling2D
import pandas as pd
import numpy as np

from delta.config import config
from delta.imagery import imagery_dataset
from delta.ml.train import train

from delta.search.conv_autoencoder import ConvAutoencoderGenotype

def assemble_dataset():

    # Use wrapper class to create a Tensorflow Dataset object.
    # - The dataset will provide image chunks and corresponding labels.
    tc = config.training()

    ids = imagery_dataset.AutoencoderDataset(config.images(), config.chunk_size(), tc.chunk_stride)

    return ids

def psnr(img1, img2):
    return tf.image.psnr(img1, img2, max_val=1)

class Individual(multiprocessing.Process):
    fitness_queue = multiprocessing.Queue(0)

    def __init__(self, output_folder, device_manager, new_genotype=False, child_index=0):
        multiprocessing.Process.__init__(self)

        self.history = None
        self.child_index = child_index
        self.output_folder = output_folder
        self.device_manager = device_manager
        self.devices = None
        self.input_shape = None

        if new_genotype is False:
            self.genotype = ConvAutoencoderGenotype()
        else:
            self.genotype = new_genotype

    def _log_genetics(self):
        # Save genetics to csv
        gene_attrs = {}
        for gene in reversed(self.genotype.trace_encoder()):
            for attr, val in gene.attrs.items():
                if attr in gene_attrs:
                    gene_attrs[attr].append(val)
                else:
                    gene_attrs[attr] = [val]
            if "Connection id" in gene_attrs:
                gene_attrs["Connection id"].append(gene.conn)
            else:
                gene_attrs["Connection id"] = [gene.conn]

        if os.path.exists(os.path.join(self.output_folder, str(self.child_index))):
            csv_filename = os.path.join(self.output_folder, str(self.child_index), "genotype.csv")
            pd.DataFrame(gene_attrs).to_csv(csv_filename)
        else:
            print("Cannot save file. Temp path does not exist.")

    def _request_device(self):
        self.devices = self.device_manager.request()

    def _release_device(self):
        for device in self.devices:
            self.device_manager.release(device)

    @classmethod
    def histories(cls):
        histories = []

        while cls.fitness_queue.qsize() > 0:
            msg = cls.fitness_queue.get(block=False)
            histories.append(msg)
        histories = sorted(histories, key=lambda history: history[0])
        histories = list(map(lambda history: history[1], histories))

        return histories

    def self_mutate(self):
        self.genotype.mutate_hidden_genes()

    def generate_child(self, child_index):
        child_genotype = self.genotype.replicate()
        child = Individual(self.output_folder, self.device_manager, child_genotype, child_index)
        return child

    def build_model(self):
        model = self.genotype.build_model(self.input_shape)

        self._log_genetics()

        return model

    def _criterion_loss(self, model, history, gamma):
        def _filter_none(shape):
            return filter(lambda x: x is not None, shape)

        def _multiply(shape):
            return reduce(lambda x, y: x * y, shape)

        def criterion_loss(loss):
            reconstruction_loss =  loss

            criterion =  (np.log(shape_flatten_input) * shape_flatten_features) - 2*np.log(loss)

            weighted_loss = (1 - gamma) * reconstruction_loss + gamma * criterion
            return weighted_loss

        for idx, layer in enumerate(model.layers):
            # Checks for first layer of decoder
            if isinstance(layer, (Conv2DTranspose, GlobalAveragePooling2D)):
                feature_layer_idx = idx - 2
                break

        feature_shape = model.layers[feature_layer_idx].output_shape

        shape_flatten_input = _multiply(_filter_none(self.input_shape))
        shape_flatten_features = _multiply(_filter_none(feature_shape))

        _history = {}

        for metric in history.keys():
            if "loss" in metric:
                _history[metric] = list(map(criterion_loss, history[metric]))
            else:
                _history[metric] = history[metric]
        return _history

    def run(self):
        with tf.Graph().as_default():
            self._request_device()

            train_spec = config.training()
            train_spec.devices = self.devices
            _patience = lambda x: 10 if x > 100 else (x ** -1) * 10 **3
            train_spec.callbacks.extend([
                tf.keras.callbacks.EarlyStopping(patience=_patience(train_spec.epochs))
            ])
            #train_spec.metrics = [psnr]

            ids = assemble_dataset()

            self.input_shape = (config.chunk_size(), config.chunk_size(), ids.num_bands())

            model, history = train(self.build_model, ids, train_spec)

            history.history = self._criterion_loss(model, history.history, 0.2)

            model_path = os.path.join(self.output_folder, str(self.child_index), "model.h5")
            model.save(model_path)

            self._release_device()

            msg = (self.child_index, history.history)

            self.fitness_queue.put(msg)
