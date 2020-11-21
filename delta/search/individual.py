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

import multiprocessing
import os
import copy
from functools import reduce

import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, GlobalAveragePooling2D

import pandas as pd

from delta.config import config
from delta.imagery import imagery_dataset
from delta.ml.train import train

from delta.search.conv_autoencoder import ConvAutoencoderGenotype

def assemble_dataset(train_spec):

    # Use wrapper class to create a Tensorflow Dataset object.
    # - The dataset will provide image chunks and corresponding labels.
    ids = imagery_dataset.AutoencoderDataset(config.dataset.images(),
                                             config.train.network.chunk_size(),
                                             train_spec.chunk_stride)
    return ids

def psnr(img1, img2):
    return tf.image.psnr(img1, img2, max_val=1)

def weighted_loss(model, history, gamma):
    def _filter_none(shape):
        return filter(lambda x: x is not None, shape)

    def _multiply(shape):
        return reduce(lambda x, y: x * y, shape)

    def _weighted_loss(mse):
        reconstruction_loss = mse

        parameters = shape_flatten_features

        loss = reconstruction_loss + gamma * parameters
        return loss

    for idx, layer in enumerate(model.layers):
        # Checks for first layer of decoder
        if isinstance(layer, (Conv2DTranspose, GlobalAveragePooling2D)):
            feature_layer_idx = idx
            break

    feature_shape = model.layers[feature_layer_idx].output_shape

    shape_flatten_features = _multiply(_filter_none(feature_shape))

    _history = {}

    for metric in history.keys():
        if "loss" in metric:
            _history[metric] = list(map(_weighted_loss, history[metric]))
        else:
            _history[metric] = history[metric]
    return _history

class Individual(multiprocessing.Process):
    fitness_queue = multiprocessing.Queue(0)

    def __init__(self, output_folder, device_manager=None, new_genotype=False, child_index=0):
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
        """
        Class method that fetches training history from
        queue for all children

        Parameters
        ----------
        Returns
        ----------
        histories : list
            Ordered list of training histories ordered by
            child id
        """
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

    def run(self):
        train_spec = copy.deepcopy(config.train.spec())
        train_spec.devices = self.devices
        _patience = lambda x: 10 if x > 100 else (x ** -1) * 10 **3
        #train_spec.callbacks.extend([
        #    tf.keras.callbacks.EarlyStopping(patience=_patience(train_spec.epochs))
        #])

        ads = assemble_dataset(train_spec)

        self.input_shape = (config.train.network.chunk_size(), config.train.network.chunk_size(), ads.num_bands())

        model, history = train(self.build_model, ads, train_spec)

        history.history = weighted_loss(model, history.history, config.search.gamma())

        model_path = os.path.join(self.output_folder, str(self.child_index), "model.h5")
        model.save(model_path)

        msg = (self.child_index, history.history)

        self.fitness_queue.put(msg)
