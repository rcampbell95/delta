import argparse
import os
import multiprocessing
import time
import copy

import pandas as pd

from tensorflow.keras.models import clone_model
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Activation, Reshape, Flatten, BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2DTranspose, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow as tf

from delta.imagery import imagery_dataset
from delta.config import config
import delta.config.modules
from delta.ml import train

def assemble_dataset():
    tc = config.train.spec()
    images = config.dataset.images()
    labels = config.dataset.labels()

    ids = imagery_dataset.ImageryDataset(images, labels,
                                         config.train.network.chunk_size(),
                                         config.train.network.output_size(),
                                         tc.chunk_stride)

    return ids

class Classifier(multiprocessing.Process):
    def __init__(self, train_spec):
        multiprocessing.Process.__init__(self)
        self.train_spec = train_spec

        self.strategy = self.train_spec.strategy

        self.model_path = self.train_spec.model_path
        self.model = None

        if self.strategy == 0 or self.strategy == 3:
            self.trainable = False
        else:
            self.trainable = True

        assert self.model_path is not None


    def build_model(self):
        if self.model is not None:
            print("Model already built and loaded in memory. Aborting operation")
        else:
            with model_file_lock:
                self.model = keras.models.load_model(self.model_path, compile=False)
            for idx, layer in enumerate(self.model.layers):
                # Checks for first layer of decoder
                if isinstance(layer, (Conv2DTranspose, GlobalAveragePooling2D)):
                    out_idx = idx - 2
                    break

            for layer in self.model.layers:
                layer.trainable = self.trainable

            x = Flatten(name="flatten")(self.model.layers[out_idx].output)
            x = BatchNormalization(name="batch_norm_1")(x)
             #x = GlobalAveragePooling2D(name="global_avg_pooling")(self.model.layers[out_idx].output)
             #x = Dense(380)(x)
             #x = Activation("relu", name="activation_penultimate")(x)
            x = Dense(64, name="dense_1", activation="relu")(x)
            x = BatchNormalization(name="batch_norm_2")(x)
            x = Dense(64, name="dense_2", activation="relu")(x)
            x = BatchNormalization(name="batch_norm_3")(x)
            
            x = Dense((config.train.network.output_size() ** 2) * 1)(x)
            x = Reshape((config.train.network.output_size(), config.train.network.output_size(), 1))(x)
            x = Activation("sigmoid", name="activation_out")(x)
            #x = Softmax(name="softmax_out", axis=-1)(x)

            model = Model(inputs=self.model.inputs, outputs=x)

            if self.strategy == 2 or self.strategy == 3:
                self.model = clone_model(model)
            else:
                self.model = model

            print(model.summary())
        return self.model

    def run(self):
        ids = assemble_dataset()

        self.train_spec.optimizer = tf.keras.optimizers.Adam()

        self.train_spec.metrics = ['accuracy',
                                   tf.keras.metrics.AUC(curve="PR", num_thresholds=1000),
                                   tf.keras.metrics.Precision(),
                                   tf.keras.metrics.Recall(),
                                   tf.keras.metrics.FalseNegatives()]

        #self.train_spec.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        self.train_spec.callbacks = [tf.keras.callbacks.ReduceLROnPlateau(factor=.9, verbose=1)]

        train.train(self.build_model, ids, self.train_spec)

delta.config.modules.register_all()
config.initialize(None)


usage  = "usage: test_classification.py [options]"
parser = argparse.ArgumentParser(usage=usage)

parser.add_argument("--data-folder", dest="data_folder", default=None,
                    help="Specify experiment folder in mlruns folder containing runs.")
parser.add_argument("--run-csv", dest="run_csv", default=None,
                    help="Downloaded csv from mlflow ui containing run information.")


options = parser.parse_args()

run_info = pd.read_csv(options.run_csv)
path = options.data_folder

run_id_key = "Run ID"

TRAINING_STRATEGIES = {
    0: "pretrain_train_dense",
    1: "pretrain_train_all",
    2: "no_pretrain_train_all",
    3: "no_pretrain_train_dense"
}

model_file_lock = multiprocessing.Lock()

for run_id in run_info[run_id_key]:

    row = run_info[run_info[run_id_key] == run_id]

    procs = []

    for strategy in TRAINING_STRATEGIES:

        print("\nTraining {}\n".format(run_id))

        model_path = os.path.join(path, "{}/artifacts/model.h5".format(run_id))
        print(model_path)
        assert os.path.exists(model_path)

        ts = copy.deepcopy(config.train.spec())

        ts.model_path = model_path
        ts.strategy = strategy

        if "Shape" in row and "Run ID" in row:
            ts.tags = {"Shape": row["Shape"].item(), 
                       "Encoder ID": row["Run ID"].item(),
                       "Gamma": row["Gamma"].item(),
                       "Strategy": TRAINING_STRATEGIES[strategy]}

        procs.append(Classifier(ts))

    for proc in procs:
        proc.start()
        #proc.join()
        time.sleep(10)
    for proc in procs:
        proc.join()
