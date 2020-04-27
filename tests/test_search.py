import argparse
import os
import multiprocessing

import pandas as pd

from tensorflow.keras.models import clone_model
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Activation, Reshape, Flatten
from tensorflow.keras.layers import Conv2DTranspose, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow as tf

from delta.imagery import imagery_dataset
from delta.config import config
from delta.ml import train

def assemble_dataset():
    tc = config.training()
    images = config.images()
    labels = config.labels()

    ids = imagery_dataset.ImageryDataset(images, labels,
                                         config.chunk_size(),
                                         config.output_size(),
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
        if hasattr(self, "model"):
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
            #x = GlobalAveragePooling2D(name="global_avg_pooling")(self.model.layers[out_idx].output)
            x = Dense(config.output_size() ** 2)(x)
            x = Reshape((config.output_size(), config.output_size(), 1))(x)
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

        self.train_spec.metrics = ['accuracy',
                                   tf.keras.metrics.Precision(),
                                   tf.keras.metrics.Recall()]

        train.train(self.build_model, ids, self.train_spec)

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

        ts = config.training()

        if "Shape" in row and "Run ID" in row:
            ts.tags = {"shape": row["Shape"].item(), "encoder_id": row["Run ID"].item()}

        ts.model_path = model_path
        ts.strategy = strategy

        procs.append(Classifier(ts))

    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
