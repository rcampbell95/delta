#from utils import normalize
import scipy.io as sio
from tensorflow.keras.utils import to_categorical
import numpy as np
#from sklearn.metrics import classification_report
#from sklearn.metrics import accuracy_score
import pandas as pd
import mlflow
import argparse
import sys
import os

import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from delta.imagery import imagery_dataset
from delta import config


from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import clone_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Reshape, Flatten, Activation, Input, Conv2D, Conv2DTranspose, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
import tensorflow as tf

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import multiprocessing
import threading

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

def log_params(config_values):
    #import time

    for category in config_values.keys():
        for param in config_values[category].keys():
            mlflow.log_param(param, config_values[category][param])
 
    
class MetricHistory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        import mlflow

        for metric in self.model.metrics_names:
            train_metric = float(logs.get(metric))
            val_metric = float(logs.get("val_" + metric))

            mlflow.log_metric(metric, train_metric, step=epoch)
            mlflow.log_metric("val_" + metric, val_metric, step=epoch)

class ReloadBestModel(tf.keras.callbacks.Callback):
    def __init__(self, model_folder):
        self.model_folder = model_folder

    def on_train_end(self, logs={}):
        #print("Loading best model saved by checkpoint...")
        self.model = keras.models.load_model(self.model_folder + "/model.h5")

class Classifier(multiprocessing.Process):
    def __init__(self, run_info):
        multiprocessing.Process.__init__(self)
        self.run_info = run_info

        self.strategy = self.run_info["strategy"]
        self.run_info["steps_per_epoch"] = 10

        self.model_path = self.run_info["model_path"]
        assert(self.model_path is not None)


    def build_model(self, trainable):
        if hasattr(self, "model"):
            print("Model already built and loaded in memory. Aborting operation")
        else:
            model_file_lock.acquire()
            self.model = keras.models.load_model(self.model_path)
            model_file_lock.release()

            for idx, layer in enumerate(self.model.layers):
                if isinstance(layer, Conv2DTranspose) or isinstance(layer, Flatten):
                    out_idx = idx - 2
                    break

            for layer in self.model.layers:
                layer.trainable = trainable 

            x = Flatten(name="flatten")(self.model.layers[out_idx].output)
            x = Dense(1, activation="sigmoid", name="dense_out")(x)
            
            
            model = Model(inputs=self.model.inputs, outputs=x)

            if self.strategy == 2 or self.strategy == 3:
                self.model = clone_model(model)
            else:
                self.model = model
            #self.model = Model(inputs=self.model.inputs, outputs=x)
            print(self.model.summary()) 

    def load_data(self):
        image_type = self.config_values["input_dataset"]["image_type"]

        if image_type == "svhn":
            train_data = sio.loadmat('train_32x32.mat')
            test_data = sio.loadmat('test_32x32.mat')
            x_train = train_data["X"]
            y_train = train_data["y"]
            x_test = test_data["X"]
            y_test = test_data["y"]

            x_train = np.transpose(x_train, (3, 0, 1, 2))
            x_test = np.transpose(x_test, (3, 0, 1, 2))

            y_train = to_categorical(y_train - 1)
            y_test = to_categorical(y_test - 1)

        elif image_type == "cifar":
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            
            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)

        elif image_type == "landsat":
            train_ds = self.__assemble_dataset()
            val_directory = self.config_values['input_dataset']['val_directory']
            train_directory = self.config_values['input_dataset']['data_directory']

            self.config_values['input_dataset']['data_directory'] = val_directory
            
            val_ds = self.__assemble_dataset()
            self.config_values['input_dataset']['data_directory'] = train_directory

            return train_ds, val_ds
        
        # Add svhn and cifar support
        #x_train, x_test = normalize(x_train), normalize(x_test)
        return (x_train, y_train) , (x_test, y_test)

    def __assemble_dataset(self):
        batch_size = self.config_values["ml"]["batch_size"]
        num_epochs = self.config_values["ml"]["num_epochs"]

        ids = imagery_dataset.ImageryDatasetTFRecord(self.config_values)
        ds = ids.dataset(filter_zero=False)

        #print("Num regions = " + str(ids.total_num_regions()))
        #if ids.total_num_regions() < batch_size:
        #    raise Exception('Batch size (%d) is too large for the number of input regions (%d)!'
        #                    % (batch_size, ids.total_num_regions()))
        ds = ds.batch(batch_size)

        #dataset = dataset.shuffle(buffer_size=1000) # Use a random order
        ds = ds.repeat(None) # Need to be set here for use with train_and_evaluate

        #if options.test_limit:
        #    ds = ds.take(options.test_limit)

        return ds

  

    def __get_callbacks(self, model_folder):
        callbacks = []

        #log_dir="./logs/random_feature_extractor/{}/{}".format(self.run_info["encoder_run_id"], TRAINING_STRATEGIES[self.strategy])
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


        #callbacks.append(tensorboard_callback)

        self.checkpoint_path = model_folder + "/" + TRAINING_STRATEGIES[self.strategy] + '_model.h5'
        callbacks.append(ReloadBestModel(model_folder))
        callbacks.append(ModelCheckpoint(self.checkpoint_path, monitor='val_loss', 
                                    mode='min', save_best_only=True, verbose=0))
    
        callbacks.append(MetricHistory())    
        return callbacks

    def train_model(self, trainset, valset, steps_per_epoch):
        import math

        validation_split = .8
        model_folder = self.config_values["ml"]["model_folder"]
        epochs = self.config_values["ml"]["num_epochs"]
        base_lr = 0.001

        callbacks = self.__get_callbacks(model_folder)

        optimizer = keras.optimizers.Adam(lr=base_lr, beta_1=0.9, beta_2=0.999, amsgrad=False)

        self.model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['binary_accuracy',
                         tf.keras.metrics.Precision(),
                         tf.keras.metrics.Recall()])

        history = self.model.fit(x = trainset,
                                 epochs=epochs, 
                                 batch_size = None,
                                 steps_per_epoch = math.ceil(steps_per_epoch * validation_split),
                                 validation_data = valset, 
                                 shuffle=True, 
                                 validation_steps = math.ceil(steps_per_epoch * (1 - validation_split)),
                                 callbacks=callbacks,
                                 verbose=1)

        if self.strategy == 0:
            # Fine tune model
            self.model.trainable = True 
            fine_tune_epochs = 10

            optimizer = keras.optimizers.Adam(lr=base_lr, beta_1=0.9, beta_2=0.999, amsgrad=False)


            self.model.compile(optimizer=optimizer,
                               loss='binary_crossentropy',
                               metrics=['binary_accuracy',
                               tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()])

            self.model.fit(x = trainset,
                           epochs = epochs + fine_tune_epochs,
                           initial_epoch = epochs,
                           batch_size = None,
                           steps_per_epoch = math.ceil(steps_per_epoch * validation_split),
                           validation_data = valset,
                           shuffle = True,
                           validation_steps = math.ceil(steps_per_epoch * (1 - validation_split)),
                           callbacks=callbacks,
                           verbose=1)

            self.model.save(self.checkpoint_path)

        mlflow.log_artifact(self.checkpoint_path)
            
    def test_model(self, x_test, y_test):
        import mlflow
        if self.model is None:
            print("Error. Train model before evaluating.")

        accuracy = self.model.evaluate(x=x_test, y=y_test, batch_size=batch_size)
    
        for idx, metric in enumerate(self.model.metrics_names):
            mlflow.log_metric("test_" + metric, accuracy[idx])

    def run(self):        
        with threadLimiter:
            import mlflow
            from delta import config

            ident = str(self.pid)
            print("Starting process " + ident)
            config_file_lock.acquire()

            config.load_config_file(options.config_file)
            self.config_values = config.get_config()

            
            with mlflow.start_run():
                log_params(self.config_values)
                mlflow.log_param("encoder_run_id", self.run_info["encoder_run_id"])
                mlflow.log_param("train_setup", TRAINING_STRATEGIES[self.strategy])
                mlflow.log_param("steps_per_epoch", self.run_info["steps_per_epoch"])
                #mlflow.log_param("shape", self.run_info["shape"])

                config_file_lock.release()

                trainset, valset = self.load_data()

                if self.strategy == 0 or self.strategy == 3:
                    trainable = False
                else:
                    trainable = True

                self.build_model(trainable=trainable) 
            
                self.train_model(trainset, valset, self.run_info["steps_per_epoch"])

                # TODO -- add artifact logging

                tf.keras.backend.clear_session()
                print("Process {} completed".format(ident))

usage  = "usage: test_classification.py [options]"
parser = argparse.ArgumentParser(usage=usage)


parser.add_argument("--config-file", dest="config_file", default=None,
                    help="Dataset configuration file.")
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

POOL_SIZE = 3

model_file_lock = multiprocessing.Lock()
config_file_lock = multiprocessing.Lock()
threadLimiter = multiprocessing.BoundedSemaphore(POOL_SIZE)

config.load_config_file(options.config_file)
config_values = config.get_config()

TRACKING_URI = "mysql+pymysql://{}:{}@{}:{}/{}".format(config_values["backend_store"]["user"], 
                                                       config_values["backend_store"]["password"], 
                                                       config_values["backend_store"]["endpoint"],
                                                       config_values["backend_store"]["port"],
                                                       config_values["backend_store"]["tablename"])

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(config_values["ml"]["experiment"])

procs = []
for run_id in run_info[run_id_key]:
    #run_row = run_info[run_info[run_id_key] == run_id]

    for strategy in TRAINING_STRATEGIES:

        print("\nTraining {}\n".format(run_id))
        
        try:
            model_path = path + run_id + "/artifacts/model.h5"
            assert(os.path.exists(model_path))
        except:
            model_path = path + run_id + "/artifacts/autoencoder_model.h5"

        # TODO -- Make  process pool class?
        # TODO -- Block main process until all worker processes have been created
        # Uses three processes, one per model type
        run_info = {
            "model_path" : model_path,
            "strategy" : strategy,
            "encoder_run_id" : run_id,
            #"shape" : run_row["shape"].item()
        }


        procs.append(Classifier(run_info))
        procs[-1].start()
