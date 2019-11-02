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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from delta.imagery import imagery_dataset


from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import clone_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Reshape, Flatten, Activation, Input, Conv2D, Conv2DTranspose, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
import tensorflow as tf

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
            mlflow.log_metric(metric, logs.get(metric), step=epoch)
            mlflow.log_metric("val_" + metric, logs.get("val_" + metric), step=epoch)

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
        self.run_info["steps_per_epoch"] = 1
        #self.config_values = config_values
        #if model_template:
        #    try:
        #        self.model = clone_model(model_template.model)
        #        #print(self.model_path)
        #    except:
        #        print("Model clone cannot be instatiated.")
        #else:
        self.model_path = self.run_info["model_path"]
        assert(self.model_path is not None)


    def build_model(self, trainable=True):
        if hasattr(self, "model"):
            print("Model already built and loaded in memory. Aborting operation")
        else:
            model_file_lock.acquire()
            self.model = keras.models.load_model(self.model_path)
            model_file_lock.release()

            for idx, layer in enumerate(self.model.layers):
                if isinstance(layer, Conv2DTranspose) or isinstance(layer, MaxPooling2D):
                    out_idx = idx - 2
                    break

            for layer in self.model.layers:
                layer.trainable = trainable 

            x = Flatten(name="flatten_out")(self.model.layers[out_idx].output)
            x = Dense(2, activation="softmax", name="dense_out")(x)
            
            
            model = Model(inputs=self.model.inputs, outputs=x)

            if self.strategy == 2:
                self.model = clone_model(model)
            else:
                self.model = model
            #self.model = Model(inputs=self.model.inputs, outputs=x)
            #print(self.model.summary()) 

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
            #read_lock.acquire()
            train_ds = self.__assemble_dataset()
            val_directory = self.config_values['input_dataset']['val_directory']
            train_directory = self.config_values['input_dataset']['data_directory']

            self.config_values['input_dataset']['data_directory'] = val_directory
            
            val_ds = self.__assemble_dataset()
            #read_lock.release()
            self.config_values['input_dataset']['data_directory'] = train_directory

            return train_ds, val_ds
            
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
        #callbacks.append(EarlyStopping(monitor='val_loss', mode='min', 
        #                               verbose=1, patience=epochs // 10,
        #                               restore_best_weights=True))
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

        callbacks = self.__get_callbacks(model_folder)

        self.model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['sparse_categorical_accuracy'])

        history = self.model.fit(x = trainset,
                                 epochs=epochs, 
                                 batch_size = None,
                                 steps_per_epoch = math.ceil(steps_per_epoch * validation_split),
                                 validation_data = valset, 
                                 shuffle=True, 
                                 validation_steps = math.ceil(steps_per_epoch * (1 - validation_split)),
                                 callbacks=callbacks,
                                 verbose=0)

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

            #print(self.config_values)
            
            with mlflow.start_run():
                log_params(self.config_values)
                #mlflow.log_params(self.config_values)
                mlflow.log_param("encoder_run_id", self.run_info["encoder_run_id"])
                mlflow.log_param("train_setup", TRAINING_STRATEGIES[self.strategy])
                mlflow.log_param("steps_per_epoch", self.run_info["steps_per_epoch"])
                
                config_file_lock.release()

                trainset, valset = self.load_data()

                trainable = False if self.strategy == 0 else True
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

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("log_batch_classification")

run_id_key = "Run ID"

TRAINING_STRATEGIES = {
    0: "pretrain_train_dense",
    1: "pretrain_train_all", 
    2: "no_pretrain_train_all"
}

POOL_SIZE = 3

model_file_lock = multiprocessing.Lock()
config_file_lock = multiprocessing.Lock()
threadLimiter = multiprocessing.BoundedSemaphore(POOL_SIZE)

procs = []
for run_id in run_info[run_id_key]:
    for strategy in TRAINING_STRATEGIES:
        print("\nTraining {}\n".format(run_id))
        
        try:
            model_path = path + run_id + "/artifacts/model.h5"
            assert(os.path.exists(model_path))
        except:
            model_path = path + run_id + "/artifacts/autoencoder_model.h5"

        # TODO -- Make  thread pool class?
        # TODO -- Block main process all worker processes have been created
        # Uses three threads, one per model type
        run_info = {
            "model_path" : model_path,
            "strategy" : strategy,
            "encoder_run_id" : run_id
        }


        procs.append(Classifier(run_info))
        procs[-1].start()

        #for thread in threads:
        #threads[-1].join()

        ### Serial
        # model = Classifier(model_path, strategy)
        # model.run()

        # run_row = run_info[run_info[run_id_key] == run_id]
        # print("\n\nTraining {}\n".format(run_id))

        
        # try:
        #     model_path = path + run_id + "/artifacts/model.h5"
        #     assert(os.path.exists(model_path))
        # except:
        #     model_path = path + run_id + "/artifacts/autoencoder_model.h5"

        # # Load and preprocess data data
        # #     
        # ####################
        # # Train and test model with locked features
        # ####################
        # mlflow.start_run()
        # log_params(config_values)
        # mlflow.log_param(run_id_key, run_id)
        # mlflow.log_param("train_setup", "pretrain_train_dense")
        
        # model = Classifier(config_values, model_path)
        # trainset, valset = model.load_data()

        # model.build_model(trainable=False)

        # model.train_model(trainset, valset)

        # #model.test_model(x_test, y_test)

        # tf.keras.backend.clear_session()
        # mlflow.end_run()

        # #################
        # # Train and test end to end with pretrained featuress
        # #################

        # mlflow.start_run()
        # log_params(config_values)
        # mlflow.log_param("run_id", run_id)
        # mlflow.log_param("train_setup", "pretrain_train_all")    
        
        # model = Classifier(model_path)
        # trainset, valset = model.load_data()

        # model.build_model(trainable=True)

        # print("\n\nTrain model with extracted encoder\n\n")
        
        # model.train_model(trainset, valset)

        # #model.test_model(x_test, y_test)

        # tf.keras.backend.clear_session()
        # mlflow.end_run()

        # #################
        # # Train model end-to-end without pretrained features
        # #################    

        # print("\n\nTrain end to end model\n\n")
        # mlflow.start_run()
        # log_params(config_values)
        # mlflow.log_param(run_id_key, run_id)
        # mlflow.log_param("train_setup", "no_pretrain_train_all")
            
        # baseline = Classifier(model_template=model)
        # trainset, valset = model.load_data()   


        # baseline.train_model(trainset, valset)    

        # #baseline.test_model(x_test, y_test)

        # tf.keras.backend.clear_session()
        # mlflow.end_run()
