import argparse
import os
import sys

#os.environ["CUDA_VISIBLE_DEVICES"]="-1" # DEBUG: Process only on the CPU!

import mlflow
import tensorflow as tf #pylint: disable=C0413
from tensorflow import keras #pylint: disable=C0413

from delta import config #pylint: disable=C0413
from delta.imagery import imagery_dataset #pylint: disable=C0413
from delta.ml.train import Experiment


# Test out importing tarred Landsat images into a dataset which is passed
# to a training function.

def make_model(channel, in_len):
    # assumes square chunks.
#    fc1_size = channel * in_len ** 2
#     fc2_size = fc1_size * 2
#     fc3_size = fc2_size
#     fc4_size = fc1_size
    # To be consistent with Robert's poster
    fc2_size = 253
    fc3_size = 253
    fc4_size = 81

    dropout_rate = 0.3 # Taken from Robert's code.

    # Define network
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(in_len, in_len, channel)),
        # Note: use_bias is True by default, which is also the case in pytorch, which Robert used.
        keras.layers.Dense(fc2_size, activation=tf.nn.relu),
        keras.layers.Dropout(rate=dropout_rate),
        keras.layers.Dense(fc3_size, activation=tf.nn.relu),
        keras.layers.Dropout(rate=dropout_rate),
        keras.layers.Dense(fc4_size, activation=tf.nn.relu),
        keras.layers.Dropout(rate=dropout_rate),
        keras.layers.Dense(1, activation=tf.nn.sigmoid),
        ])
    return model

def init_network(num_bands, chunk_size):
    """Create a TF model to train"""

    model = make_model(num_bands, chunk_size)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001) # TODO
    model.compile(optimizer=optimizer, loss='mean_squared_logarithmic_error', metrics=['accuracy'])

    return model

def main(args):
    parser = argparse.ArgumentParser(usage='sc_process_test.py [options]')

    parser.add_argument("--config-file", dest="config_file", required=False,
                        help="Dataset configuration file.")

    parser.add_argument("--data-folder", dest="data_folder", required=False,
                        help="Specify data folder instead of supplying config file.")
    parser.add_argument("--label-folder", dest="label_folder", required=False,
                        help="Specify label folder instead of supplying config file.")

    parser.add_argument("--num-gpus", dest="num_gpus", required=False, default=0, type=int,
                        help="Try to use this many GPUs.")
    parser.add_argument("--test-limit", dest="test_limit", type=int, default=0,
                        help="If set, use a maximum of this many input values for training.")

    try:
        options = parser.parse_args(args[1:])
    except argparse.ArgumentError:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if (not options.config_file) and (not options.data_folder):
        parser.print_help(sys.stderr)
        print('Must specify either --config-file or --data-folder')
        sys.exit(1)

    config_values = config.parse_config_file(options.config_file, options.data_folder)
    if options.label_folder:
        config_values['input_dataset']['label_directory'] = options.label_folder
    batch_size = config_values['ml']['batch_size']
    num_epochs = config_values['ml']['num_epochs']
    output_folder = config_values['ml']['output_folder']

    # With TF 1.12, the dataset needs to be constructed inside a function passed in to
    # the estimator "train_and_evaluate" function to avoid getting a graph error!
    def assemble_dataset():

        # Use wrapper class to create a Tensorflow Dataset object.
        # - The dataset will provide image chunks and corresponding labels.
        ids = imagery_dataset.ImageryDatasetTFRecord(config_values)
        ds = ids.dataset()

        #print("Num regions = " + str(ids.total_num_regions()))
        #if ids.total_num_regions() < batch_size:
        #    raise Exception('Batch size (%d) is too large for the number of input regions (%d)!'
        #                    % (batch_size, ids.total_num_regions()))
        ds = ds.batch(batch_size)

        #dataset = dataset.shuffle(buffer_size=1000) # Use a random order
        ds = ds.repeat(num_epochs) # Need to be set here for use with train_and_evaluate

        if options.test_limit:
            ds = ds.take(options.test_limit)

        return ds

    print('Creating experiment')
    mlflow_tracking_dir = os.path.join(output_folder, 'mlruns')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(mlflow_tracking_dir):
        os.mkdir(mlflow_tracking_dir)
    experiment = Experiment(mlflow_tracking_dir,
                            'autoencoder_%s'%(config_values['input_dataset']['image_type'],),
                            output_dir=output_folder)
    mlflow.log_param('image type',   config_values['input_dataset']['image_type'])
    mlflow.log_param('image folder', config_values['input_dataset']['data_directory'])
    mlflow.log_param('chunk size',   config_values['ml']['chunk_size'])

    # Get these values without initializing the dataset (v1.12)
    ds_info = imagery_dataset.ImageryDatasetTFRecord(config_values)
    model = make_model(ds_info.num_bands(), config_values['ml']['chunk_size'])
    print('num images = ', ds_info.num_images())

    # Estimator interface requires the dataset to be constructed within a function.
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = experiment.train(model, assemble_dataset, steps_per_epoch=1000,
                                 log_model=False, num_gpus=options.num_gpus)
    #model = experiment.train_keras(model, assemble_dataset,
    #                               num_epochs=config_values['ml']['num_epochs'],
    #                               steps_per_epoch=150,
    #                               log_model=False, num_gpus=options.num_gpus)


    print('sc_process_test finished!')

if __name__ == "__main__":
    sys.exit(main(sys.argv))
