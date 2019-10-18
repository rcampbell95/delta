import os
import sys
import argparse
from find_network import find_network
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from delta.imagery import imagery_dataset 
from delta import config 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def assemble_dataset_for_predict(config_values):
    # Slightly simpler version of the previous function
    try:
        ids = imagery_dataset.AutoencoderDataset(config_values)
    except:
        print("Cannot create autoencoder dataset")
    ds  = ids.dataset().batch(1) # Batch needed to match the original format
    return ds

def get_debug_bands(image_type):
    '''Pick the best bands to use for debug images'''
    bands = [0]
    if image_type == 'worldview':
        # TODO: Distinguish between WV2 and WV3
        bands = [4,2,1] # RGB
    # TODO: Support more sensors
    return bands

usage  = "usage: debug_images [options]"
parser = argparse.ArgumentParser(usage=usage)
parser = argparse.ArgumentParser(usage='debug_images.py [options]')

parser.add_argument("--config-file", dest="config_file", default=None,
                    help="Dataset configuration file.")
parser.add_argument("--data-folder", dest="data_folder", default=None,
                    help="Specify data folder instead of supplying config file.")
parser.add_argument("--image-type", dest="image_type", default=None,
                    help="Specify image type along with the data folder."
                    +"(landsat, landsat-simple, worldview, or rgba)")
parser.add_argument("--num-debug-images", dest="num_debug_images", default=0, type=int,
                        help="Run this many images through the AE after training and write the "
                        "input/output pairs to disk.")


options = parser.parse_args()

config_values = config.parse_config_file(options.config_file,
                                             options.data_folder, options.image_type)



# path = config_values["ml"]["output_folder"] + "/" + config_values["ml"]["model_dest_name"]
path = "./mlruns/0/7f451edb1fa6476ba92165938abbf57d/artifacts/autoencoder_model.h5"
model = tf.keras.models.load_model(path)

# Make a non-shuffled dataset with a simple iterator

print(config_values["input_dataset"]["shuffle_buffer_size"])
ids = imagery_dataset.AutoencoderDataset(config_values)
ds = ids.dataset(filter_zero=False).batch(1)
iterator = ds.make_one_shot_iterator()
next_element = iterator.get_next()
  
debug_bands = get_debug_bands(config_values['input_dataset']['image_type'])
#print('debug_bands = ' + str(debug_bands))
scale = ids.scale_factor()
sess = tf.Session()

for i in range(0, options.num_debug_images):
    print('Preparing debug image ' + str(i))

    value = sess.run(next_element)

    # Get the output value out of its weird format, then convert for image output
    # Code to test with Keras instead of Estimator
    result = model.predict(value[0])
    pic = (result[0, :, :, debug_bands] * scale).astype(np.float32)
    pic = np.moveaxis(pic, 0, -1)

    #print(pic)
    plt.subplot(1,2,1)
    #print(value[0].shape)
    in_pic = (value[0][0, :, :, debug_bands] * scale).astype(np.float32)
    in_pic = np.moveaxis(in_pic, 0, -1) # Not sure why this is needed
    #print('data')
    #print(in_pic.shape)
    #print(in_pic)
    plt.imshow(in_pic.squeeze())
    plt.title('Input image %03d' % (i, ))

    plt.subplot(1,2,2)
    plt.imshow(pic.squeeze())
    plt.title('Output image %03d' % (i, ))

    #in_pic2 = (value[1][0,debug_bands,:,:] * scale).astype(np.uint8)
    #in_pic2 = np.moveaxis(in_pic2, 0, -1)
    #print('label')
    #print(in_pic2.shape)
    #print(in_pic2)

    debug_image_filename = os.path.join(config_values["ml"]["output_folder"],
                                        'Autoencoder_input_output_%03d.png' % (i, ))
    plt.savefig(debug_image_filename)
