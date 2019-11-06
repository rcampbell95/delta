import argparse
import os
import sys
import time
import math
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" # DEBUG: Process only on the CPU!

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
import matplotlib.pyplot as plt

import mlflow
import tensorflow as tf #pylint: disable=C0413
from tensorflow import keras #pylint: disable=C0413

from delta import config #pylint: disable=C0413
from delta.imagery import imagery_dataset #pylint: disable=C0413

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


parser = argparse.ArgumentParser(usage='predict_image.py [options]')

parser.add_argument("--config-file", dest="config_file", required=True,
                    help="Dataset configuration file.")
parser.add_argument("--model-path", dest="model_path", required=False,
                    help="Path to keras model for classification.")


try:
    options = parser.parse_args()
except argparse.ArgumentError:
    parser.print_help(sys.stderr)
    sys.exit(1)

config.load_config_file(options.config_file)
config_values = config.get_config()

# TODO: Read these from file!
height = 520
width = 1077
tile_size = 256 # TODO: Where to get this?
num_tiles_x = int(math.floor(width/tile_size))
num_tiles_y = int(math.floor(height/tile_size))
num_tiles = num_tiles_x*num_tiles_y


# Make a non-shuffled dataset for only one image
def make_evaluate_ds():
    ids = imagery_dataset.ImageryDatasetTFRecord(config_values)
    ds = ids.dataset(filter_zero=False, shuffle=False)
    ds = ds.batch(200)
    #ds = ds.repeat(None)

    return ds
    
def make_classify_ds():
    config_values['ml']['chunk_overlap'] = 0#int(config_values['ml']['chunk_size']) - 1 # TODO
    ids = imagery_dataset.ImageryDatasetTFRecord(config_values)
    ds = ids.dataset(filter_zero=False, shuffle=False, predict=True)
    ds = ds.batch(200)
    return ds

model = keras.models.load_model(options.model_path)
model.summary()
print(model.layers[-1].activation)
estimator = keras.estimator.model_to_estimator(model)

print('Classifying the image...')
start = time.time()
output_data = []

flood_count = 0
no_flood_count = 0

#metrics = estimator.evaluate(input_fn=make_evaluate_ds)

#print(metrics)

for pred in estimator.predict(input_fn=make_classify_ds):
    #print(pred)
    value = pred['dense_out']
    #print(value)
    
    value = 1 if value.item() > 0.5 else 0
    #value = np.where(value == max(value))[0]
    if value == 0:
        no_flood_count += 1
    else:
        flood_count += 1
    output_data.append(value)

print("flood count: ", flood_count)
print("Not flooded count: ", no_flood_count)

stop = time.time()
print('Output count = ' + str(len(output_data)))
print('Elapsed time = ' + str(stop-start))

# TODO: When actually classifying do not crop off partial tiles!
#       May need to use the old imagery dataset class to do this!
num_patches = len(output_data)
patches_per_tile = int(num_patches / num_tiles)
print('patches_per_tile = ' + str(patches_per_tile))
patch_edge = int(math.sqrt(patches_per_tile))
print('patch_edge = ' + str(patch_edge))

# Convert the single vector of prediction values into the shape of the image
# TODO: Account for the overlap value!
i = 0
pic = np.zeros([num_tiles_y*patch_edge, num_tiles_x*patch_edge], dtype=np.float32)
for ty in range(0,num_tiles_y):
    print(ty)
    for tx in range(0,num_tiles_x):
        row = ty*patch_edge
        for y in range(0,patch_edge): #pylint: disable=W0612
            col = tx*patch_edge
            for x in range(0,patch_edge): #pylint: disable=W0612
                pic[row, col] = output_data[i]
                i += 1
                col += 1
            row += 1

# Write the output image to disk
plt.subplot(1,1,1)
plt.imshow(pic)
plt.savefig(os.path.join(config_values["ml"]["output_folder"], "out.png"))

#jpeg = tf.image.encode_jpeg(pic, quality=100, format='grayscale')
#output_path = '/home/smcmich1/repo/delta/output.jpg'
#writer = tf.write_file(output_path, jpeg)
#sess.run(writer)

print('sc_process_test finished!')
