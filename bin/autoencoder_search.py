import os
import sys
import argparse
from find_network import find_network
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from delta import config 

import tensorflow as tf
import numpy as np

usage  = "usage: search_landsat [options]"
parser = argparse.ArgumentParser(usage=usage)

parser = argparse.ArgumentParser(usage='search_landsat.py [options]')

parser.add_argument("--config-file", dest="config_file", default=None,
                    help="Dataset configuration file.")
parser.add_argument("--data-folder", dest="data_folder", default=None,
                    help="Specify data folder instead of supplying config file.")
parser.add_argument("--image-type", dest="image_type", default=None,
                    help="Specify image type along with the data folder."
                    +"(landsat, landsat-simple, worldview, or rgba)")
options = parser.parse_args()

config.load_config_file(options.config_file)
config_values = config.get_config()

find_network(config_values)
