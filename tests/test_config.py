import os
import yaml

from delta.config import config
from delta.ml import model_parser

def test_general():
    config.reset()

    assert config.gpus() == -1

    test_str = '''
    general:
      gpus: 3
      threads: 5
      block_size_mb: 10
      interleave_images: 3
      tile_ratio: 1.0
      cache:
        dir: nonsense
        limit: 2
    '''
    config.load(yaml_str=test_str)

    assert config.gpus() == 3
    assert config.threads() == 5
    assert config.block_size_mb() == 10
    assert config.interleave_images() == 3
    assert config.tile_ratio() == 1.0
    cache = config.cache_manager()
    assert cache.folder() == 'nonsense'
    assert cache.limit() == 2

def test_images_dir():
    config.reset()
    test_str = '''
    images:
      type: tiff
      preprocess: false
      directory: data/
      extension: .tiff
    '''
    config.load(yaml_str=test_str)
    im = config.images()
    assert not im.preprocess()
    assert im.type() == 'tiff'
    assert len(im) == 1
    assert im[0].endswith('landsat.tiff') and os.path.exists(im[0])

def test_images_files():
    config.reset()
    test_str = '''
    images:
      type: tiff
      preprocess: false
      files: [data/landsat.tiff]
    '''
    config.load(yaml_str=test_str)
    im = config.images()
    assert not im.preprocess()
    assert im.type() == 'tiff'
    assert len(im) == 1
    assert im[0] == 'data/landsat.tiff'

def test_model_from_dict():
    config.reset()
    test_str = '''
    params:
        v1 : 10
    layers:
    - Flatten:
        input_shape: in_shape
    - Dense:
        units: v1
        activation : relu
    - Dense:
        units: out_shape
        activation : softmax
    '''

    input_shape = (17, 17, 8)
    output_shape = 3
    params_exposed = { 'out_shape' : output_shape, 'in_shape' : input_shape}
    model = model_parser.model_from_dict(yaml.safe_load(test_str), params_exposed)()
    model.compile(optimizer='adam', loss='mse')

    assert model.input_shape[1:] == input_shape
    assert model.output_shape[1] == output_shape
    assert len(model.layers) == 3

def test_network_file():
    config.reset()
    test_str = '''
    network:
      chunk_size: 5
      classes: 3
      model:
        yaml_file: networks/convpool.yaml
    '''
    config.load(yaml_str=test_str)
    assert config.chunk_size() == 5
    assert config.classes() == 3
    model = model_parser.config_model(2)()
    assert model.input_shape == (None, config.chunk_size(), config.chunk_size(), 2)
    assert model.output_shape == (None, config.output_size(), config.output_size(), config.classes())

def test_network_inline():
    config.reset()
    test_str = '''
    network:
      chunk_size: 5
      output_size: 1
      classes: 3
      model:
        params:
          v1 : 10
        layers:
        - Flatten:
            input_shape: in_shape
        - Dense:
            units: v1
            activation : relu
        - Dense:
            units: out_dims
            activation : softmax
    '''
    config.load(yaml_str=test_str)
    assert config.chunk_size() == 5
    assert config.classes() == 3
    model = model_parser.config_model(2)()
    assert model.input_shape == (None, config.chunk_size(), config.chunk_size(), 2)
    assert model.output_shape == (None, config.classes())

def test_train():
    config.reset()
    test_str = '''
    train:
      chunk_stride: 2
      batch_size: 5
      steps: 10
      epochs: 3
      loss_function: loss
      metrics: [metric]
      experiment_name: name
      optimizer: opt
      validation:
        steps: 20
        from_training: true
    '''
    config.load(yaml_str=test_str)
    tc = config.training()
    assert tc.chunk_stride == 2
    assert tc.batch_size == 5
    assert tc.steps == 10
    assert tc.epochs == 3
    assert tc.loss_function == 'loss'
    assert tc.metrics == ['metric']
    assert tc.experiment == 'name'
    assert tc.optimizer == 'opt'
    assert tc.validation.steps == 20
    assert tc.validation.from_training

def test_mlflow():
    config.reset()

    assert config.mlflow_enabled()

    test_str = '''
    mlflow:
      enabled: false
      uri: nonsense
      frequency: 5
      checkpoint_frequency: 10
    '''
    config.load(yaml_str=test_str)

    assert not config.mlflow_enabled()
    assert config.mlflow_uri() == 'nonsense'
    assert config.mlflow_freq() == 5
    assert config.mlflow_checkpoint_freq() == 10

def test_tensorboard():
    config.reset()

    assert not config.tb_enabled()

    test_str = '''
    tensorboard:
      enabled: false
      dir: nonsense
      frequency: 5
    '''
    config.load(yaml_str=test_str)

    assert not config.tb_enabled()
    assert config.tb_dir() == 'nonsense'
    assert config.tb_freq() == 5
