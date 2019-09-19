import numpy as np

def define_gene(self):
    self.params = {"filter_size": (16, 32, 64, 128),
                    "kernel_size": (1, 2, 3, 4, 5),
                    "skip": (True, False),
                    "regularization": ("spatial_dropout", "dropout"),
                    "dropout_rate": (0, 0.3, 0.5),
                    "activation": ("selu", "relu", "tanh"),
                    "pooling": (True, False),
                    "output": ("dense", "transpose") }
    
    self.attrs = {"filter_size": 64, 
                    "kernel_size": 3, 
                    "skip": False, 
                    "regularization": "spatial_dropout", 
                    "dropout_rate": 0, 
                    "activation": "relu",
                    "pooling": False,
                    "output": "dense"}


def build_model(self, input_shape, shape, random_seed):
    from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Activation, Add, SpatialDropout2D, Dropout
    from tensorflow.keras.layers import Flatten, Dense, Reshape, MaxPooling2D, UpSampling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.utils import multi_gpu_model

    import tensorflow as tf

    tf.random.set_random_seed(random_seed)
    
    # Define input convolutional layer
    out_channels = input_shape[0]
    pool_size = 2
    out_kernel_size = 1

    inputs = Input(shape=input_shape)
    x = inputs
    
    coding_sequence = self.trace_encoder()
    coding_sequence = coding_sequence[1:]
    
    skip_values = []
    
    # Build encoder
    for idx, gene in enumerate(reversed(coding_sequence)):
        kernel_size = (gene.attrs["kernel_size"], gene.attrs["kernel_size"])

        x = Conv2D(filters = gene.attrs["filter_size"], kernel_size = kernel_size, 
                    padding = "same")(x)

        #x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)
        x = Activation(gene.attrs["activation"])(x)

        if gene.attrs["regularization"] == "spatial_dropout":
            x = SpatialDropout2D(rate=gene.attrs["dropout_rate"])(x)
        elif gene.attrs["regularization"] == "dropout":
            x = Dropout(rate=gene.attrs["dropout_rate"])(x)
        
        if gene.attrs["skip"]:
            skip_values.append(x)


    # Build decoder
    if shape == "symmetric":
        for idx, gene in enumerate(coding_sequence):
            kernel_size = (gene.attrs["kernel_size"], gene.attrs["kernel_size"])

            
            #x = UpSampling2D(size=(pool_size, pool_size))(x)
            x = Conv2DTranspose(filters=gene.attrs["filter_size"], 
                                kernel_size=kernel_size, 
                                padding="same")(x)
            
            if gene.attrs["skip"]:
                try:
                    x = Add()([skip_values.pop(), x])
                except:
                    print(gene.node_id)
            x = Activation(gene.attrs["activation"])(x)

            if gene.attrs["regularization"] == "spatial_dropout":
                x = SpatialDropout2D(rate=gene.attrs["dropout_rate"])(x)
            elif gene.attrs["regularization"] == "dropout":
                x = Dropout(rate=gene.attrs["dropout_rate"])(x)

    # Add decoder layer for asymmetric autoencoder or add output layer for symmetric autoencoder
    if shape == "symmetric" or coding_sequence[0].attrs["output"] == "transpose":
        # Define output deconvolutional layer
        output = Conv2DTranspose(filters=out_channels, kernel_size=out_kernel_size, name="output")(x)
    # Add decoder for flatten + dense asymmetric autoencoder
    elif coding_sequence[0].attrs["output"] == "dense" and shape == "asymmetric":
        # Add maxpooling to reduce number of parameters when for flatten + dense 
        x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Dense(input_shape[-1] ** 2 * input_shape[0])(x)
        output = Reshape(input_shape)(x)

    model = Model(inputs=inputs, outputs=output)

    print(model.summary())

    try:
        model = multi_gpu_model(model, cpu_relocation=True)
        print("Training using multiple GPUs")
    except:
        print("Training using CPU or single GPU")

    model.compile(optimizer="adam",
            loss='mse')
    
    return model
