import numpy as np
from genetics import Gene, Genotype

def define_gene(self):
    self.params = {"filter_size": (16, 32, 64, 128),
                    "kernel_size": (1, 3, 5, 7),
                    "skip": (True, False),
                    "regularization": ("spatial_dropout", "dropout"),
                    "dropout_rate": (0.0, 0.1, 0.3, 0.5),
                    "activation": ("selu", "relu", "tanh"),
                    "output": ("dense", "transpose") }

Gene.define_gene = define_gene

class ConvAutoencoderGenotype(Genotype):
    def __init__(self, config_values, genes=False):
        super(ConvAutoencoderGenotype, self).__init__(config_values, genes)


    def build_model(self, config_values, input_shape):
        from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Activation, Add, SpatialDropout2D, Dropout
        from tensorflow.keras.layers import Flatten, Dense, Reshape, MaxPooling2D, UpSampling2D
        from tensorflow.keras.layers import AlphaDropout
        from tensorflow.keras.models import Model
        from tensorflow.keras.utils import multi_gpu_model

        import tensorflow as tf
        
        # Define input convolutional layer
        out_channels = int(config_values["ml"]["channels"])
        out_dims = input_shape[1]
        pool_size = 2
        out_kernel_size = 1
        shape = config_values["evolutionary_search"]["shape"]

        inputs = Input(shape=input_shape)
        x = inputs
        
        coding_sequence = self.trace_encoder()
        coding_sequence = coding_sequence[1:]
        num_nodes = len(coding_sequence)
        
        skip_values = []
        
        # Build encoder
        for idx, gene in enumerate(reversed(coding_sequence)):
            kernel_size = (gene.attrs["kernel_size"], gene.attrs["kernel_size"])

            x = Conv2D(filters = gene.attrs["filter_size"], kernel_size = kernel_size, 
                        padding = "same")(x)

            x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)
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

                
                x = UpSampling2D(size=(pool_size, pool_size))(x)
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
                elif gene.attrs["regularization"] == "alpha_dropout":
                    x = AlphaDropout(rate=gene.attrs["dropout_rate"])(x)

        # Add decoder layer for asymmetric autoencoder or add output layer for symmetric autoencoder
        if shape == "symmetric" or coding_sequence[0].attrs["output"] == "transpose":
            # Define output deconvolutional layer
            if shape == "symmetric":
                strides = (1, 1)
            else:
                strides = (2 ** num_nodes, 2 ** num_nodes)

            output = Conv2DTranspose(filters=out_channels, 
                                     kernel_size=out_kernel_size, 
                                     strides=strides, 
                                     name="output")(x)
        # Add decoder for flatten + dense asymmetric autoencoder
        elif coding_sequence[0].attrs["output"] == "dense" and shape == "asymmetric":
            # Add maxpooling to reduce number of parameters when for flatten + dense 
            x = Flatten()(x)
            x = Dense(out_dims ** 2 * out_channels)(x)
            output = Reshape(input_shape)(x)

        model = Model(inputs=inputs, outputs=output)

        print(model.summary())

        if int(config["evolutionary_search"]["num_children"]) == 1:
            try:
                model = multi_gpu_model(model, gpus=config_values["ml"]["num_gpus"])
                print("Training using multiple GPUs")
            except:
                print("Training using CPU or single GPU")

            model.compile(optimizer="adam",
                    loss='mse')
            
        return model
