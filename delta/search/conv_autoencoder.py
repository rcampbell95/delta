from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Activation, SpatialDropout2D, Dropout
from tensorflow.keras.layers import Dense, Reshape, GlobalAveragePooling2D
from tensorflow.keras.models import Model

import tensorflow as tf

from delta.search.genetics import Gene, Genotype

from delta.config import config

def define_gene(self):
    self.params = {"filter_size": (4, 8, 16, 32, 64, 128),
                   "kernel_size": (1, 3, 5, 7),
                   "regularization": ("spatial_dropout", "dropout"),
                   "dropout_rate": (0.0, 0.1, 0.3, 0.5),
                   "activation": ("selu", "relu", "tanh"),
                   "alpha": (0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5)}
    return self.params

Gene.define_gene = define_gene

class ConvAutoencoderGenotype(Genotype):
    def build_model(self, input_shape):
        OUT_CHANNELS = input_shape[-1]
        OUT_DIMS = input_shape[1]
        POOL_STRIDE = 2
        OUT_KERNEL_SIZE = 1

        ae_shape = config.autoencoder_shape()

        coding_sequence = self.trace_encoder()
        output_gene = coding_sequence[0]
        coding_sequence = coding_sequence[1:]

        inputs = Input(shape=input_shape)
        x = inputs

        # Build encoder
        for gene in reversed(coding_sequence):
            kernel_size = (gene.attrs["kernel_size"], gene.attrs["kernel_size"])

            x = Conv2D(filters = gene.attrs["filter_size"],
                       kernel_size = kernel_size,
                       strides = (POOL_STRIDE, POOL_STRIDE),
                       padding = "same",
                       activity_regularizer=tf.keras.regularizers.l1(output_gene.attrs["alpha"]))(x)

            x = Activation(gene.attrs["activation"])(x)

            if gene.attrs["regularization"] == "spatial_dropout":
                x = SpatialDropout2D(rate=gene.attrs["dropout_rate"])(x)
            elif gene.attrs["regularization"] == "dropout":
                x = Dropout(rate=gene.attrs["dropout_rate"])(x)

        # Build decoder
        if ae_shape == "symmetric":
            for gene in coding_sequence:
                kernel_size = (gene.attrs["kernel_size"], gene.attrs["kernel_size"])

                x = Conv2DTranspose(filters=gene.attrs["filter_size"],
                                    kernel_size=kernel_size,
                                    strides=(POOL_STRIDE, POOL_STRIDE),
                                    padding="same")(x)

                x = Activation(gene.attrs["activation"])(x)

                if gene.attrs["regularization"] == "spatial_dropout":
                    x = SpatialDropout2D(rate=gene.attrs["dropout_rate"])(x)
                elif gene.attrs["regularization"] == "dropout":
                    x = Dropout(rate=gene.attrs["dropout_rate"])(x)

            # Define output deconvolutional layer
            output = Conv2DTranspose(filters=OUT_CHANNELS,
                                     kernel_size=OUT_KERNEL_SIZE,
                                     name="output",
                                     activation="sigmoid")(x)
        elif ae_shape == "asymmetric":
            x = GlobalAveragePooling2D()(x)
            x = Dense(OUT_DIMS ** 2 * OUT_CHANNELS)(x)
            x = Activation("sigmoid")(x)
            output = Reshape(input_shape)(x)

        model = Model(inputs=inputs, outputs=output)

        print(model.summary())

        return model
