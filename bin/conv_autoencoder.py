from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Activation, SpatialDropout2D, Dropout
from tensorflow.keras.layers import Dense, Reshape, GlobalAveragePooling2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

from genetics import Gene, Genotype


from delta.config import config

def define_gene(self):
    self.params = {"filter_size": (4, 8, 16, 32, 64, 128),
                   "kernel_size": (1, 3, 5, 7),
                   "regularization": ("spatial_dropout", "dropout"),
                   "dropout_rate": (0.0, 0.1, 0.3, 0.5),
                   "activation": ("selu", "relu", "tanh"),
                   "output": ("dense", "transpose")}

    return self.params

Gene.define_gene = define_gene

class ConvAutoencoderGenotype(Genotype):
    def build_model(self, input_shape):
        out_channels = input_shape[-1]
        out_dims = input_shape[1]
        #pool_size = 2
        out_kernel_size = 1
        shape = config.model_shape()

        inputs = Input(shape=input_shape)
        x = inputs

        coding_sequence = self.trace_encoder()
        coding_sequence = coding_sequence[1:]

        # Build encoder
        for gene in reversed(coding_sequence):
            kernel_size = (gene.attrs["kernel_size"], gene.attrs["kernel_size"])

            x = Conv2D(filters = gene.attrs["filter_size"], kernel_size = kernel_size,
                       padding = "same")(x)

            x = MaxPooling2D(pool_size=2)(x)

            x = Activation(gene.attrs["activation"])(x)

            if gene.attrs["regularization"] == "spatial_dropout":
                x = SpatialDropout2D(rate=gene.attrs["dropout_rate"])(x)
            elif gene.attrs["regularization"] == "dropout":
                x = Dropout(rate=gene.attrs["dropout_rate"])(x)

        # Build decoder
        if shape == "symmetric":
            for gene in coding_sequence:
                kernel_size = (gene.attrs["kernel_size"], gene.attrs["kernel_size"])

                x = UpSampling2D(size=2)(x)

                x = Conv2DTranspose(filters=gene.attrs["filter_size"],
                                    kernel_size=kernel_size,
                                    padding="same")(x)

                x = Activation(gene.attrs["activation"])(x)

                if gene.attrs["regularization"] == "spatial_dropout":
                    x = SpatialDropout2D(rate=gene.attrs["dropout_rate"])(x)
                elif gene.attrs["regularization"] == "dropout":
                    x = Dropout(rate=gene.attrs["dropout_rate"])(x)

        # Add decoder layer for asymmetric autoencoder or add output layer for symmetric autoencoder
        if shape == "symmetric" or coding_sequence[0].attrs["output"] == "transpose":
            # Define output deconvolutional layer
            output = Conv2DTranspose(filters=out_channels,
                                     kernel_size=out_kernel_size,
                                     name="output",
                                     activation="sigmoid")(x)
        # Add decoder for flatten + dense asymmetric autoencoder
        elif coding_sequence[0].attrs["output"] == "dense" and shape == "asymmetric":
            x = GlobalAveragePooling2D()(x)
            x = Dense(out_dims ** 2 * out_channels)(x)
            x = Activation("sigmoid")(x)
            output = Reshape(input_shape)(x)

        model = Model(inputs=inputs, outputs=output)

        print(model.summary())

        return model
