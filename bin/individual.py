import multiprocessing

import tensorflow as tf

from delta.config import config
from delta.imagery import imagery_dataset
from delta.ml.train import train

from conv_autoencoder import ConvAutoencoderGenotype
from gpu_manager import GPU_Manager

def assemble_dataset():

    # Use wrapper class to create a Tensorflow Dataset object.
    # - The dataset will provide image chunks and corresponding labels.
    tc = config.training()

    ids = imagery_dataset.AutoencoderDataset(config.images(), config.labels(), config.chunk_size(), tc.chunk_stride)

    return ids

class Individual(multiprocessing.Process):
    fitness_queue = multiprocessing.Queue(0)

    def __init__(self, config_values, new_genotype=False, child_index=0):
        multiprocessing.Process.__init__(self)

        self.history = None
        self.child_index = child_index
        self.config_values = config_values

        if new_genotype is False:
            self.genotype = ConvAutoencoderGenotype(config_values)
        else:
            self.genotype = new_genotype

    def self_mutate(self):
        self.genotype.mutate_hidden_genes()

    def generate_child(self, child_index):
        child_genotype = self.genotype.replicate(self.config_values)
        child = Individual(self.config_values, child_genotype, child_index)
        return child

    @classmethod
    def histories(cls):
        histories = []
        while cls.fitness_queue.qsize() > 0:
            msg = cls.fitness_queue.get(block=False)
            histories.append(msg[1])
        return histories


    def build_model(self):
        chunk_size = config.chunk_size()
        channels = int(self.config_values["ml"]["channels"])

        input_shape = (chunk_size, chunk_size, channels)

        model = self.genotype.build_model(self.config_values, input_shape)

        return model

        # Save model summary to text file
        # TODO -- make child directories if not available
        # TODO -- best option would be to make temp directory and delete after training

        # Save genetics to csv
        #gene_attrs = {}
        #for gene in self.genotype.genes:
        #    for attr, val in gene.attrs.items():
        #        if attr in gene_attrs:
        #            gene_attrs[attr].append(val)
        #        else:
        #            gene_attrs[attr] = [val]
        #    if "Connection id" in gene_attrs:
        #        gene_attrs["Connection id"].append(gene.conn)
        #    else:
        #        gene_attrs["Connection id"] = [gene.conn]

        #pd.DataFrame(gene_attrs).to_csv(os.path.join(self.config["ml"]["output_folder"], \
        # str(self.child_index), "genotype.csv"))

    def run(self):
        # TODO Set up context with GPU
        device_manager = GPU_Manager()
        device = device_manager.request_device()

        with tf.Graph().as_default():
            with tf.device(device):
                train_spec = config.training()
                ids = assemble_dataset()

                _, history = train(self.build_model, ids, train_spec)

                msg = (self.child_index, history.history)

                self.fitness_queue.put(msg)
