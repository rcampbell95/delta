import numpy as np
import conv_autoencoder
import tensorflow as tf
from tensorflow.keras.utils import plot_model

class Gene:
    def __init__(self, node_id, config):
        self.define_gene()
        self.node_id = node_id
        self.rows = int(config["evolutionary_search"]["grid_height"])
        self.level_back = int(config["evolutionary_search"]["level_back"])

        self.random_init()
    
    def define_gene(self):
        self.params = {}
        self.attrs = {}

    def random_init(self):
        for param, value in self.attrs.items():
            # Select a random value in the set of parameters
            r_idx = int(np.random.random() * 10)
            r_idx = r_idx % len(self.params[param])
            self.attrs[param] = self.params[param][r_idx]

        self.mutate_node_id()   
        
    def mutate_node_id(self):
        node_id_lower = self.node_id - self.node_id % self.rows - self.level_back * self.rows
        node_id_upper = self.node_id - self.node_id % self.rows - 1
        
        self.conn = np.random.randint(node_id_lower, node_id_upper + 1)
        
    def mutate(self, r):
        """
        Mutate the gene with some probability r
        """
        prb_mutate = np.random.random()
        
        if prb_mutate <= r:
            for param, value in self.attrs.items():
                # Select a random value in the set of parameters
                r_idx = int(np.random.random() * 10)
                r_idx = r_idx % len(self.params[param])
                self.attrs[param] = self.params[param][r_idx]
                
            self.mutate_node_id()

            return True
        return False

class Genotype:
    def __init__(self, config, genes=False):
        self.height = int(config["evolutionary_search"]["grid_height"])
        self.width = int(config["evolutionary_search"]["grid_width"])
        self.n_genes = self.height * self.width + 1
        self.mutation_rate = float(config["evolutionary_search"]["r"])
        
        if not genes:
            self.genes = [Gene(i, config) for i in range(self.n_genes)]
        else:
            self.genes = genes
                
    def replicate(self, config=None):
        child_genes = self.genes.copy()
        encoder = self.trace_encoder()
        
        phenotype_mutated = False
        while not phenotype_mutated:
            for idx, gene in enumerate(child_genes):
                if gene in encoder and gene.mutate(self.mutation_rate):
                    phenotype_mutated = True
        child = Genotype(config, child_genes)
        return child

    def mutate_hidden_genes(self):
        """Mutate hidden genes of parent"""
        encoder = self.trace_encoder()

        hidden_mutated = False

        while not hidden_mutated:
            for idx, gene in enumerate(self.genes):
                if gene not in encoder:
                    hidden_mutated = gene.mutate(self.mutation_rate)
    
    def trace_encoder(self):
        curr_id = self.genes[-1].node_id
        encoder_nodes = []
        #encoder_nodes.append(self.genes[-1])
        
        while curr_id > -1:
            encoder_nodes.append(self.genes[curr_id])
            curr_id = self.genes[curr_id].conn
            
        return encoder_nodes
    
    def build_model(self, input_shape, shape):
        pass

Gene.define_gene = conv_autoencoder.define_gene
Genotype.build_model = conv_autoencoder.build_model


if __name__ == "__main__":
    parent = Genotype()

    print("{:^10s} {:^15s} {:^10s} {:^15s} {:^15s} {:^10s}".format("Node id", "Connection id", "Mutated", "Filter Size",
                                                      "Kernel Size", "Skip"))

    child = parent.replicate()
    coding_sequence = child.trace_encoder()

    ids = [gene.node_id for gene in coding_sequence]
    ids = ids[1:]

    print("input", *reversed(ids), "output", sep=" -> ")
    child.genotype.build_cae()

    output = model(tf.cast(np.random.rand(32, 32, 32, 3), tf.float32))
    print(output.shape)

    plot_model(model, to_file='parent_model.png', show_shapes=True)

    for i in range(5):
        child = child.replicate()

    model = child.genotype.build_cae()

    plot_model(model, to_file="child_model.png", show_shapes=True)



