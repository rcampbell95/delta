import copy

import numpy as np

from delta.config import config


class Gene:
    def __init__(self, node_id):
        """
        Parameters
        -----------
            node_id: number
                id used for identifying node in cartesian grid

            config: dict
                search configuration

        Returns
        -----------
        """
        self.params = self.define_gene()

        self.node_id = node_id
        self.rows = int(config.model_grid_height())
        self.level_back = int(config.model_level_back())
        self.conn = -1

        self.random_init()


    def define_gene(self):
        """
        Defines the gene for building corresponding models
        using a dictionary of parameters with corresponding search space of values.
        Parameters
        -----------
        Returns
        -----------
            gene : dict
        """
        raise NotImplementedError("Implement 'define_gene'")

    def random_init(self):
        """
        Initializes the gene to random values in search space.

        Parameters
        -----------
        Returns
        -----------
        """
        self.attrs = {}

        for param in self.params:
            # Select a random value in the set of parameters
            r_idx = np.random.randint(0, len(self.params[param]))

            self.attrs[param] = self.params[param][r_idx]

        self.mutate_node_id()

    def mutate_node_id(self):
        node_id_lower = self.node_id - self.node_id % self.rows - self.level_back * self.rows
        node_id_upper = self.node_id - self.node_id % self.rows - 1

        self.conn = np.random.randint(node_id_lower, node_id_upper + 1)

    def mutate(self, r):
        """
        Mutate the gene with some probability r

        Parameters
        -----------
        r : number
            Mutation rate

        Returns
        -----------
            None
        """
        to_mutate = np.random.binomial(1, r, 1)

        if to_mutate.item() == 1:
            for param, _ in self.attrs.items():
                # Select a random value in the set of parameters
                r_idx = np.random.randint(0, len(self.params[param]))

                self.attrs[param] = self.params[param][r_idx]

            self.mutate_node_id()

            return True
        return False

class Genotype:
    def __init__(self, genes=False):
        self.height = int(config.model_grid_height())
        self.width = int(config.model_grid_width())
        self.n_genes = self.height * self.width + 1
        self.mutation_rate = float(config.r())

        if not genes:
            self.genes = [Gene(i) for i in range(self.n_genes)]
        else:
            self.genes = genes

    def replicate(self):
        """
        Generates a child by mutating parent

        Parameters:
        -----------
            config: Search configuration

        Returns:
        -----------
            child: Child genotype created by mutating parent
        """
        child_genes = [copy.deepcopy(gene) for gene in self.genes]
        child = self.__class__(child_genes)
        encoder = child.trace_encoder()

        phenotype_mutated = False
        while not phenotype_mutated:
            for gene in child_genes:
                if gene in encoder and gene.mutate(self.mutation_rate):
                    phenotype_mutated = True

        return child

    def mutate_hidden_genes(self):
        """Mutate hidden (non-functioning) genes of parent"""
        encoder = self.trace_encoder()

        hidden_mutated = False

        while not hidden_mutated:
            for gene in self.genes:
                if gene not in encoder:
                    hidden_mutated = gene.mutate(self.mutation_rate)

    def trace_encoder(self):
        """
        Trace encoder back from output node
        Parameters:
        -----------
            None

        Returns:
        -----------
            encoder_nodes: Reversed list of nodes(genes) in encoder
        """
        curr_id = self.genes[-1].node_id
        encoder_nodes = []
        #encoder_nodes.append(self.genes[-1])

        while curr_id > -1:
            encoder_nodes.append(self.genes[curr_id])
            curr_id = self.genes[curr_id].conn

        return encoder_nodes

    def build_model(self, input_shape):
        """
        Build a model according to the individual's genotype
        Parameters:
        -----------
            input_shape: Input shape of the model to build

        Returns:
        -----------
            model: Keras model for training, uncompiled
        """
