
from delta.search.find_network import find_network
from delta.config import config


def main(options):
    try:
        model = find_network()

        model.save(options.model)
    except KeyboardInterrupt:
        print()
        print('Training cancelled.')
