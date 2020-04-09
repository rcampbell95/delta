
from delta.search.find_network import find_network
from delta.config import config

def setup_parser(subparsers):
    sub = subparsers.add_parser('search', help='Search for a feature representation.')

    #sub.add_argument('--img', dest='img', action='store_true', help='Save image of .')
    sub.add_argument('model', help='Path to save the network to.')

    sub.set_defaults(function=main)
    # TODO: move chunk_size into model somehow
    config.setup_arg_parser(sub, labels=True, train=False)

def main(options):
    try:
        model = find_network()

        print(options.model, model)
    except KeyboardInterrupt:
        print()
        print('Training cancelled.')
