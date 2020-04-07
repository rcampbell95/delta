"""
Lists all avaiable commands.
"""

from . import classify, train, search, mlflow_ui

SETUP_COMMANDS = [train.setup_parser,
                  classify.setup_parser,
                  search.setup_parser,
                  mlflow_ui.setup_parser
                 ]
