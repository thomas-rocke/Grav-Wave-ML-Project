import argparse
import multiprocessing
import Logger
from ML_Identification import *

LOG = Logger.get_logger(__name__)

if __name__ == '__main__':

    LOG.info("End of program.\n\n\n")
    LOG.info("Start of program.")

    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Handles the creation, training and evaluating of machine learning models for the identification of optical modes from superpositions")
    parser.add_argument("-i", "--initialise", action="store", dest="ML", help="Initialise a model")
    parser.add_argument("-t", "--train", action="store_true", help="Train a model")
    parser.add_argument("-s", "--save", action="store_true", help="Save the model")
    parser.add_argument("-l", "--load", action="store_true", help="Load the model")
    parser.add_argument("-o", "--optimise", action="store", dest="parameter", nargs=2, help="Optimise model by varying a parameter")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbosity")

    args = parser.parse_args()

    LOG.debug(f"Args: {args}")

    if args.ML != None:
        model = eval(args.ML)

        if args.train: model.train(info=args.verbose)
        if args.save: model.save()
        if args.load: model.load(info=args.verbose)
        if args.parameter != None: model.optimise(args.parameter[0], eval(args.parameter[1]))