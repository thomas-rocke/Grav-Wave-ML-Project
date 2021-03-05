import argparse
import multiprocessing
import Logger
from ML_Identification import *

LOG = Logger.get_logger(__name__)

if __name__ == '__main__':

    LOG.info("End of program.\n\n\n")
    LOG.info("Start of program.")

    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Handles the creation, training and evaluation of machine learning models for the identification of optical modes from superpositions")
    parser.add_argument("-i", "--initialise", action="store", dest="ML", help="initialise a model")
    parser.add_argument("-t", "--train", action="store_true", help="train a model")
    parser.add_argument("-s", "--save", action="store_true", help="save the model")
    parser.add_argument("-l", "--load", action="store_true", help="load the model")
    parser.add_argument("-e", "--evaluate", action="store", metavar="N", help="load the model")
    parser.add_argument("-o", "--optimise", action="store", dest="parameter", metavar=("param_name", "param_value"), nargs=2, help="optimise model by varying a parameter")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbosity")

    args = parser.parse_args()

    LOG.debug(f"Args: {args}")

    if args.ML != None:
        model = eval(args.ML)

        if args.train: model.train(info=args.verbose)
        if args.load: model.load(info=args.verbose)
        if args.save: model.save()
        if args.evaluate != None: model.evaluate(int(args.evaluate))
        if args.parameter != None: model.optimise(args.parameter[0], eval(args.parameter[1]), plot=True, save=not args.verbose)