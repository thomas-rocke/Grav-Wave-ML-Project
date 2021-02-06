import logging
import os
from datetime import datetime

def get_logger(logger_name):
    '''
    Create the logger.
    '''
    # logging.basicConfig(filename="Logs" + os.sep + "{:%d-%m-%Y}.log".format(datetime.now()), level="DEBUG", format=log_format)

    logger = logging.getLogger(logger_name) # Create logger instance

    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("Logs" + os.sep + "{:%d-%m-%Y}.log".format(datetime.now()))

    console_handler.setLevel(logging.ERROR)
    file_handler.setLevel(logging.DEBUG)

    console_format = logging.Formatter("%(module)s - %(levelname)s - %(message)s")
    file_format = logging.Formatter("%(asctime)s - %(module)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s")

    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propogate = False

    return logger