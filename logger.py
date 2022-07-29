import logging
from datetime import datetime
import os
import pandas as pd



LOG_FILE_PATH='logs.log'



logging.basicConfig(filename=LOG_FILE_PATH,
filemode="w",
format='[%(asctime)s]^;%(levelname)s^;%(lineno)d^;%(filename)s^;%(funcName)s()^;%(message)s',level=logging.INFO
)