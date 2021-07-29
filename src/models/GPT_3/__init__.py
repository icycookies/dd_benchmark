from flask import Flask
from flask_cors import *
import logging


import logging.config



app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config.from_object('GPT_3.settings')
app.config['JSON_AS_ASCII'] = False

# flask log
# BASE_DIR = app.config.get("ROOT_DIR")
LOG_DIR = app.config.get("LOG_DIR")
# handler = logging.FileHandler(LOG_DIR+'/log/flask.log')
# app.debug = True
# logging_format = logging.Formatter('%(asctime)s - %(levelname)s -%(pathname)s- %(filename)s - %(funcName)s - %(lineno)s - %(message)s')
# handler.setFormatter(logging_format)
# app.logger.addHandler(handler)


# traceback logging
# logging.config.dictConfig(app.config.get("log_config"))
log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "verbose": {
                "format": "%(levelname)s %(asctime)s [%(module)s:%(funcName)s] %(process)d %(thread)d %(filename)s:%(lineno)d %(message)s"
            },
            "simple": {
                "format": "%(levelname)s %(asctime)s - %(filename)s:%(lineno)d - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "simple"
            },
            "script_file": {
                "level": "DEBUG",
                "class": "logging.handlers.TimedRotatingFileHandler",
                "formatter": "verbose",
                "filename": LOG_DIR + "/log/scripts.log",
                "when": "D",
                "interval": 1,
                "backupCount": 30,
                "encoding": "utf-8"
            },
            "process_file": {
                "level": "DEBUG",
                "class": "logging.handlers.TimedRotatingFileHandler",
                "formatter": "verbose",
                "filename": LOG_DIR + "/log/process.log",
                "when": "D",
                "interval": 1,
                "backupCount": 30,
                "encoding": "utf-8"
            }
        },
        "loggers": {
            "script": {
                "handlers": ["script_file", "console"],
                "level": "DEBUG"
            },
            "process": {
                "handlers": ["process_file", "console"],
                "level": "DEBUG"
            }
        }
    }
logging.config.dictConfig(log_config)
