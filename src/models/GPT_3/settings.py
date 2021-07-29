import os


PROCESS_FILE_PATH = "/code/script"
SERVICE_CONFIG = {
    "host": "0.0.0.0",
    "port": 19542,
    "debug": False
}


BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LOG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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





# DEBUG_DIR = os.path.join(BASE_DIR, 'debugFile')
# DEBUG_SH_PATH = os.path.join(DEBUG_DIR, 'generate_text.sh')
#
# print(BASE_DIR)
print(LOG_DIR)

GENERATE_DIR = os.path.join(BASE_DIR, 'generate')
# GENERATE_SH_PATH = os.path.join(BASE_DIR, 'GPT_3/apps/generate/generate_string.py')
print(LOG_DIR)

