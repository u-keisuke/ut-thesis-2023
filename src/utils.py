import logging
import os


def get_logger(name, path):
    # make folder for path
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    format = "%(levelname)-9s  %(asctime)s [%(filename)s:%(lineno)d] %(message)s"

    # StreamHandlerによる出力フォーマット
    st_handler = logging.StreamHandler()
    st_handler.setLevel(logging.INFO)
    st_handler.setFormatter(logging.Formatter(format))

    # FileHandlerによる出力フォーマット
    fl_handler = logging.FileHandler(filename=path, encoding="utf-8")
    fl_handler.setLevel(logging.DEBUG)
    fl_handler.setFormatter(logging.Formatter(format))

    logger.addHandler(st_handler)
    logger.addHandler(fl_handler)

    return logger


def add_list_to_dict(target_dict, key, value):
    if key in target_dict.keys():
        target_dict[key].append(value)
    else:
        target_dict[key] = [value]
