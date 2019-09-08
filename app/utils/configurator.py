import argparse

from trafaret_config import commandline

from app.settings import TRAFARET_FOR_CONFIG, DEFAULT_CONFIG_PATH


def get_config(argv):
    ap = argparse.ArgumentParser()
    commandline.standard_argparse_options(
        ap,
        default_config=DEFAULT_CONFIG_PATH
    )
    options, unknown = ap.parse_known_args(argv)

    config = commandline.config_from_options(options, TRAFARET_FOR_CONFIG)

    return config
