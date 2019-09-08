import logging


def init_logger(config):
    logging.basicConfig(level=logging.DEBUG,
                        format=f'{config["service"]["name"]} - ' \
                               f'{config["service"]["version"]} - ' \
                               f'%(asctime)s - ' \
                               f'%(name)s - ' \
                               f'%(levelname)s - ' \
                               f'%(message)s',
                        datefmt='%m-%d %H:%M'
                        )