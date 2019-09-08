import sys

from aiohttp import web
from app.routes import setup_routes
from app.utils.configurator import get_config
from app.utils.logs import init_logger

def init_app(argv):
    app = web.Application()
    app['config'] = get_config(argv)
    init_logger(app['config'])
    setup_routes(app)
    return app


def start(argv):
    app = init_app(argv)
    web.run_app(app, port=app['config']['service']['port'])


if __name__ == '__main__':
    #start(sys.argv[1:])
    start('config/default.yaml')
