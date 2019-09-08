import pathlib

import trafaret as T

BASE_DIR = pathlib.Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / 'config' / 'default.yaml'

TRAFARET_FOR_CONFIG = T.Dict({
    'service': T.Dict({
        'name': T.String,
        'version': T.String,
        'port': T.Int(gte=1024)
    })
})