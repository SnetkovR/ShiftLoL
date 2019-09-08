from aiohttp import web
from app.predictor import get_prediction
import logging
import json


async def handler(request: web.Request) -> web.Response:
    request.app['config']
    return web.Response(status=200, text='Some text')


async def hello_handler(request: web.Request) -> web.Response:
    return web.Response(text='Hello world')


async def predict_handler(request):

    # WARNING: don't do that if you plan to receive large files!
    data = await request.content.read()
    data = json.loads(data)
    print(data)
    print(type(data))
    logging.info('data:', data)
    team_1, team_2 = data['team_1'], data['team_2']

    return web.json_response(await get_prediction(team_1, team_2))
