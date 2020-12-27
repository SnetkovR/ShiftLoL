import asyncio
import concurrent.futures
from .lol_pro_games_predictions import ProGamesPredictions


def get_prediction(team_1, team_2):
    loop = asyncio.get_event_loop()
    model = ProGamesPredictions()
    
    return model.predict(team_1, team_2)
