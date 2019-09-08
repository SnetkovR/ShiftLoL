import asyncio
import concurrent.futures
from lol_pro_games_predictions import ProGamesPredictions


async def get_prediction(team_1, team_2):
    loop = asyncio.get_event_loop()
    model = ProGamesPredictions()
    
    with concurrent.futures.ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool, model.predict, team_1, team_2)
    return result
