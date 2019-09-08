from app.api.handlers import handler, hello_handler, predict_handler


def setup_routes(app):
    app.router.add_get('/', handler)
    app.router.add_get('/hello', hello_handler)
    app.router.add_post('/predict', predict_handler)