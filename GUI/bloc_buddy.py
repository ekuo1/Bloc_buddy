from app import app, db
from app.models import Climb, Screenshot

@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'Climb': Climb, 'Screenshot': Screenshot}