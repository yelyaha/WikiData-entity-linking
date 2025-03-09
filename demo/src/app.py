import os
import logging

from time import time
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from sqlalchemy.orm import sessionmaker

from postgres_db.postgres_models import PostgresManager
from postgres_db.postgres_engine import create_engine_from_env

dpath = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path="/static", static_folder=os.path.join(dpath, "static"))
CORS(app)

model_name = os.environ.get('MODEL_NAME', None)

engine = create_engine_from_env()
Session = sessionmaker(bind=engine)
session = Session()
postgres_manager = PostgresManager(session, model_name, embed_column="embedding", embed_dim=768)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['GET'])
def search():
    try:
        query = request.args['query']
        top_k = int(request.args['topk'])
        threshold = float(request.args['threshold'])
    
        start_time = time()
        results = postgres_manager.search_embed(query, top_k, threshold)
        search_time = time() - start_time
        
        return jsonify({"results": results, "time": f"{search_time:.4f}"})
    
    except Exception as e:
        app.logger.error("Error in search endpoint: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
