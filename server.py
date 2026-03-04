from flask import Flask, jsonify
import datetime

app = Flask(__name__)

# Endpoint di health check (funziona già)
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.datetime.now().isoformat()})

# Endpoint principale mancante - AGGIUNGI QUESTO
@app.route('/')
def home():
    return jsonify({
        "message": "Fitness AI API", 
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "generate": "/generate"
        }
    })

# Altri endpoint...
@app.route('/generate', methods=['POST'])
def generate_text():
    # La tua logica qui
    return jsonify({"result": "text generated"})
