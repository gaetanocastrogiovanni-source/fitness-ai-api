from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))

# Generator lazy loading
generator = None

def get_generator():
    global generator
    if generator is None:
        generator = pipeline(
            "text-generation", 
            model="distilgpt2",
            tokenizer="distilgpt2",
            device=-1  # usa CPU
        )
    return generator

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ready", "model": "distilgpt2"})

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', 'Hello')
        
        gen = get_generator()
        result = gen(
            prompt,
            max_length=data.get('max_length', 50),
            temperature=0.7,
            do_sample=True
        )
        
        return jsonify({
            "success": True,
            "generated_text": result[0]['generated_text']
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=False)
