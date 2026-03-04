from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Inizializza il modello e tokenizer
print("Caricamento del modello GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
print("Modello caricato con successo!")

# Configura il padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'online', 
        'model': 'GPT-2',
        'ready': True
    })

@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({
                'error': 'Il campo "prompt" è obbligatorio'
            }), 400
        
        prompt = data['prompt']
        max_length = data.get('max_length', 100)
        temperature = data.get('temperature', 0.7)
        
        # Tokenizza l'input
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        
        # Genera il testo
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decodifica l'output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({
            'prompt': prompt,
            'generated_text': generated_text,
            'tokens_generated': len(outputs[0]) - len(inputs[0])
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Errore durante la generazione: {str(e)}'
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint per interazioni conversazionali"""
    try:
        data = request.get_json()
        message = data.get('message', 'Ciao!')
        
        prompt = f"Utente: {message}\nAI:"
        
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=len(inputs[0]) + 50,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Estrae solo la parte dopo "AI:"
        if "AI:" in response:
            ai_response = response.split("AI:")[-1].strip()
        else:
            ai_response = response.replace(prompt, "").strip()
        
        return jsonify({
            'user_message': message,
            'ai_response': ai_response
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return """
    <html>
        <head>
            <title>GPT-2 API</title>
        </head>
        <body>
            <h1>GPT-2 API è attiva! 🚀</h1>
            <p>Endpoints disponibili:</p>
            <ul>
                <li>POST /generate - Genera testo</li>
                <li>POST /chat - Chat interattiva</li>
                <li>GET /health - Stato del servizio</li>
            </ul>
        </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
