from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Carica GPT-2 (piccolo, leggero)
generator = pipeline("text-generation", model="gpt2")

@app.route("/api", methods=["POST"])
def ai_advice():
    """
    Riceve JSON:
    {
        "prompt": "Dammi un consiglio fitness breve"
    }
    Restituisce:
    {
        "advice": "..."
    }
    """
    data = request.get_json()
    prompt = data.get("prompt", "Dammi un consiglio fitness breve")

    # Genera testo
    result = generator(prompt, max_length=50, num_return_sequences=1)

    advice = result[0]["generated_text"]

    return jsonify({"advice": advice})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
