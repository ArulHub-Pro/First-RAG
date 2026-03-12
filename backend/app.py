from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_pipeline import get_answer

app = Flask(__name__)
CORS(app)

@app.route("/ask", methods=["POST"])
def ask():

    data = request.json
    question = data.get("question")

    answer = get_answer(question)

    return jsonify({
        "question": question,
        "answer": answer
    })

if __name__ == "__main__":
    app.run(debug=True)