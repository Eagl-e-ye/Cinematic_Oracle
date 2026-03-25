from flask import Flask, request, jsonify, render_template
from chatbot import MovieChatbot
import pandas as pd

app = Flask(__name__)
app.secret_key = "*************"

df = pd.read_csv("complete_dataset_merged.csv")
print(f"✓ Dataset loaded: {len(df)} rows, columns: {list(df.columns)}")

bots = {}

@app.route("/")
def index():
    return render_template("cinematic_oracle.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data    = request.json
        user_id = data.get("user_id", "default")
        message = data.get("message", "")

        if user_id not in bots:
            bots[user_id] = MovieChatbot(df, max_optional_questions=2)

        bot = bots[user_id]
        reply, movies = bot.chat(message)

        if not isinstance(movies, list):
            movies = []

        return jsonify({
            "reply":  reply,
            "movies": movies,
            "ready":  len(movies) > 0
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "reply":  "The Oracle is momentarily silent. Please try again.",
            "movies": [],
            "ready":  False
        }), 500


@app.route("/reset", methods=["POST"])
def reset():
    try:
        data    = request.get_json(silent=True) or {}
        user_id = data.get("user_id", "default")
    except Exception:
        user_id = "default"

    if user_id in bots:
        bots[user_id].reset()

    return jsonify({"status": "reset"})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)