from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return "<h1>Hello, Flask!</h1>"  # まずはHTML直接返す

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)