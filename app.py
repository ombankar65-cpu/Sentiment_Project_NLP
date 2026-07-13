import os
import pickle
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load both the vectorizer and the model files
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

model = None
vectorizer = None

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

# HTML Template with modern layout, tailored colors, and shadow effects
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis Dashboard</title>
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            --card-bg: #ffffff;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --shadow-sm: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            --positive: #10b981;
            --negative: #ef4444;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0;
            padding: 20px;
        }

        .container {
            background-color: var(--card-bg);
            padding: 40px;
            border-radius: 16px;
            box-shadow: var(--shadow-lg);
            width: 100%;
            max-width: 500px;
            transition: transform 0.2s ease;
        }

        h1 {
            font-size: 1.8rem;
            margin-top: 0;
            margin-bottom: 8px;
            text-align: center;
            color: var(--text-main);
        }

        p.subtitle {
            text-align: center;
            color: var(--text-muted);
            margin-bottom: 30px;
            font-size: 0.95rem;
        }

        textarea {
            width: 100%;
            height: 120px;
            padding: 14px;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            font-size: 1rem;
            resize: none;
            box-sizing: border-box;
            outline: none;
            box-shadow: var(--shadow-sm);
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        textarea:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }

        button {
            width: 100%;
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 14px;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 8px;
            margin-top: 20px;
            cursor: pointer;
            box-shadow: var(--shadow-sm);
            transition: background-color 0.2s, transform 0.1s;
        }

        button:hover {
            background-color: var(--primary-hover);
        }

        button:active {
            transform: scale(0.98);
        }

        .result-box {
            margin-top: 25px;
            padding: 16px;
            border-radius: 8px;
            text-align: center;
            font-weight: 600;
            font-size: 1.1rem;
            box-shadow: var(--shadow-sm);
            animation: fadeIn 0.4s ease forwards;
        }

        .positive {
            background-color: rgba(16, 185, 129, 0.1);
            color: var(--positive);
            border: 1px solid rgba(16, 185, 129, 0.2);
        }

        .negative {
            background-color: rgba(239, 68, 68, 0.1);
            color: var(--negative);
            border: 1px solid rgba(239, 68, 68, 0.2);
        }

        .error {
            background-color: #fef2f2;
            color: var(--negative);
            border: 1px solid #fee2e2;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="container">
    <h1>Sentiment Analysis</h1>
    <p class="subtitle">Enter your text below to analyze its emotional tone.</p>
    
    <form method="POST" action="/">
        <textarea name="text" placeholder="Type or paste your text here..." required>{{ text }}</textarea>
        <button type="submit">Analyze Sentiment</button>
    </form>

    {% if error %}
    <div class="result-box error">
        {{ error }}
    </div>
    {% elif prediction %}
    <div class="result-box {{ prediction | lower }}">
        Sentiment: {{ prediction }}
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    text = ""
    error = None

    if request.method == "POST":
        text = request.form.get("text", "")
        if not model or not vectorizer:
            error = "Model or vectorizer files could not be loaded on the server."
        else:
            try:
                # 1. Convert raw text into numerical features using the vectorizer
                vectorized_text = vectorizer.transform([text])
                
                # 2. Feed the 2D sparse matrix to the Logistic Regression model
                pred = model.predict(vectorized_text)[0]
                prediction = str(pred).capitalize()
            except Exception as e:
                error = f"Prediction failed: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction, text=text, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
