import os
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# 1. Model Path Handling (File dhone ke liye best tarika)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'best.tflite')

# 2. Model Load (Global load karein taaki speed fast rahe)
if os.path.exists(model_path):
    model = YOLO(model_path, task='detect')
    print("✅ Model loaded successfully!")
else:
    model = None
    print(f"❌ Error: {model_path} nahi mili. GitHub check karein!")

@app.route('/')
def home():
    return "<h1>DigitalMadhu Police AI Server is Live!</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model file missing on server"}), 500
    
    # Yahan baad mein hum FlutterFlow se image lene ka logic add karenge
    return jsonify({"status": "Model ready for detection", "model": "YOLOv8-TFLite"})

# --- YE SABSE ZAROORI PART HAI RENDER KE LIYE ---
if __name__ == "__main__":
    # Render se port uthayega, agar nahi mila toh default 10000 use karega
    port = int(os.environ.get("PORT", 10000))
    # host='0.0.0.0' hona zaroori hai taaki bahar se access ho sake
    app.run(host='0.0.0.0', port=port)