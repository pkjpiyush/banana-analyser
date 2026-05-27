from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json, io

app = Flask(__name__)
model = tf.keras.models.load_model("model/banana_model_finetuned.h5")

with open("model/class_indices.json") as f:
    class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}

# Shelf life as min/max days for interpolation
shelf_life_range = {
    "unripe":   {"min": 5, "max": 8, "color": "#4fc3f7", "emoji": "🟢"},
    "ripe":     {"min": 2, "max": 4,  "color": "#81c784", "emoji": "🟡"},
    "overripe": {"min": 1, "max": 2,  "color": "#ffb74d", "emoji": "🟠"},
    "rotten":   {"min": 0, "max": 0,  "color": "#e57373", "emoji": "🔴"}
}

suggestions = {
    "unripe": [
        "Store at room temperature to ripen naturally",
        "Keep away from direct sunlight",
        "Place near other fruits to speed up ripening",
        "Do not refrigerate — it stops the ripening process",
        "Check daily for yellow coloring to appear"
    ],
    "ripe": [
        "Best time to eat for maximum nutrition",
        "Refrigerate to slow down further ripening",
        "Peel and freeze if not eating within 2 days",
        "Great for smoothies, cereal, or eating fresh",
        "Keep away from other fruits to avoid over-ripening them"
    ],
    "overripe": [
        "Use immediately for best taste",
        "Perfect for banana bread or muffins",
        "Blend into smoothies or milkshakes",
        "Mash and freeze in portions for future baking",
        "Do not leave at room temperature any longer"
    ],
    "rotten": [
        "Do not consume — discard immediately",
        "Check surrounding fruits for contamination",
        "Clean the storage area to prevent mold spread",
        "Compost if possible instead of throwing away",
        "Inspect your storage conditions to prevent future waste"
    ]
}

storage_tips = {
    "unripe": [
        "🌡️ Ideal temperature: 25–30°C (room temperature)",
        "💨 Keep in a well ventilated area",
        "🚫 Never store below 12°C — causes chilling injury",
        "🍎 Place near apples or tomatoes to ripen faster",
        "☀️ Avoid direct sunlight — causes uneven ripening"
    ],
    "ripe": [
        "🌡️ Room temp: consume within 2–4 days",
        "❄️ Refrigerate at 8–12°C to extend by 2–3 more days",
        "⚠️ Skin will darken in fridge but fruit stays fresh inside",
        "🚫 Do not freeze unless peeled first",
        "📦 Store separately from other fruits"
    ],
    "overripe": [
        "❄️ Refrigerate immediately at 8–10°C",
        "🧊 Peel and freeze at -18°C for up to 3 months",
        "🚫 Do not store at room temperature any longer",
        "🍌 Best used within 24 hours if not refrigerated",
        "📦 Store in airtight container if refrigerating"
    ],
    "rotten": [
        "🗑️ Discard immediately — do not store",
        "🧹 Clean storage area with mild disinfectant",
        "🔍 Check all nearby fruits for contamination",
        "💨 Ventilate the storage area",
        "🚫 Do not compost near food storage areas"
    ]
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).resize((224, 224)).convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)[0]

    # Get all class scores
    class_scores = {idx_to_class[i]: float(predictions[i]) for i in range(4)}

    # Primary class with threshold adjustment
    class_idx = int(np.argmax(predictions))
    class_name = idx_to_class[class_idx]
    confidence = float(predictions[class_idx]) * 100

    # If rotten confidence is below 70% and overripe is close, call it overripe
    rotten_idx = [k for k, v in idx_to_class.items() if v == "rotten"][0]
    overripe_idx = [k for k, v in idx_to_class.items() if v == "overripe"][0]
    rotten_score = float(predictions[rotten_idx]) * 100
    overripe_score = float(predictions[overripe_idx]) * 100

    if class_name == "rotten" and rotten_score < 60 and overripe_score > 35:
        class_name = "overripe"
        confidence = overripe_score

    # Interpolated shelf life calculation
    total_days = 0
    total_weight = 0
    for cls, score in class_scores.items():
        mid_days = (shelf_life_range[cls]["min"] + shelf_life_range[cls]["max"]) / 2
        total_days += mid_days * score
        total_weight += score
    interpolated_days = total_days / total_weight

    # Format shelf life string
    days_rounded = round(interpolated_days)
    stage_labels = {
        "unripe": "Unripe — Not ready yet",
        "ripe": "Ripe — Best time to eat",
        "overripe": "Overripe — Use soon",
        "rotten": "Rotten — Discard immediately"
    }
    if days_rounded == 0:
        shelf_str = "0 days — Discard immediately"
    elif days_rounded == 1:
        shelf_str = f"1 day ({stage_labels[class_name]})"
    else:
        shelf_str = f"{days_rounded} days ({stage_labels[class_name]})"

    info = shelf_life_range[class_name]

    return jsonify({
        "class": class_name.upper(),
        "confidence": round(confidence, 1),
        "shelf_life": shelf_str,
        "interpolated_days": round(interpolated_days, 1),
        "color": info["color"],
        "emoji": info["emoji"],
        "suggestions": suggestions[class_name],
        "storage_tips": storage_tips[class_name],
        "all_scores": {k: round(v*100, 1) for k, v in class_scores.items()}
    })

if __name__ == '__main__':
    app.run(debug=True)