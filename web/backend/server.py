import sys
import os
import base64
import io

from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import torch
import torchvision.transforms as T
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # web/backend
PROJECT_DIR = os.path.join(BASE_DIR, "..", "..")           # Project
sys.path.append(PROJECT_DIR)

from AI.train import CNN    # cnn 클래스 이름

app = Flask(__name__)
CORS(app)

DATA_DIR = os.path.join(PROJECT_DIR, "AI", "data", "train")
labels = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])
num_classes = 4037
print(f"[INFO] Loaded {num_classes} classes")

MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")

model = CNN(num_classes=num_classes)

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = T.Compose([
    T.Grayscale(),
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(mean=(0.5,), std=(0.5,))
])

def decoding(label):
    return bytes.fromhex(label[2:]).decode("gb2312")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "image" not in data:
        return jsonify({"error": "image field missing"}), 400

    img_base64 = data["image"]

    # DataURL 형식 → "data:image/png;base64,XXXX" → 실제 base64만 추출
    if "," in img_base64:
        img_base64 = img_base64.split(",")[1]

    # base64 → PIL Image
    try:
        img_bytes = base64.b64decode(img_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        img = ImageOps.invert(img)
        img.save("debug_input.png")
    except Exception as e:
        return jsonify({"error": "Invalid image"}), 400

    # 전처리
    x = transform(img).unsqueeze(0)

    # 추론
    with torch.no_grad():
        '''
        pred = model(x)
        cls = pred.argmax(dim=1).item()

    label = decoding(labels[cls])

    return jsonify({
        "class": f'{label}({labels[cls]})',
        "index": cls
    })
    '''
        output = model(x)
        probs = torch.softmax(output, dim = 1)

        _topk = 5
        values, indices = torch.topk(probs, _topk)

        values = values.squeeze().tolist()
        indices = indices.squeeze().tolist()

    return jsonify({
        "topk": [
            {"hanja": decoding(labels[cls]),
            "class": labels[cls],
            "probability": prob}
            for prob, cls in zip(values, indices)
        ]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
