# ================================================================
# 분리수거 AI - Flask API 서버
# ================================================================
# 실행 방법:
#   pip install flask flask-cors tf_keras transformers==4.40.0
#   python app.py
#
# ngrok으로 외부 공개 (선택):
#   ngrok http 5000
#   → 발급된 URL을 웹 UI의 API 엔드포인트 입력창에 붙여넣기
# ================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)  # 웹 UI에서 fetch 호출 허용 (CORS 설정 필수!)

# ── 모델 로드 ────────────────────────────────────────────────────
print("모델 로딩 중...")

import tf_keras as keras
from transformers import AutoTokenizer, TFRobertaModel
import zipfile, json

MODEL_DIR = "./model"

def load_model_from_weights(keras_path, weights_path, custom_objects=None):
    with zipfile.ZipFile(keras_path, 'r') as z:
        config = json.loads(z.read('config.json'))
    model = keras.models.model_from_json(json.dumps(config), custom_objects=custom_objects)
    weights = np.load(weights_path, allow_pickle=True)
    model.set_weights(weights)
    return model

tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "roberta_finetuned"))
model = load_model_from_weights(
    os.path.join(MODEL_DIR, "p_best.keras"),
    os.path.join(MODEL_DIR, "weights.npy"),
    custom_objects={"TFRobertaModel": TFRobertaModel}
)

print("모델 로드 완료!")

# 카테고리 라벨 (DISPOSAL_GUIDE 인덱스와 동일하게 맞춰주세요)
LABELS = {
    0:  "종이",
    1:  "종이팩",
    2:  "종이컵",
    3:  "캔",
    4:  "재사용유리 (공병)",
    5:  "유리",
    6:  "페트 (PET)",
    7:  "플라스틱",
    8:  "비닐",
    9:  "종이 (이물질 오염)",
    10: "종이컵 (이물질 오염)",
    11: "캔 (이물질 오염)",
    12: "유리 (이물질 오염)",
    13: "페트 (이물질 + 다중포장재)",
    14: "페트 (이물질 오염)",
    15: "플라스틱 (이물질 오염)",
    16: "비닐 (이물질 오염)",
    17: "재사용유리 (다중포장재)",
    18: "유리 (다중포장재)",
    19: "페트 (다중포장재)",
    20: "스티로폼",
    21: "스티로폼 (이물질 오염)",
    22: "건전지",
}

# ── /predict 엔드포인트 ──────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    요청 형식 (JSON):
        { "text": "쓰레기 상태 설명 텍스트" }

    응답 형식 (JSON):
        {
            "predicted_class": 6,         # 0~22 정수 인덱스
            "label": "페트 (PET)",         # 카테고리 이름
            "confidence": 0.94            # 0.0~1.0 신뢰도
        }
    """
    data = request.get_json(force=True)

    text = str(data.get("text", "")).strip() or "분리수거 항목"
    image_b64 = data.get("image", None)

    try:
        # 이미지 전처리
        if image_b64:
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]
            img_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img, dtype=np.float32)
        else:
            img_array = np.zeros((224, 224, 3), dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

        # 토크나이징
        inputs = tokenizer(
            text,
            return_tensors="tf",
            padding="max_length",
            truncation=True,
            max_length=64
        )

        # 모델 추론 (이미지 + 텍스트)
        # model.inputs 순서: image, input_ids, attention_mask
        outputs = model([img_batch, inputs["input_ids"], inputs["attention_mask"]])

        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs

        probs = np.array(logits)[0]  # (num_classes,)
        # softmax 수동 계산
        exp = np.exp(probs - np.max(probs))
        probs = exp / exp.sum()

        predicted_class = int(np.argmax(probs))
        confidence      = float(probs[predicted_class])
        label           = LABELS.get(predicted_class, f"클래스 {predicted_class}")

        return jsonify({
            "predicted_class": predicted_class,
            "label":           label,
            "confidence":      round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── /health 헬스체크 ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "BertEfficient_recycle"})


# ── 실행 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # debug=False 권장 (모델 중복 로드 방지)
    app.run(host="0.0.0.0", port=5000, debug=False)
