"""AWS Lambda: GET / serves the web page, POST /predict classifies an image."""
import base64
import io
import json
import os
from pathlib import Path

import boto3
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image, ImageOps

SIZE = 224
PAGE = (Path(__file__).parent / "index.html").read_text()


def _load_model():
    """Runs once per cold start: pull the model from S3 into /tmp."""
    s3 = boto3.client("s3")
    bucket, prefix = os.environ["MODEL_BUCKET"], os.environ.get("MODEL_PREFIX", "")
    s3.download_file(bucket, f"{prefix}model.tflite", "/tmp/model.tflite")
    labels = json.loads(s3.get_object(Bucket=bucket, Key=f"{prefix}labels.json")["Body"].read())
    interp = Interpreter(model_path="/tmp/model.tflite")
    interp.allocate_tensors()
    return interp, labels


INTERP, LABELS = _load_model()
IN_IDX = INTERP.get_input_details()[0]["index"]
OUT_IDX = INTERP.get_output_details()[0]["index"]


def predict(image_bytes: bytes, k: int = 3):
    # Must match prepare.py's to_square() exactly.
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    img = ImageOps.fit(img, (SIZE, SIZE), Image.Resampling.BILINEAR)
    INTERP.set_tensor(IN_IDX, np.asarray(img, np.float32)[None])
    INTERP.invoke()
    probs = INTERP.get_tensor(OUT_IDX)[0]
    return [{"label": LABELS[i], "confidence": round(float(probs[i]), 4)} for i in np.argsort(-probs)[:k]]


def _resp(status, body, ctype="application/json"):
    return {
        "statusCode": status,
        "headers": {"content-type": ctype},
        "body": body if isinstance(body, str) else json.dumps(body),
    }


def lambda_handler(event, context):
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    path = event.get("rawPath", "/")

    if method == "GET" and path == "/":
        return _resp(200, PAGE, "text/html; charset=utf-8")
    if method == "GET" and path == "/labels":
        return _resp(200, {"labels": LABELS})
    if method == "POST" and path == "/predict":
        body = event.get("body") or ""
        data = base64.b64decode(body) if event.get("isBase64Encoded") else body.encode("latin-1")
        if not data:
            return _resp(400, {"error": "send an image as the request body"})
        try:
            return _resp(200, {"predictions": predict(data)})
        except Exception as e:
            return _resp(400, {"error": f"could not read image: {e}"})
    return _resp(404, {"error": "not found"})
