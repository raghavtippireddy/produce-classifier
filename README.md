# Produce Classifier

Upload a photo of a fruit or vegetable and get back what it is. A MobileNetV2 model is
fine-tuned in TensorFlow, shrunk to a 4.5 MB TensorFlow Lite file, and served from AWS Lambda.

**Live demo:** https://uk7hatuj6h45wpq7kaqtj7t3xm0bhfxz.lambda-url.us-east-2.on.aws/

## Results (558 held-out test photos, 33 classes)

| | |
|---|---|
| Top-1 accuracy | **86.9%** |
| Correct answer in top 3 | **95.3%** |
| Model size | 4.5 MB (float16) |

Weakest classes are look-alike root vegetables: radish, turnip and sweet potato are right
59–64% of the time (only 11–17 test photos each, so those numbers are noisy). Biggest win
would be more photos of those classes.

```
browser ──HTTPS──> Lambda Function URL ──> Lambda (Python 3.12, arm64)
                                             ├─ GET  /         → web page
                                             ├─ GET  /labels   → class list
                                             └─ POST /predict  → top-3 guesses
                                                   │ cold start
                                                   ▼
                                             S3 (private): model.tflite, labels.json
```

## Data

[Fruits and Vegetables Image Recognition](https://huggingface.co/datasets/Nattakarn/fruit-and-vegetable-image-recognition)
(mirror of Kritik Seth's Kaggle set): about 3,100 real-world web photos. `prepare.py` merges
labels that name the same thing (capsicum / paprika / bell pepper, sweetcorn / corn) and
removes test images that also appear in the training set, so the test score isn't inflated.

## Run it

```bash
conda activate produce          # Python 3.11, TensorFlow 2.18
python prepare.py               # parquet → data/*.npz (224×224)
python train.py                 # trains, exports model/model.tflite, prints test accuracy
export PATH="$HOME/aws-cli:$PATH" AWS_REGION=us-east-2   # AWS CLI v2
aws login --region us-east-2    # browser sign-in, lasts 12 hours
./deploy.sh                     # prints the live URL; re-run to push a new model
./teardown.sh                   # deletes everything deploy.sh made
```

`BUDGET_EMAIL=you@example.com ./deploy.sh` also creates a $5/month AWS budget alert.

## Design choices

- **Two-stage transfer learning.** First only the new top layer trains (fast, stable), then the
  last ~50 layers of MobileNetV2 are unfrozen at a 100× lower learning rate. On this run stage 2
  added nothing (validation stayed at 83–84%); almost all the learning is stage 1.
- **float16, not int8.** int8 made a 2.5 MB file but lost 1.4 points of accuracy; float16 lost
  0.2 points for 4.5 MB.
- **TFLite instead of full TensorFlow on Lambda.** Full TensorFlow is ~500 MB and would need a
  container image; the TFLite runtime fits in a normal zip, starts faster, and costs less.
  `train.py` scores the exported `.tflite` file itself, not just the Keras model.
- **Function URL instead of API Gateway.** One fewer service for a single-function app.
- **Preprocessing lives in one place.** The model takes raw 0–255 pixels; the resize/crop in
  `lambda/handler.py` matches `prepare.py` exactly.
- **Browser shrinks photos before upload.** Keeps requests well under Lambda's 6 MB limit.

## Cost

Within the AWS free tier for demo traffic (Lambda: 1M requests and 400k GB-seconds a month;
S3: a few MB). The URL is public with no login, so anyone with it can call it. Run
`./teardown.sh` when you're done showing it off.
