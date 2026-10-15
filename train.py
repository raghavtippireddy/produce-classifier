"""Fine-tune MobileNetV2 on the produce photos and export a TensorFlow Lite model."""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "model"
OUT.mkdir(exist_ok=True)
SEED = 42
BATCH = 32

tf.keras.utils.set_random_seed(SEED)
labels = json.loads((DATA / "labels.json").read_text())
train, test = np.load(DATA / "train.npz"), np.load(DATA / "test.npz")

# Hold out 15% of train (per class) for validation; the test set stays untouched until the end.
rng = np.random.default_rng(SEED)
val_idx = np.concatenate([
    rng.permutation(np.where(train["y"] == c)[0])[: max(1, int(0.15 * (train["y"] == c).sum()))]
    for c in range(len(labels))
])
is_val = np.zeros(len(train["y"]), bool)
is_val[val_idx] = True
x_tr, y_tr = train["x"][~is_val], train["y"][~is_val]
x_va, y_va = train["x"][is_val], train["y"][is_val]
x_te, y_te = test["x"], test["y"]
print(f"train {len(y_tr)}  val {len(y_va)}  test {len(y_te)}  classes {len(labels)}")

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2, value_range=(0, 255)),
])


def dataset(x, y, training):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(len(y), seed=SEED)
    ds = ds.batch(BATCH).map(lambda a, b: (tf.cast(a, tf.float32), b))
    if training:
        ds = ds.map(lambda a, b: (augment(a, training=True), b))
    return ds.prefetch(tf.data.AUTOTUNE)


ds_tr, ds_va, ds_te = dataset(x_tr, y_tr, True), dataset(x_va, y_va, False), dataset(x_te, y_te, False)

# The model takes raw 0-255 pixels so the Lambda doesn't need to know MobileNet's scaling.
base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base.trainable = False
inputs = tf.keras.Input((224, 224, 3))
x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(inputs)
x = base(x, training=False)  # keep batch-norm stats frozen, even when fine-tuning
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(labels), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)


def fit(lr, epochs):
    model.compile(tf.keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
    stop = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy")
    return model.fit(ds_tr, validation_data=ds_va, epochs=epochs, callbacks=[stop], verbose=2)


print("\n== stage 1: train the new top layer only")
fit(1e-3, 15)
print("\n== stage 2: fine-tune the last ~50 layers of MobileNetV2")
base.trainable = True
for layer in base.layers[:-50]:
    layer.trainable = False
fit(1e-5, 10)

_, keras_acc = model.evaluate(ds_te, verbose=0)
print(f"\nKeras test accuracy: {keras_acc:.3f}")

# Export to TFLite with float16 weights: half the size, no measurable accuracy loss.
# (int8 dynamic-range was 2.5 MB but cost ~1.4 points of test accuracy.)
model.export(str(OUT / "saved_model"))
conv = tf.lite.TFLiteConverter.from_saved_model(str(OUT / "saved_model"))
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.target_spec.supported_types = [tf.float16]
(OUT / "model.tflite").write_bytes(conv.convert())
(OUT / "labels.json").write_text(json.dumps(labels))
print(f"model.tflite: {(OUT / 'model.tflite').stat().st_size / 1e6:.1f} MB")

# Score the exact file that will ship, with the same runtime the Lambda uses.
interp = Interpreter(model_path=str(OUT / "model.tflite"))
interp.allocate_tensors()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
probs = []
for img in x_te:
    interp.set_tensor(inp["index"], img[None].astype(np.float32))
    interp.invoke()
    probs.append(interp.get_tensor(out["index"])[0])
probs = np.array(probs)
pred = probs.argmax(1)
top3 = (np.argsort(-probs, 1)[:, :3] == y_te[:, None]).any(1)
print(f"TFLite test accuracy: top-1 {np.mean(pred == y_te):.3f}  top-3 {top3.mean():.3f}  (n={len(y_te)})")

per_class = sorted(
    ((np.mean(pred[y_te == c] == c), labels[c], int((y_te == c).sum())) for c in range(len(labels)) if (y_te == c).any())
)
print("weakest classes (accuracy, class, n):")
for acc, name, n in per_class[:6]:
    wrong = [labels[p] for p in pred[(y_te == labels.index(name)) & (pred != labels.index(name))]]
    print(f"  {acc:.2f}  {name:<14} n={n}  confused with: {', '.join(sorted(set(wrong)))}")
