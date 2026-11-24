"""Converts openWakeWord's exported ONNX detector (Flatten, then Linear,
LayerNorm, ReLU twice, then Linear and Sigmoid) to TFLite by rebuilding it in
Keras with the same weights; onnx_tf, openwakeword's own converter, does not
run on TensorFlow 2.19. Checks the TFLite model against ONNX Runtime."""
import sys

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
from onnx import numpy_helper

src, dst = sys.argv[1], sys.argv[2]
graph = onnx.load(src).graph
weights = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
shape = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim][1:]

# Initializers in graph order: (Linear W, b, LayerNorm g, b) twice, then Linear W, b.
linear = [w for n, w in weights.items() if w.ndim == 2]
vectors = [n for n, w in weights.items() if w.ndim == 1]
print("shape", shape, "linear", [w.shape for w in linear], "vectors", vectors)


def get(name_part):
    hits = [n for n in weights if name_part in n]
    assert len(hits) == 1, (name_part, hits)
    return weights[hits[0]]


inp = tf.keras.Input(shape=shape, batch_size=1)
x = tf.keras.layers.Flatten()(inp)
stages = [("layer1", "layernorm1"), ("blocks.0.fcn_layer", "blocks.0.layer_norm")]
out = x
for lin, norm in stages:
    W, b = get(f"{lin}.weight"), get(f"{lin}.bias")
    dense = tf.keras.layers.Dense(W.shape[0])
    out = dense(out)
    dense.set_weights([W.T, b])
    ln = tf.keras.layers.LayerNormalization(epsilon=1e-5)
    out = ln(out)
    ln.set_weights([get(f"{norm}.weight"), get(f"{norm}.bias")])
    out = tf.keras.layers.ReLU()(out)
W, b = get("last_layer.weight"), get("last_layer.bias")
last = tf.keras.layers.Dense(W.shape[0], activation="sigmoid")
out = last(out)
last.set_weights([W.T, b])
model = tf.keras.Model(inp, out)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite = converter.convert()
open(dst, "wb").write(tflite)

# Same answers as the ONNX model on random and real-looking inputs.
sess = ort.InferenceSession(src)
interp = tf.lite.Interpreter(model_content=tflite)
interp.allocate_tensors()
i_in, i_out = interp.get_input_details()[0]["index"], interp.get_output_details()[0]["index"]
worst = 0.0
for seed in range(200):
    x = np.random.default_rng(seed).normal(0, 3, size=[1] + shape).astype(np.float32)
    a = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    interp.set_tensor(i_in, x)
    interp.invoke()
    worst = max(worst, float(np.abs(a - interp.get_tensor(i_out)).max()))
print(f"wrote {dst}; largest difference from ONNX over 200 inputs: {worst:.2e}")
assert worst < 1e-4
