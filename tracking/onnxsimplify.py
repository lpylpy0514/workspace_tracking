import onnx
from onnxsim import simplify

model = onnx.load("vttrack.onnx")

model_simp, check = simplify(model)

onnx.save(model_simp, "simplified.onnx")
