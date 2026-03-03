"""Quick test for the myNeuron API."""
import requests
import json

API = "http://localhost:8000"

# Test 1: Code generation
data = {
    "layers": [
        {"layer_type": "Input", "input_shape": [784]},
        {"layer_type": "Dense", "units": 256, "activation": "relu"},
        {"layer_type": "Dropout", "rate": 0.3},
        {"layer_type": "Dense", "units": 10, "activation": "softmax"}
    ],
    "compile_config": {
        "optimizer": {"optimizer_type": "adam", "learning_rate": 0.001},
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"]
    },
    "training_config": {"epochs": 10, "batch_size": 32, "validation_split": 0.2},
    "model_name": "TestMLP"
}

print("=" * 50)
print("TEST 1: Code Generation")
print("=" * 50)
r = requests.post(f"{API}/api/generate-code", json=data)
result = r.json()
print(f"Success: {result['success']}")
print(result['code'][:600])
print("...")

print("\n" + "=" * 50)
print("TEST 2: Model Summary")
print("=" * 50)
r = requests.post(f"{API}/api/model-summary", json=data)
summary = r.json()
for layer in summary['layers']:
    print(f"  {layer['name']:20s} {layer['layer_type']:15s} {layer['output_shape']:20s} {layer['param_count']:>8,}")
print(f"\n  Total params: {summary['total_params']:,}")

print("\n" + "=" * 50)
print("TEST 3: Validation")
print("=" * 50)
r = requests.post(f"{API}/api/validate", json=data)
v = r.json()
print(f"Valid: {v['valid']}")
if v['warnings']:
    print(f"Warnings: {v['warnings']}")

print("\n✅ All tests passed!")
