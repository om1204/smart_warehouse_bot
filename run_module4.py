import numpy as np
from part4_networks.hopfield_network import train, recall, corrupt, generate_pattern_1, generate_pattern_2, generate_pattern_3, generate_pattern_4
from part4_networks.rnn_model import train_rnn, rnn_predict_anomaly

print("========================================")
print("MODULE D: CONNECTIONIST MODELS")
print("========================================\n")

print("--- 1. Hopfield Network (Associative Recall) ---")
patterns = [
    generate_pattern_1(),
    generate_pattern_2(),
    generate_pattern_3(),
    generate_pattern_4()
]
print("Training Hopfield network on 4 patterns...")
W = train(patterns)

corrupted = corrupt(patterns[0], 0.15)
print(f"Original pattern 1 energy: {-(0.5 * np.dot(patterns[0].T, np.dot(W, patterns[0]))):.2f}")
print(f"Corrupted pattern energy: {-(0.5 * np.dot(corrupted.T, np.dot(W, corrupted))):.2f}")

recalled = recall(W, corrupted)
print(f"Recalled pattern energy: {-(0.5 * np.dot(recalled.T, np.dot(W, recalled))):.2f}")
if np.array_equal(recalled, patterns[0]):
    print("SUCCESS: Network converged back to the correct stored pattern!\n")
else:
    print("FAILED: Did not converge to the exact original pattern.\n")


print("--- 2. Recurrent Neural Network (Anomaly Detection) ---")
print("Training RNN on synthetic dataset...")
model = train_rnn()

print("Testing RNN on a sample reading window...")
sample_window = [
    {"vibration": 0.5, "current": 1.0, "temperature": 30.0},
    {"vibration": 0.6, "current": 1.1, "temperature": 30.5},
    {"vibration": 2.5, "current": 3.2, "temperature": 45.0}, # Spike
    {"vibration": 2.6, "current": 3.3, "temperature": 46.0}
]
# Pad to 10
while len(sample_window) < 10:
    sample_window.append({"vibration": 2.6, "current": 3.3, "temperature": 46.0})

is_anomaly = rnn_predict_anomaly(sample_window, model)
print(f"Prediction for new sequence (Is Anomaly?): {is_anomaly}")
print("\nModule D Execution Complete!")
