import pickle
import numpy as np

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Check what the generator actually produces
print("Testing generator output types...")
print(f"Vocab size: {len(tokenizer.word_index) + 1}")

# Create a simple test
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Simulate what data_generator does
test_seq = [1, 2, 3, 4, 5]
padded = pad_sequences([test_seq], maxlen=40)[0]
one_hot = to_categorical([10], num_classes=8811)[0]

print(f"\nPadded sequence dtype: {padded.dtype}")
print(f"Padded sequence shape: {padded.shape}")
print(f"\nOne-hot dtype: {one_hot.dtype}")
print(f"One-hot shape: {one_hot.shape}")

# Test with numpy array
feature = np.random.rand(4096)
print(f"\nFeature dtype: {feature.dtype}")
print(f"Feature shape: {feature.shape}")
