import tensorflow as tf
import matplotlib.pyplot as plt
import collections
import random
import numpy as np
import os
import time
import json
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Configuration ---
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EMBEDDING_DIM = 256
UNITS = 512
EPOCHS = 20
IMAGE_SIZE = (380, 380)
VOCAB_SIZE = 5000  # Will be set dynamically
MAX_LENGTH = 50   # Will be set dynamically
ATTENTION_FEATURES_SHAPE = 64

print(f"TensorFlow Version: {tf.__version__}")

# --- 1. Data Loading ---
def load_captions(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text

def create_captions_dict(text):
    captions = {}
    for line in text.split('\n'):
        tokens = line.split('\t')
        if len(tokens) < 2: continue
        image_id, caption = tokens[0], tokens[1]
        image_id = image_id.split('.')[0] # Remove file extension
        if image_id not in captions:
            captions[image_id] = []
        # Add start and end tokens
        caption = f"<start> {caption} <end>"
        captions[image_id].append(caption)
    return captions

print("📂 Loading captions...")
captions_file = 'data/Flickr8k.token.txt'
text = load_captions(captions_file)
captions_dict = create_captions_dict(text)

# Flatten captions for tokenizer
all_captions = []
image_path_to_caption = []
images_dir = 'data/Images/'

all_img_name_vector = []
for image_id, caption_list in captions_dict.items():
    full_image_path = os.path.join(images_dir, image_id + '.jpg')
    if os.path.exists(full_image_path):
        for caption in caption_list:
            all_captions.append(caption)
            all_img_name_vector.append(full_image_path)
    else:
        pass # Skip missing images

# --- 2. Tokenizer ---
print("🔤 Tokenizing...")
tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_LENGTH,
    standardize=None # Captions are already clean-ish or we want raw
)

# Learn vocabulary
tokenizer.adapt(all_captions)

# Save tokenizer mapping
import pickle
word_to_index = tf.keras.layers.StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary())
index_to_word = tf.keras.layers.StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary(), invert=True)

with open('tokenizer_attention.pkl', 'wb') as f:
    pickle.dump({'vocab': tokenizer.get_vocabulary(), 'config': tokenizer.get_config()}, f)

# Create Training Data
caption_vector = tokenizer(all_captions)

# Split Train/Test
img_name_train, img_name_val, cap_train, cap_val = train_test_split(all_img_name_vector, caption_vector.numpy(), test_size=0.2, random_state=0)

print(f"Train samples: {len(img_name_train)}")
print(f"Val samples: {len(img_name_val)}")

# --- 3. Feature Extraction (InceptionV3) ---
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (380, 380)) # EfficientNetB4 expects 380x380
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img, image_path

image_model = tf.keras.applications.EfficientNetB4(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output # Shape will be distinct for B4
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# Caching features to disk (Important for Attention)
# We need to process unique images only
unique_images = sorted(set(all_img_name_vector))
print(f"Unique images to process: {len(unique_images)}")

# Check if features are already cached
images_to_extract = [p for p in unique_images if not os.path.exists(p + '_effnet.npy')]

if len(images_to_extract) > 0:
    print(f"🖼️ Extracting features for {len(images_to_extract)} missing images (Resuming)...")
    image_dataset = tf.data.Dataset.from_tensor_slices(images_to_extract)
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(16)

    for img, path in tqdm(image_dataset):
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3])) # (16, 8*8, 2048) or similar
        
        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature + '_effnet', bf.numpy())
else:
    print("✅ All features already extracted!")

# --- 4. Dataset Pipeline ---
def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '_effnet.npy')
    return img_tensor, cap

dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int64]), num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# --- 5. Model Architecture (Attention) ---
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))

        # score shape == (batch_size, 64, 1)
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape == (batch_size, 64, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

encoder = CNN_Encoder(EMBEDDING_DIM)
decoder = RNN_Decoder(EMBEDDING_DIM, UNITS, len(tokenizer.get_vocabulary()))

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# --- 6. Training Loop ---
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

@tf.function
def train_step(img_tensor, target):
    loss = 0
    generated_tokens = [] # Just for monitoring

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([word_to_index('<start>')] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return total_loss

print("🚀 Starting Attention Training...")
loss_plot = []

for epoch in range(EPOCHS):
    start = time.time()
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss = train_step(img_tensor, target)
        total_loss += batch_loss

        if batch % 100 == 0:
            print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')

    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / len(img_name_train))

    if (epoch + 1) % 5 == 0:
        ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch+1}')

    print(f'Epoch {epoch+1} Loss {total_loss/len(img_name_train):.4f}')
    print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')

print("🎉 Attention Training Complete!")
