import os
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Configuration
BATCH_SIZE = 32
EPOCHS = 80  # User requested 80 epochs for better accuracy
IMAGE_SIZE = (299, 299)
vocab_size = None
max_length = None

print("🚀 Starting InceptionV3 Training Pipeline...")
print(f"TensorFlow Version: {tf.__version__}")

# 1. Load Data
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
        image_id = image_id.split('.')[0]
        if image_id not in captions:
            captions[image_id] = []
        captions[image_id].append(caption)
    return captions

def clean_captions(captions):
    import string
    table = str.maketrans('', '', string.punctuation)
    for key, caption_list in captions.items():
        for i in range(len(caption_list)):
            caption = caption_list[i]
            caption = caption.lower()
            caption = caption.translate(table)
            caption = ' '.join([word for word in caption.split() if len(word) > 1])
            caption = 'startseq ' + caption + ' endseq'
            caption_list[i] = caption

print("📂 Loading captions...")
captions_file = 'data/Flickr8k.token.txt'
text = load_captions(captions_file)
captions = create_captions_dict(text)
clean_captions(captions)
print(f"✅ Loaded {len(captions)} images with captions")

# 2. Extract Features with InceptionV3
def extract_features(directory):
    model = InceptionV3(weights='imagenet')
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    print(model.summary())
    
    features = {}
    image_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    
    for image_file in tqdm(image_files, desc="Extracting Features"):
        image_path = os.path.join(directory, image_file)
        image = load_img(image_path, target_size=IMAGE_SIZE)
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        
        feature = model.predict(image, verbose=0)
        image_id = image_file.split('.')[0]
        features[image_id] = feature
        
    return features

features_file = 'features_inception.pkl'
images_path = 'data/Images/'

if os.path.exists(features_file):
    print("📦 Loading existing InceptionV3 features...")
    with open(features_file, 'rb') as f:
        features = pickle.load(f)
else:
    print("🖼️ Extracting features with InceptionV3 (This takes time!)...")
    features = extract_features(images_path)
    with open(features_file, 'wb') as f:
        pickle.dump(features, f)

print(f"✅ Features shape: {features[list(features.keys())[0]].shape}") # Should be (1, 2048)

# 3. Prepare Tokenizer
all_captions = []
for key in captions:
    for caption in captions[key]:
        all_captions.append(caption)

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in all_captions)

print(f"📚 Vocab Size: {vocab_size}")
print(f"📏 Max Length: {max_length}")

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Save config
import json
with open('config.json', 'w') as f:
    json.dump({'max_length': max_length, 'vocab_size': vocab_size, 'model_type': 'inception_v3'}, f)

# 4. Prepare Data Generator
train_images_file = 'data/Flickr_8k.trainImages.txt'
with open(train_images_file, 'r') as f:
    train_ids = set([line.split('.')[0] for line in f.read().split('\n') if line])

print(f"🚂 Training on {len(train_ids)} images")

def data_generator(captions, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = [], [], []
    n = 0
    while True:
        for image_id, caption_list in captions.items():
            if image_id not in train_ids or image_id not in features: continue
            
            feature = features[image_id][0] # (2048,)
            
            for caption in caption_list:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
                    n += 1
                    if n == batch_size:
                        yield (np.array(X1).astype('float32'), np.array(X2).astype('int32')), np.array(y).astype('float32')
                        X1, X2, y = [], [], []
                        n = 0

# 5. Define Model (InceptionV3 adapted)
inputs1 = Input(shape=(2048,)) # InceptionV3 feature size
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(512, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 512, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(512)(se2)

decoder1 = Add()([fe2, se3])
decoder2 = Dense(512, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# 6. Train using tf.data.Dataset
steps = len(train_ids) * 5 // BATCH_SIZE

def generator_wrapper():
    gen = data_generator(captions, features, tokenizer, max_length, vocab_size, BATCH_SIZE)
    for batch in gen:
        yield batch

dataset = tf.data.Dataset.from_generator(
    generator_wrapper,
    output_signature=(
        (
            tf.TensorSpec(shape=(None, 2048), dtype=tf.float32),
            tf.TensorSpec(shape=(None, max_length), dtype=tf.int32)
        ),
        tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32)
    )
)

checkpoint = ModelCheckpoint('best_model_inception.keras', monitor='loss', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='loss', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose=1)

print("🏃‍♂️ Start Training...")
model.fit(
    dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

model.save('final_model_inception.keras')
print("🎉 Training Complete!")
