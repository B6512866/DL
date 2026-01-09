from flask import Flask, request, render_template_string, jsonify, send_file
import numpy as np
import pickle
from PIL import Image
# Switch to InceptionV3
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import tempfile
import base64
from io import BytesIO
from pyngrok import ngrok

app = Flask(__name__)

# Load model and tokenizer
print("Loading model and tokenizer...")

# Check for InceptionV3 model first, else fallback
if os.path.exists('best_model_inception.keras'):
    print("✨ Loading InceptionV3 Model")
    model = load_model('best_model_inception.keras')
    # Load InceptionV3 feature extractor
    inc_model = InceptionV3(weights='imagenet')
    feature_extractor = Model(inputs=inc_model.inputs, outputs=inc_model.layers[-2].output)
    IMAGE_SIZE = (299, 299)
else:
    print("⚠️ InceptionV3 model not found, falling back to VGG16")
    if os.path.exists('best_model.keras'):
        model = load_model('best_model.keras')
    else:
        print("❌ No model found! Please run training script first.")
        model = None
        
    # Load VGG16 feature extractor
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
    vgg_model = VGG16()
    feature_extractor = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    preprocess_input = vgg_preprocess # Override preprocessing
    IMAGE_SIZE = (224, 224)

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Get max_length from tokenizer config
try:
    with open('config.json', 'r') as f:
        import json
        config = json.load(f)
        max_length = config['max_length']
except:
    max_length = 34 # Default fallback

print(f"✅ Model loaded! Max length: {max_length}")

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace('startseq', '').strip()

def translate_to_thai(text):
    try:
        return GoogleTranslator(source='en', target='th').translate(text)
    except Exception as e:
        return f"Translation error: {str(e)}"

def text_to_speech_thai(text):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_path = temp_file.name
        temp_file.close()
        
        tts = gTTS(text=text, lang='th')
        tts.save(temp_path)
        
        return temp_path
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        return None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📸 Image Captioning AI</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover {
            background: #f0f4ff;
            border-color: #764ba2;
        }
        .upload-area input {
            display: none;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 18px;
            border-radius: 10px;
            cursor: pointer;
            transition: transform 0.2s;
            display: block;
            margin: 20px auto;
        }
        .btn:hover {
            transform: scale(1.05);
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .result {
            margin-top: 30px;
            display: none;
        }
        .result.show {
            display: block;
        }
        .caption-box {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
        }
        .caption-box h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        .caption-text {
            font-size: 18px;
            line-height: 1.6;
            color: #333;
        }
        .preview-img {
            max-width: 100%;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }
        .loading.show {
            display: block;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        audio {
            width: 100%;
            margin-top: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📸 Image Captioning AI</h1>
        <p class="subtitle">อัพโหลดรูปภาพและรับคำบรรยายพร้อมเสียงภาษาไทย!</p>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <p style="font-size: 48px; margin-bottom: 10px;">📤</p>
            <p style="font-size: 18px; color: #667eea;">คลิกเพื่ออัพโหลดรูปภาพ</p>
            <p style="color: #999; margin-top: 10px;">รองรับ JPG, PNG</p>
            <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(event)">
        </div>
        
        <div id="preview"></div>
        
        <button class="btn" id="generateBtn" onclick="generateCaption()" disabled>
            🔮 สร้างคำบรรยายและเสียง
        </button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 15px; color: #667eea;">กำลังประมวลผล...</p>
        </div>
        
        <div class="result" id="result">
            <div class="caption-box">
                <h3>🇬🇧 English Caption</h3>
                <p class="caption-text" id="captionEn"></p>
            </div>
            
            <div class="caption-box">
                <h3>🇹🇭 คำบรรยายภาษาไทย</h3>
                <p class="caption-text" id="captionTh"></p>
            </div>
            
            <div class="caption-box">
                <h3>🔊 เสียงบรรยายภาษาไทย</h3>
                <audio id="audio" controls autoplay></audio>
            </div>
        </div>
    </div>
    
    <script>
        let selectedFile = null;
        
        function handleFileSelect(event) {
            selectedFile = event.target.files[0];
            if (selectedFile) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('preview').innerHTML = 
                        '<img src="' + e.target.result + '" class="preview-img">';
                    document.getElementById('generateBtn').disabled = false;
                };
                reader.readAsDataURL(selectedFile);
            }
        }
        
        async function generateCaption() {
            if (!selectedFile) return;
            
            document.getElementById('loading').classList.add('show');
            document.getElementById('result').classList.remove('show');
            document.getElementById('generateBtn').disabled = true;
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                document.getElementById('captionEn').textContent = data.caption_en;
                document.getElementById('captionTh').textContent = data.caption_th;
                document.getElementById('audio').src = data.audio_url;
                
                document.getElementById('loading').classList.remove('show');
                document.getElementById('result').classList.add('show');
                document.getElementById('generateBtn').disabled = false;
            } catch (error) {
                alert('เกิดข้อผิดพลาด: ' + error.message);
                document.getElementById('loading').classList.remove('show');
                document.getElementById('generateBtn').disabled = false;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        img = img.resize(IMAGE_SIZE) # Use dynamic size
        img_array = keras.utils.img_to_array(img)
        img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
        
        # Use appropriate preprocessing
        img_array = preprocess_input(img_array)
        
        # Extract features
        features = feature_extractor.predict(img_array, verbose=0)
        
        # Generate caption
        caption_en = generate_caption(model, tokenizer, features, max_length)
        caption_en_formatted = caption_en.capitalize() + "."
        
        # Translate
        caption_th = translate_to_thai(caption_en)
        
        # Generate speech
        audio_path = text_to_speech_thai(caption_th)
        
        return jsonify({
            'caption_en': caption_en_formatted,
            'caption_th': caption_th,
            'audio_url': f'/audio/{os.path.basename(audio_path)}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    temp_dir = tempfile.gettempdir()
    return send_file(os.path.join(temp_dir, filename), mimetype='audio/mpeg')

if __name__ == '__main__':
    # Start ngrok
    try:
        public_url = ngrok.connect(5000).public_url
        print("\n" + "="*70)
        print("🚀 App is Online! / แอปออนไลน์แล้ว!")
        print(f"🌍 Public URL: {public_url}")
        print("📍 Local URL:  http://localhost:5000")
        print("="*70 + "\n")
    except Exception as e:
        print(f"Ngrok Error: {e}")

    app.run(host='0.0.0.0', port=5000, debug=False)
