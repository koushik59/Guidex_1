from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import time
import pyttsx3
from ultralytics import YOLO
import threading
import queue
import base64
import numpy as np
from io import BytesIO
import os
import atexit
import signal
import easyocr
import json
import urllib.error
import urllib.request

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Change to script directory to ensure relative paths work
os.chdir(BASE_DIR)

print(f"[DEBUG] BASE_DIR: {BASE_DIR}")
print(f"[DEBUG] TEMPLATE_DIR: {TEMPLATE_DIR}")
print(f"[DEBUG] STATIC_DIR: {STATIC_DIR}")
print(f"[DEBUG] Current working directory: {os.getcwd()}")
print(f"[DEBUG] Templates exist: {os.path.exists(TEMPLATE_DIR)}")
print(f"[DEBUG] index.html exists: {os.path.exists(os.path.join(TEMPLATE_DIR, 'index.html'))}")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
CORS(app)

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
DETECTION_WIDTH = 416
DETECTION_INTERVAL = 0.18
STREAM_FPS = 24
JPEG_QUALITY = 70
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# ------------------ Setup ------------------
DANGEROUS_CLASSES = [
    "person", "car", "bus", "truck", "motorcycle", "bicycle",
    "chair", "couch", "potted plant", "bed", "bench", "dining table",
    "tv", "laptop", "sink", "refrigerator", "toilet", "umbrella",
    "backpack", "handbag", "suitcase", "fire hydrant", "stop sign",
    "traffic light", "pothole", "stairs", "water puddle", "construction zone",
    "book", "cell phone"
]

LARGE_VEHICLES = ["car", "bus", "truck"]
MEDIUM_VEHICLES = ["motorcycle"]
OBSTACLES = ["chair", "couch", "bed", "bench", "dining table", "refrigerator", "toilet", 
             "fire hydrant", "stop sign", "pothole", "stairs", "water puddle", "construction zone"]
SMALL_OBJECTS = ["person", "bicycle", "potted plant", "tv", "laptop", "sink", 
                 "umbrella", "backpack", "handbag", "suitcase", "traffic light", "book", "cell phone"]

model = YOLO("yolov8n.pt")

class SilentSpeechEngine:
    def say(self, message):
        print(f"[TTS disabled to prevent echo] {message}")

    def runAndWait(self):
        return None

    def stop(self):
        return None

# Mute local server speech synthesis by default to prevent double-speech conflict with the browser
engine = SilentSpeechEngine()

# Initialize EasyOCR
print("Initializing EasyOCR (this may take a moment to download models on first run)...")
reader = easyocr.Reader(['en'])
print("EasyOCR initialized.")

# Global state
alert_queue = queue.Queue()
audio_alert_queue = queue.Queue()
camera = None
is_running = False
latest_detections = []
latest_frame = None
latest_annotated_frame = None
latest_frame_lock = threading.Lock()
camera_thread = None
detection_thread = None
camera_stop_event = threading.Event()
detection_stop_event = threading.Event()

# Alert / environment configuration
alert_mode = "english"
environment_mode = "outdoor"
current_lang = "en"

# Per-object-type cooldown
COOLDOWN_MAP = {
    "large_vehicle": 2.0,
    "medium_vehicle": 3.0,
    "small_object": 5.0,
}

last_alert_times = {
    "large_vehicle": 0.0,
    "medium_vehicle": 0.0,
    "small_object": 0.0,
}

object_track_state = {}

# ------------------ Config File Handling ------------------
CONFIG_FILE = "config.json"
config_data = {}

def load_config():
    global config_data, GEMINI_API_KEY
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config_data = json.load(f)
            print(f"[CONFIG] Loaded config: {config_data}")
        except Exception as e:
            print(f"[CONFIG] Error reading config: {e}")
            config_data = {}
    else:
        config_data = {}
    
    # Auto-load key environment variables into config_data if not set
    if not config_data.get("openrouter_key"):
        config_data["openrouter_key"] = os.getenv("OPENROUTER_API_KEY")
    if not config_data.get("gemini_key"):
        config_data["gemini_key"] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
    GEMINI_API_KEY = config_data.get("gemini_key")

load_config()

# ------------------ Face Recognition Database ------------------
FACE_DIR = "known_faces"
known_face_templates = {}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def train_face_recognizer():
    global known_face_templates
    print("[Face Recognition] Loading face templates...")
    known_face_templates = {}
    
    if not os.path.exists(FACE_DIR):
        os.makedirs(FACE_DIR)
        
    for filename in os.listdir(FACE_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0]
            filepath = os.path.join(FACE_DIR, filename)
            img = cv2.imread(filepath)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(detected) > 0:
                # Use the largest detected face
                (x, y, w, h) = max(detected, key=lambda f: f[2] * f[3])
                face_crop = gray[y:y+h, x:x+w]
                face_crop = cv2.resize(face_crop, (100, 100))
                face_crop = cv2.equalizeHist(face_crop)
                known_face_templates[name] = face_crop
                print(f"[Face Recognition] Registered template for '{name}'.")
            else:
                # Fallback to whole image if no face detected
                face_crop = cv2.resize(gray, (100, 100))
                face_crop = cv2.equalizeHist(face_crop)
                known_face_templates[name] = face_crop
                print(f"[Face Recognition] Registered whole-image template for '{name}' (no face auto-detected).")
                
    print(f"[Face Recognition] Loaded {len(known_face_templates)} templates: {list(known_face_templates.keys())}")

# Train face recognizer on start
train_face_recognizer()

LABEL_TRANSLATIONS = {
    "hi": {
        "person": "व्यक्ति", "car": "कार", "bus": "बस", "truck": "ट्रक", "motorcycle": "मोटरसाइकिल",
        "bicycle": "साइकिल", "chair": "कुर्सी", "couch": "सोफा", "potted plant": "पौधा", "bed": "बिस्तर",
        "bench": "बेंच", "dining table": "मेज़", "tv": "टीवी", "laptop": "लैपटॉप", "sink": "सिंक",
        "refrigerator": "फ्रिज", "toilet": "शौचालय", "umbrella": "छाता", "backpack": "बैग",
        "handbag": "पर्स", "suitcase": "सूटकेश", "fire hydrant": "फायर हाइड्रेंट", "stop sign": "स्टॉप साइन",
        "traffic light": "यातायात सिग्नल", "pothole": "गड्ढा", "stairs": "सीढ़ियाँ", "water puddle": "पानी का गड्ढा",
        "construction zone": "निर्माण क्षेत्र", "book": "किताब", "cell phone": "फ़ोन"
    },
    "te": {
        "person": "వ్యక్తి", "car": "కారు", "bus": "బస్సు", "truck": "ట్రక్కు", "motorcycle": "మోటార్ సైకిల్",
        "bicycle": "సైకిల్", "chair": "కుర్చీ", "couch": "సోఫా", "potted plant": "మొక్క", "bed": "మంచం",
        "bench": "బెంచ్", "dining table": "భోజన బల్ల", "tv": "టీవీ", "laptop": "ల్యాప్‌టాప్", "sink": "సింక్",
        "refrigerator": "ఫ్రిజ్", "toilet": "టాయిలెట్", "umbrella": "గొడుగు", "backpack": "బ్యాక్‌ప్యాక్",
        "handbag": "హ్యాండ్‌బ్యాగ్", "suitcase": "సూట్‌కేస్", "fire hydrant": "ఫైర్ హైడ్రెంట్", "stop sign": "స్టాప్ సైన్",
        "traffic light": "ట్రాఫిక్ సిగ్నల్", "pothole": "గుంత", "stairs": "మెట్లు", "water puddle": "నీటి గుంత",
        "construction zone": "నిర్మాణ ప్రాంతం", "book": "పుస్తకం", "cell phone": "సెల్ ఫోన్"
    }
}

def translate_text(text, target_lang):
    if not text:
        return ""
    if target_lang == 'en':
        return text
    
    gemini_key = config_data.get("gemini_key") or GEMINI_API_KEY
    openrouter_key = config_data.get("openrouter_key")
    openrouter_model = config_data.get("openrouter_model") or "google/gemini-2.5-flash"
    
    lang_names = {"en": "English", "hi": "Hindi (हिंदी)", "te": "Telugu (తెలుగు)"}
    target_lang_name = lang_names.get(target_lang, "English")
    
    prompt = f"Translate the following text to {target_lang_name}. Output ONLY the translated text, do not add any quotes, explanation, or markdown wrappers: {text}"
    
    if openrouter_key:
        url = "https://openrouter.ai/api/v1/chat/completions"
        payload = {
            "model": openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 300
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openrouter_key}",
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[Translation] OpenRouter failed: {e}")
            
    elif gemini_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 300,
            }
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": gemini_key,
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
                parts = result["candidates"][0]["content"]["parts"]
                return " ".join(part.get("text", "") for part in parts).strip()
        except Exception as e:
            print(f"[Translation] Gemini failed: {e}")
            
    return text

def get_location_description(location, lang):
    if not location:
        return ""
        
    gemini_key = config_data.get("gemini_key") or GEMINI_API_KEY
    openrouter_key = config_data.get("openrouter_key")
    openrouter_model = config_data.get("openrouter_model") or "google/gemini-2.5-flash"
    
    lang_names = {"en": "English", "hi": "Hindi (हिंदी)", "te": "Telugu (తెలుగు)"}
    target_lang_name = lang_names.get(lang, "English")
    
    prompt = (
        f"Give a brief, helpful 1-2 sentence description of the place '{location}' for a visually impaired user navigating there. "
        f"Output ONLY the description in the language {target_lang_name}. Do not include markdown or quotes."
    )
    
    if openrouter_key:
        url = "https://openrouter.ai/api/v1/chat/completions"
        payload = {
            "model": openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 120
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openrouter_key}",
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[Location Desc] OpenRouter failed: {e}")
            
    elif gemini_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 120,
            }
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": gemini_key,
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                result = json.loads(response.read().decode("utf-8"))
                parts = result["candidates"][0]["content"]["parts"]
                return " ".join(part.get("text", "") for part in parts).strip()
        except Exception as e:
            print(f"[Location Desc] Gemini failed: {e}")
            
    if lang == 'hi':
        return f"{location} एक प्रसिद्ध गंतव्य स्थान है।"
    elif lang == 'te':
        return f"{location} అనేది ఒక ముఖ్యమైన ప్రాంతం."
    return f"{location} is a notable destination."

def generate_alert_message(label, category, level, direction, speed_mps=0.0, target_lang="en"):
    directions_map = {
        "en": {"left": "left", "right": "right", "center": "center"},
        "hi": {"left": "बाईं ओर", "right": "दाईं ओर", "center": "बीच में"},
        "te": {"left": "ఎడమ వైపు", "right": "కుడి వైపు", "center": "మధ్యలో"}
    }
    
    dir_word = directions_map.get(target_lang, directions_map["en"]).get(direction, direction)
    
    if target_lang == "hi":
        if "green traffic light" in label:
            return "सिग्नल हरा है, पार करना सुरक्षित है।"
        elif "red traffic light" in label:
            return "सिग्नल लाल है, पार न करें।"
        elif "familiar face" in label:
            name = label.replace("familiar face ", "")
            return f"आपके {dir_word} परिचित चेहरा {name} देखा गया है।"
        elif category == "large_vehicle":
            if level == "HIGH":
                return f"आपके {dir_word} बड़ा वाहन बहुत करीब है। कृपया रुकें।"
            else:
                return f"आपके {dir_word} से बड़ा वाहन आ रहा है। सावधान रहें।"
        elif category == "medium_vehicle":
            return f"आपके {dir_word} मोटरसाइकिल बहुत करीब है। कृपया रुकें।"
        elif label == "person":
            if speed_mps > 0.3:
                return f"आपके {dir_word} से व्यक्ति तेज़ी से आ रहा है।"
            elif speed_mps < -0.3:
                return f"आपके {dir_word} से व्यक्ति दूर जा रहा है।"
            else:
                return f"आपके {dir_word} स्थिर व्यक्ति है।"
        else:
            translated_label = LABEL_TRANSLATIONS["hi"].get(label, label)
            return f"आपके {dir_word} {translated_label} बहुत करीब है। कृपया रुकें।"
            
    elif target_lang == "te":
        if "green traffic light" in label:
            return "సిగ్నల్ ఆకుపచ్చగా ఉంది, దాటడం సురక్షితం."
        elif "red traffic light" in label:
            return "సిగ్నల్ ఎరుపు రంగులో ఉంది, దాటవద్దు."
        elif "familiar face" in label:
            name = label.replace("familiar face ", "")
            return f"మీ {dir_word} లో పరిచయస్తుడు {name} గుర్తించబడ్డారు."
        elif category == "large_vehicle":
            if level == "HIGH":
                return f"మీ {dir_word} లో పెద్ద వాహనం చాలా దగ్గరగా ఉంది. దయచేసి ఆగండి."
            else:
                return f"మీ {dir_word} నుండి పెద్ద వాహనం వస్తోంది. జాగ్రత్తగా ఉండండి."
        elif category == "medium_vehicle":
            return f"మీ {dir_word} లో మోటార్ సైకిల్ చాలా దగ్గరగా ఉంది. దయచేసి ఆగండి."
        elif label == "person":
            if speed_mps > 0.3:
                return f"మీ {dir_word} నుండి ఒక వ్యక్తి వేగంగా వస్తున్నాడు."
            elif speed_mps < -0.3:
                return f"మీ {dir_word} నుండి ఒక వ్యక్తి దూరంగా వెళ్తున్నాడు."
            else:
                return f"మీ {dir_word} లో ఒక వ్యక్తి నిలబడి ఉన్నాడు."
        else:
            translated_label = LABEL_TRANSLATIONS["te"].get(label, label)
            return f"మీ {dir_word} లో {translated_label} చాలా దగ్గరగా ఉంది. దయచేసి ఆగండి."
            
    else: # English
        if "green traffic light" in label:
            return "Signal is green, safe to cross."
        elif "red traffic light" in label:
            return "Signal is red, do not cross."
        elif "familiar face" in label:
            name = label.replace("familiar face ", "")
            return f"Familiar face {name} detected on your {direction}."
        elif category == "large_vehicle":
            if level == "HIGH":
                return f"Large vehicle very close on your {direction}. Please stop."
            else:
                return f"Large vehicle approaching from your {direction}. Be cautious."
        elif category == "medium_vehicle":
            return f"Motorcycle very close on your {direction}. Please stop."
        elif label == "person":
            if speed_mps > 0.3:
                return f"Person approaching fast on your {direction}."
            elif speed_mps < -0.3:
                return f"Person moving away on your {direction}."
            else:
                return f"Stationary person on your {direction}."
        else:
            return f"{label} very close on your {direction}. Please stop."

# ------------------ Translation Responses ------------------
RESPONSES = {
    "en": {
        "help": "How can I help you?",
        "navigating": "Finding a walking route to {destination}.",
        "repeat": "Repeating your current direction.",
        "location": "Checking your current location.",
        "clear": "Clearing the current route.",
        "start": "Starting obstacle detection now.",
        "stop": "Stopping obstacle detection now.",
        "vision": "Using Smart Look for a deeper scene check.",
        "describe": "Let me describe what is in front of you.",
        "read": "I will scan the camera view and read any text I can find.",
        "indoor": "Switching to indoor mode.",
        "outdoor": "Switching to outdoor mode.",
        "sos": "Activating SOS.",
        "no_key": "Please configure your API key in Settings to use advanced AI."
    },
    "hi": {
        "help": "मैं आपकी क्या मदद कर सकता हूँ?",
        "navigating": "{destination} के लिए पैदल मार्ग खोज रहा हूँ।",
        "repeat": "आपका वर्तमान मार्ग निर्देश दोहरा रहा हूँ।",
        "location": "आपके वर्तमान स्थान की जाँच कर रहा हूँ।",
        "clear": "वर्तमान मार्ग को हटा रहा हूँ।",
        "start": "बाधा पहचान अभी शुरू की जा रही है।",
        "stop": "बाधा पहचान अभी बंद की जा रही है।",
        "vision": "विस्तृत दृश्य जांच के लिए स्मार्ट लुक का उपयोग कर रहा हूँ।",
        "describe": "मुझे आपके सामने का दृश्य बताने दें।",
        "read": "मैं कैमरा दृश्य को स्कैन करूँगा और कोई भी उपलब्ध पाठ पढ़ूँगा।",
        "indoor": "इंडोर मोड पर स्विच कर रहा हूँ।",
        "outdoor": "आउटडोर मोड पर स्विच कर रहा हूँ।",
        "sos": "आपातकालीन एसओएस सक्रिय किया जा रहा है।",
        "no_key": "कृपया उन्नत एआई का उपयोग करने के लिए सेटिंग्स में अपनी एपीआई कुंजी कॉन्फ़िगर करें।"
    },
    "te": {
        "help": "నేను మీకు ఎలా సహాయం చేయగలను?",
        "navigating": "{destination} కు నడక మార్గాన్ని కనుగొంటున్నాను.",
        "repeat": "మీ ప్రస్తుత నావిగేషన్ సూచనను పునరావృతం చేస్తున్నాను.",
        "location": "మీ ప్రస్తుత స్థానాన్ని తనిఖీ చేస్తున్నాను.",
        "clear": "ప్రస్తుత మార్గాన్ని తొలగిస్తున్నాను.",
        "start": "అడ్డంకుల గుర్తింపును ప్రారంభిస్తున్నాను.",
        "stop": "అడ్డంకుల గుర్తింపును ఆపివేస్తున్నాను.",
        "vision": "స్మార్ట్ లుక్ ఉపయోగించి కెమెరాని పరీక్షిస్తున్నాను.",
        "describe": "మీ ముందు ఉన్న దృశ్యాన్ని వివరిస్తాను.",
        "read": "నేను కెమెరా వ్యూని స్కాన్ చేసి అందులోని వచనాన్ని చదువుతాను.",
        "indoor": "ఇండోర్ మోడ్‌కు మారుస్తున్నాను.",
        "outdoor": "అవుట్‌డోర్ మోడ్‌కు మారుస్తున్నాను.",
        "sos": "అత్యవసర ఎస్ఓఎస్ ప్రారంభించబడుతోంది.",
        "no_key": "దయచేసి సెట్టింగ్స్‌లో మీ API కీని కాన్ఫిగర్ చేయండి."
    }
}

# ------------------ Helper Functions ------------------

def estimate_distance(box_height, frame_height):
    if box_height == 0:
        return 999
    return (frame_height / box_height) * 0.5

def get_object_category(label):
    if label in LARGE_VEHICLES:
        return "large_vehicle"
    elif label in MEDIUM_VEHICLES:
        return "medium_vehicle"
    elif label in SMALL_OBJECTS:
        return "small_object"
    else:
        return "small_object"

def danger_level(distance, object_category):
    if object_category == "large_vehicle":
        if distance < 25:
            return "HIGH"
        elif distance < 45:
            return "MEDIUM"
        else:
            return "LOW"
    elif object_category == "medium_vehicle":
        if distance < 15:
            return "HIGH"
        elif distance < 30:
            return "MEDIUM"
        else:
            return "LOW"
    else:
        if distance < 1.5:
            return "HIGH"
        elif distance < 3:
            return "MEDIUM"
        else:
            return "LOW"

def get_direction(x1, x2, frame_width):
    center_x = (x1 + x2) / 2
    if center_x < frame_width / 3:
        return "left"
    elif center_x < 2 * frame_width / 3:
        return "center"
    else:
        return "right"

def compute_priority(level, object_category, speed_mps, label=""):
    level_factor = {"LOW": 1.0, "MEDIUM": 2.0, "HIGH": 3.0}.get(level, 1.0)
    type_factor = {
        "large_vehicle": 3.0,
        "medium_vehicle": 2.0,
        "small_object": 1.0,
    }.get(object_category, 1.0)

    speed = float(speed_mps or 0.0)
    speed_factor = 1.0 + min(abs(speed), 10.0) / 5.0

    if label == "person":
        if speed > 0.3:
            speed_factor *= 3.0
        elif speed < -0.3:
            speed_factor *= 0.2

    env_factor = 1.0
    if environment_mode == "outdoor":
        if object_category in ("large_vehicle", "medium_vehicle"):
            env_factor = 1.3
    elif environment_mode == "indoor":
        if object_category == "small_object":
            env_factor = 1.3

    return level_factor * type_factor * speed_factor * env_factor

def queue_alert(alert_dict):
    alert_queue.put(alert_dict)
    audio_alert_queue.put(alert_dict)

def get_latest_camera_frame():
    with latest_frame_lock:
        if latest_frame is not None:
            return latest_frame.copy()
    return None

def encode_frame_for_vision(frame, max_width=768):
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(frame, (max_width, int(height * scale)))

    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")

def call_vision_llm(frame, user_prompt=""):
    """Ask VLM to describe the latest camera scene."""
    openrouter_key = config_data.get("openrouter_key")
    openrouter_model = config_data.get("openrouter_model") or "google/gemini-2.5-flash"
    gemini_key = config_data.get("gemini_key") or GEMINI_API_KEY
    system_instruction = config_data.get("system_instruction") or (
        "You are Mickey, an assistive vision guide for a blind person. "
        "Analyze the camera image and give concise, practical guidance. "
        "Mention immediate hazards first, then useful navigation cues, then any readable signs or text. "
        "Use short spoken sentences. Do not overclaim. If uncertain, say so. "
        "Do not replace the user's cane, guide dog, or human judgment."
    )

    image_base64 = encode_frame_for_vision(frame)
    if not image_base64:
        return None, "Could not prepare the camera image for VLM."

    detections_context = latest_detections[:8] if latest_detections else []
    
    # Direct target language prompt instructions
    lang_names = {"en": "English", "hi": "Hindi (हिंदी)", "te": "Telugu (తెలుగు)"}
    target_lang = lang_names.get(current_lang, "English")

    prompt = (
        f"{system_instruction}\n\n"
        f"You MUST formulate your response in {target_lang}.\n"
        f"User question: {user_prompt or 'What is around me and what should I be careful about?'}\n"
        f"Fast detector context: {json.dumps(detections_context)}"
    )

    if openrouter_key:
        url = "https://openrouter.ai/api/v1/chat/completions"
        payload = {
            "model": openrouter_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.2,
            "max_tokens": 220
        }
        request_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=request_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openrouter_key}",
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as response:
                result = json.loads(response.read().decode("utf-8"))
                text = result["choices"][0]["message"]["content"].strip()
                return text, None
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"[OpenRouter] HTTP error {e.code}: {body}")
            return None, f"OpenRouter VLM request failed."
        except Exception as e:
            print(f"[OpenRouter] Request error: {e}")
            return None, "OpenRouter connection failed."

    elif gemini_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_base64,
                    }},
                ],
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 220,
            },
        }
        request_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=request_data,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": gemini_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                result = json.loads(response.read().decode("utf-8"))
                parts = result["candidates"][0]["content"]["parts"]
                text = " ".join(part.get("text", "") for part in parts).strip()
                return text, None
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"[Gemini] HTTP error {e.code}: {body}")
            return None, "Gemini VLM request failed."
        except Exception as e:
            print(f"[Gemini] Request error: {e}")
            return None, "Gemini request failed."

    return None, "No API key configured. Configure OpenRouter or Gemini keys in Settings."


def call_assistant_ai(message, lang, lat=None, lng=None, active_destination=None):
    """Call Assistant LLM to interpret commands and chat."""
    openrouter_key = config_data.get("openrouter_key")
    openrouter_model = config_data.get("openrouter_model") or "google/gemini-2.5-flash"
    gemini_key = config_data.get("gemini_key") or GEMINI_API_KEY

    lang_names = {"en": "English", "hi": "Hindi (हिंदी)", "te": "Telugu (తెలుగు)"}
    target_lang = lang_names.get(lang, "English")

    system_instruction = (
        "You are Mickey, an intelligent voice assistant built into AI smart glasses for visually impaired users. "
        f"You must converse and reply in {target_lang}. Keep your response concise (1-2 short spoken sentences). "
        "Do not use markdown formatting. "
        "You must return a JSON object with 'reply' and 'action' keys. "
        "If the user wants to perform an action, map it to one of the following action names:\n"
        "- 'navigate': user wants to walk/navigate to a place. Also include 'destination' key with the place name, and 'location_description' key containing a brief 1-2 sentence description of that location in the target language.\n"
        "- 'repeat_navigation': repeat current map directions.\n"
        "- 'location': check current GPS location.\n"
        "- 'clear_route': stop/clear current route.\n"
        "- 'start': start camera/obstacle detection.\n"
        "- 'stop': stop camera/obstacle detection.\n"
        "- 'indoor': change environment mode to indoor.\n"
        "- 'outdoor': change environment mode to outdoor.\n"
        "- 'sos': trigger emergency assistance.\n"
        "- 'read': read text or document from camera.\n"
        "- 'vision': describe the environment/what is in front (Smart Look).\n"
        "- 'identify_face': check who is in front of the camera / who is in front of me.\n"
        "If no action is needed, set 'action' to null.\n"
        "Ensure your JSON is valid and has no markdown wrap."
    )

    prompt = (
        f"Context details: Current GPS: {lat}, {lng}. Active Destination: {active_destination}.\n\n"
        f"User message: {message}"
    )

    if openrouter_key:
        url = "https://openrouter.ai/api/v1/chat/completions"
        payload = {
            "model": openrouter_model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2
        }
        request_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=request_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openrouter_key}",
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                result = json.loads(response.read().decode("utf-8"))
                text = result["choices"][0]["message"]["content"].strip()
                return json.loads(text), None
        except Exception as e:
            print(f"[OpenRouter Assistant] Request failed: {e}")
            return None, "OpenRouter failed"

    elif gemini_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        payload = {
            "contents": [{
                "parts": [
                    {"text": f"SYSTEM INSTRUCTION:\n{system_instruction}\n\nUSER INPUT:\n{prompt}"}
                ]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.2
            }
        }
        request_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=request_data,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": gemini_key,
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                result = json.loads(response.read().decode("utf-8"))
                text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                return json.loads(text), None
        except Exception as e:
            print(f"[Gemini Assistant] Request failed: {e}")
            return None, "Gemini failed"

    return None, "No API key configured."


def rule_based_mickey(message, lang):
    """Fallback rule-based command interpreter in English, Hindi, and Telugu."""
    msg = message.lower()
    
    # 1. Navigation
    destination = None
    nav_keywords = [
        "navigate to", "take me to", "go to", "directions to", "route to", "guide me to",
        "मार्गदर्शन करें", "रास्ता दिखाएं", "ले चलो", "తీసుకువెళ్ళు", "దారి చూపు", "వెళ్ళు"
    ]
    for kw in nav_keywords:
        if kw in msg:
            destination = msg.split(kw, 1)[1].strip(" .?,")
            break
            
    if destination:
        reply = RESPONSES[lang]["navigating"].format(destination=destination)
        return {"reply": reply, "action": "navigate", "destination": destination}
        
    # 2. Repeat directions
    if any(kw in msg for kw in ["repeat direction", "repeat directions", "next direction", "where do i go", "what is next", "निर्देश दोहराएं", "మళ్ళీ చెప్పు"]):
        return {"reply": RESPONSES[lang]["repeat"], "action": "repeat_navigation"}
        
    # 3. Where am I
    if any(kw in msg for kw in ["where am i", "my location", "current location", "कहाँ हूँ", "నేనెక్కడ ఉన్నాను"]):
        return {"reply": RESPONSES[lang]["location"], "action": "location"}
        
    # 4. Clear route
    if any(kw in msg for kw in ["clear route", "clear navigation", "मार्ग हटाओ", "దారి క్లియర్"]):
        return {"reply": RESPONSES[lang]["clear"], "action": "clear_route"}
        
    # 5. Start detection
    if any(kw in msg for kw in ["start", "begin", "चालू करो", "ప్రారంభించు", "ఆన్ చేయి"]):
        return {"reply": RESPONSES[lang]["start"], "action": "start"}
        
    # 6. Stop detection
    if any(kw in msg for kw in ["stop", "off", "बंद करो", "ఆపు"]):
        return {"reply": RESPONSES[lang]["stop"], "action": "stop"}
        
    # 7. Indoor
    if any(kw in msg for kw in ["indoor", "घर के अंदर", "ఇండోర్"]):
        return {"reply": RESPONSES[lang]["indoor"], "action": "indoor"}
        
    # 8. Outdoor
    if any(kw in msg for kw in ["outdoor", "बाहर", "అవుట్‌డోర్"]):
        return {"reply": RESPONSES[lang]["outdoor"], "action": "outdoor"}
        
    # 9. SOS
    if any(kw in msg for kw in ["emergency", "sos", "help", "आपातकालीन", "मदद", "సహాయం", "అత్యవసర"]):
        return {"reply": RESPONSES[lang]["sos"], "action": "sos"}
        
    # 10. Read text
    if any(kw in msg for kw in ["read", "text", "written", "पढ़ो", "చదువు", "రాసి ఉంది"]):
        return {"reply": RESPONSES[lang]["read"], "action": "read"}
        
    # 11. Vision / Smart Look
    if any(kw in msg for kw in ["smart look", "describe", "see", "सामने क्या है", "ముందు ఏముంది"]):
        return {"reply": RESPONSES[lang]["describe"], "action": "vision", "prompt": message}
        
    # 12. Face / Person identification
    if any(kw in msg for kw in ["who is in front", "who is infront", "who is this", "identify person", "मेरे सामने कौन है", "నా ముందు ఎవరున్నారు"]):
        reply_txt = "Checking who is in front of you."
        if lang == 'hi':
            reply_txt = "जांच की जा रही है कि आपके सामने कौन है।"
        elif lang == 'te':
            reply_txt = "మీ ముందు ఎవరున్నారో तనిఖీ చేస్తున్నాను."
        return {"reply": reply_txt, "action": "identify_face"}
        
    # Default help reply
    return {"reply": RESPONSES[lang]["help"], "action": None}


def process_frame(frame):
    """Process a single frame and return detection results, including face recognition."""
    results = model(frame, imgsz=DETECTION_WIDTH, verbose=False)
    frame_height, frame_width, _ = frame.shape
    
    detections = []
    current_time = time.time()
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        confidence = float(box.conf[0])
        
        if label in DANGEROUS_CLASSES and confidence > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_height = y2 - y1
            distance = estimate_distance(box_height, frame_height)
            
            object_category = get_object_category(label)
            level = danger_level(distance, object_category)
            direction = get_direction(x1, x2, frame_width)

            # Traffic Signal logic
            if label == "traffic light":
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    mask_red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
                    mask_red2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
                    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
                    mask_green = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))
                    
                    if cv2.countNonZero(mask_green) > cv2.countNonZero(mask_red) and cv2.countNonZero(mask_green) > 10:
                        label = "green traffic light"
                    elif cv2.countNonZero(mask_red) > 10:
                        label = "red traffic light"
                        
                    if y2 < frame_height - 50:
                        roi_bottom = frame[y2:, max(0, x1-50):min(frame_width, x2+50)]
                        gray_bottom = cv2.cvtColor(roi_bottom, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(gray_bottom, 50, 150, apertureSize=3)
                        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
                        if lines is not None and len(lines) > 2:
                            label += " with zebra crossing"

            # Offline Face Recognition Integration
            if label == "person" and known_face_templates:
                person_roi = frame[max(0, y1):min(frame_height, y2), max(0, x1):min(frame_width, x2)]
                if person_roi.size > 0:
                    gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                    faces_in_roi = face_cascade.detectMultiScale(gray_roi, 1.1, 3)
                    if len(faces_in_roi) > 0:
                        (fx, fy, fw, fh) = max(faces_in_roi, key=lambda f: f[2] * f[3])
                        face_crop = gray_roi[fy:fy+fh, fx:fx+fw]
                        face_crop = cv2.resize(face_crop, (100, 100))
                        face_crop = cv2.equalizeHist(face_crop)
                        
                        best_name = None
                        best_score = -1.0
                        for name, template in known_face_templates.items():
                            res = cv2.matchTemplate(face_crop, template, cv2.TM_CCOEFF_NORMED)
                            score = res[0][0]
                            if score > best_score:
                                best_score = score
                                best_name = name
                        
                        if best_score > 0.65:
                            label = f"familiar face {best_name}"

            track_key = (label.split(" ")[-1], direction)
            prev_state = object_track_state.get(track_key)
            speed_mps = 0.0
            if prev_state:
                dt = current_time - prev_state["time"]
                if dt > 0:
                    speed_mps = (prev_state["distance"] - distance) / dt
            object_track_state[track_key] = {"distance": distance, "time": current_time}

            priority_score = compute_priority(level, object_category, speed_mps, label)

            detections.append({
                "label": label,
                "distance": round(distance, 2),
                "level": level,
                "direction": direction,
                "confidence": round(confidence, 2),
                "bbox": [x1, y1, x2, y2],
                "category": object_category,
                "speed": round(speed_mps, 2),
                "priority": round(priority_score, 2),
            })

    # ---------------- Priority-based alert selection ----------------
    if detections:
        best_detection = max(detections, key=lambda d: d.get("priority", 0.0))
        category = best_detection["category"]
        level = best_detection["level"]
        direction = best_detection["direction"]
        label = best_detection["label"]

        should_alert = False
        if category == "large_vehicle":
            should_alert = level in ("HIGH", "MEDIUM")
        else:
            should_alert = level == "HIGH"

        if should_alert:
            last_time_for_category = last_alert_times.get(category, 0.0)
            cooldown = COOLDOWN_MAP.get(category, 4.0)

            if current_time - last_time_for_category > cooldown:
                english_alert = generate_alert_message(label, category, level, direction, best_detection.get("speed", 0.0), "en")
                translated_alert = generate_alert_message(label, category, level, direction, best_detection.get("speed", 0.0), current_lang)
                alert_payload = {
                    "alert": english_alert,
                    "speech": translated_alert
                }
                if is_running:
                    queue_alert(alert_payload)
                last_alert_times[category] = current_time

    global latest_detections
    latest_detections = detections
    return detections

def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        level = det["level"]
        color = (0, 0, 255) if level == "HIGH" else (0, 255, 255) if level == "MEDIUM" else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{det['label']} | {det['level']} | {det['direction']}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return frame

def open_camera():
    global camera
    print("[DEBUG] Attempting to open camera...")
    for index in [0, 1, 2, 700]:
        try:
            candidate = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(index)
            candidate.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            candidate.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            candidate.set(cv2.CAP_PROP_FPS, STREAM_FPS)
            candidate.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if candidate.isOpened():
                ret, _ = candidate.read()
                if ret:
                    camera = candidate
                    print(f"[SUCCESS] Camera opened on index {index}")
                    return True
            candidate.release()
        except Exception as e:
            print(f"[ERROR] Failed to open camera index {index}: {e}")

    camera = None
    return False

def camera_capture_worker():
    global camera, latest_frame, latest_annotated_frame
    while not camera_stop_event.is_set():
        if camera is None or not camera.isOpened():
            if not open_camera():
                print("[ERROR] Could not open any camera. Retrying in 2s...")
                time.sleep(2)
                continue

        ret, frame = camera.read()
        if not ret:
            print("[WARNING] Failed to read frame from camera. Releasing...")
            try:
                if camera:
                    camera.release()
            except Exception:
                pass
            camera = None
            time.sleep(0.2)
            continue

        with latest_frame_lock:
            latest_frame = frame.copy()
            if latest_annotated_frame is None:
                latest_annotated_frame = frame.copy()

        time.sleep(1 / STREAM_FPS)

def detection_worker():
    global latest_annotated_frame
    while not detection_stop_event.is_set():
        frame_to_process = None
        with latest_frame_lock:
            if latest_frame is not None:
                frame_to_process = latest_frame.copy()

        if frame_to_process is None:
            time.sleep(0.05)
            continue

        if is_running:
            height, width = frame_to_process.shape[:2]
            scale = 1.0
            if width > DETECTION_WIDTH:
                scale = DETECTION_WIDTH / width
                resized = cv2.resize(frame_to_process, (DETECTION_WIDTH, int(height * scale)))
            else:
                resized = frame_to_process

            detections = process_frame(resized)
            if scale != 1.0:
                for det in detections:
                    det["bbox"] = [int(coord / scale) for coord in det["bbox"]]

            annotated = draw_detections(frame_to_process, detections)
        else:
            annotated = frame_to_process
            cv2.putText(
                annotated,
                "Ready - Press Start to begin detection",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (100, 100, 100),
                2,
            )

        with latest_frame_lock:
            latest_annotated_frame = annotated

        time.sleep(DETECTION_INTERVAL)

def ensure_background_workers():
    global camera_thread, detection_thread
    if camera_thread is None or not camera_thread.is_alive():
        camera_stop_event.clear()
        camera_thread = threading.Thread(target=camera_capture_worker, daemon=True)
        camera_thread.start()

    if detection_thread is None or not detection_thread.is_alive():
        detection_stop_event.clear()
        detection_thread = threading.Thread(target=detection_worker, daemon=True)
        detection_thread.start()

def generate_frames():
    global latest_annotated_frame
    ensure_background_workers()
    while True:
        with latest_frame_lock:
            frame = latest_annotated_frame.copy() if latest_annotated_frame is not None else None

        if frame is None:
            frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "Opening camera...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (180, 180, 180),
                2,
            )
            
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ret:
            time.sleep(0.02)
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(1 / STREAM_FPS)

# ------------------ Routes ------------------

@app.route('/')
def index():
    template_path = os.path.join(TEMPLATE_DIR, 'index.html')
    if not os.path.exists(template_path):
        return f"Template not found: {template_path}", 500
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_alert_mode', methods=['POST'])
def set_alert_mode():
    global alert_mode
    try:
        data = request.get_json(force=True, silent=True) or {}
        mode = data.get("mode", "english")
    except Exception:
        mode = "english"
    alert_mode = mode
    return jsonify({"mode": alert_mode})

@app.route('/get_alert_mode', methods=['GET'])
def get_alert_mode():
    return jsonify({"mode": alert_mode})

@app.route('/set_environment', methods=['POST'])
def set_environment():
    global environment_mode
    try:
        data = request.get_json(force=True, silent=True) or {}
        mode = data.get("mode", "outdoor")
    except Exception:
        mode = "outdoor"
    environment_mode = mode
    return jsonify({"mode": environment_mode})

@app.route('/get_environment', methods=['GET'])
def get_environment():
    return jsonify({"mode": environment_mode})

@app.route('/set_language', methods=['POST'])
def set_language():
    global current_lang
    try:
        data = request.get_json(force=True, silent=True) or {}
        lang = data.get("language", "en")
    except Exception:
        lang = "en"
    current_lang = lang
    return jsonify({"language": current_lang})

@app.route('/sos', methods=['POST'])
def sos():
    print("[SOS] Emergency assistance requested from client.")
    return jsonify({"status": "received"})

@app.route('/start', methods=['POST'])
def start_detection():
    global is_running
    ensure_background_workers()
    # Clear the alert queues
    while not alert_queue.empty():
        try:
            alert_queue.get_nowait()
        except queue.Empty:
            break
    while not audio_alert_queue.empty():
        try:
            audio_alert_queue.get_nowait()
        except queue.Empty:
            break
    is_running = True
    return jsonify({"status": "started"})

@app.route('/stop', methods=['POST'])
def stop_detection():
    global is_running
    is_running = False
    return jsonify({"status": "stopped"})

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({"running": is_running})

@app.route('/translate', methods=['POST'])
def translate_api():
    try:
        data = request.get_json(force=True, silent=True) or {}
        text = data.get("text", "")
        target_lang = data.get("target_lang", "en")
        if not text:
            return jsonify({"translated": ""})
        translated = translate_text(text, target_lang)
        return jsonify({"translated": translated})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/navigate_to', methods=['POST'])
def navigate_to():
    try:
        data = request.get_json(force=True, silent=True) or {}
        location = data.get("location", "").strip()
        if not location:
            return jsonify({"status": "error", "message": "No location specified."}), 400
        desc = get_location_description(location, current_lang)
        return jsonify({
            "status": "success",
            "message": desc or f"Navigating to {location}."
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_detections', methods=['GET'])
def get_detections():
    faces_detected = [det for det in latest_detections if "familiar face" in det["label"]]
    return jsonify({
        "detections": latest_detections,
        "faces": faces_detected,
        "text": False
    })


@app.route('/get_alert', methods=['GET'])
def get_alert():
    try:
        data = alert_queue.get_nowait()
        return jsonify({
            "alert": data.get("alert"),
            "speech": data.get("speech")
        })
    except queue.Empty:
        return jsonify({"alert": None, "speech": None})

@app.route('/get_config', methods=['GET'])
def get_config():
    """Returns stored API credentials."""
    return jsonify({
        "gemini_key": config_data.get("gemini_key") or "",
        "openrouter_key": config_data.get("openrouter_key") or "",
        "openrouter_model": config_data.get("openrouter_model") or "google/gemini-2.5-flash",
        "system_instruction": config_data.get("system_instruction") or ""
    })

@app.route('/save_config', methods=['POST'])
def save_config():
    """Saves updated credentials from settings modal."""
    global config_data
    try:
        payload = request.get_json(force=True, silent=True) or {}
        config_data["gemini_key"] = payload.get("gemini_key", "").strip()
        config_data["openrouter_key"] = payload.get("openrouter_key", "").strip()
        config_data["openrouter_model"] = payload.get("openrouter_model", "google/gemini-2.5-flash").strip()
        config_data["system_instruction"] = payload.get("system_instruction", "").strip()
        
        with open(CONFIG_FILE, "w") as f:
            json.dump(config_data, f, indent=4)
        
        load_config()
        return jsonify({"status": "success", "message": "Settings saved successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/add_face', methods=['POST'])
def add_face():
    """Saves a face photo from the current camera view under name."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        name = data.get("name", "").strip().lower()
        if not name:
            return jsonify({"status": "error", "error": "No name provided."}), 400
            
        frame = get_latest_camera_frame()
        if frame is None:
            return jsonify({"status": "error", "error": "Camera frame not available."}), 503
            
        # Detect if face exists in current frame before saving
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return jsonify({"status": "error", "error": "No face detected. Please face the camera."}), 400
            
        # Save the full image to known_faces folder
        if not os.path.exists(FACE_DIR):
            os.makedirs(FACE_DIR)
            
        filepath = os.path.join(FACE_DIR, f"{name}.jpg")
        cv2.imwrite(filepath, frame)
        
        # Retrain the model
        train_face_recognizer()
        
        # Success message localized to current language
        english_msg = f"Face registered successfully for {name}."
        speech_msg = english_msg
        if current_lang == 'hi':
            speech_msg = f"{name} का चेहरा सफलतापूर्वक पंजीकृत कर लिया गया है।"
        elif current_lang == 'te':
            speech_msg = f"{name} ముఖం విజయవంతంగా నమోదు చేయబడింది."
            
        return jsonify({"status": "success", "message": english_msg, "speech": speech_msg})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/delete_face', methods=['POST'])
def delete_face():
    """Removes a registered face details."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        name = data.get("name", "").strip().lower()
        if not name:
            return jsonify({"status": "error", "error": "No name specified."}), 400
            
        filepath = os.path.join(FACE_DIR, f"{name}.jpg")
        if os.path.exists(filepath):
            os.remove(filepath)
            train_face_recognizer()
            
            english_msg = f"Face details for {name} deleted."
            speech_msg = english_msg
            if current_lang == 'hi':
                speech_msg = f"{name} का विवरण हटा दिया गया है।"
            elif current_lang == 'te':
                speech_msg = f"{name} వివరాలు తొలగించబడ్డాయి."
                
            return jsonify({"status": "success", "message": english_msg, "speech": speech_msg})
        else:
            return jsonify({"status": "error", "error": "Face not found."}), 404
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/identify_face', methods=['POST'])
def identify_face():
    """Identifies the person in front of the camera using Gemini VLM or offline templates."""
    frame = get_latest_camera_frame()
    if frame is None:
        return jsonify({"status": "error", "error": "Camera frame not available."}), 503

    # Check if there is any person in the frame (using YOLO model detection)
    person_detected = False
    detections = process_frame(frame)
    for det in detections:
        if "person" in det["label"] or "familiar face" in det["label"]:
            person_detected = True
            break
            
    # Check if Gemini VLM is configured and we have registered faces to compare
    gemini_key = config_data.get("gemini_key") or GEMINI_API_KEY
    openrouter_key = config_data.get("openrouter_key")
    openrouter_model = config_data.get("openrouter_model") or "google/gemini-2.5-flash"
    
    # Get all registered faces
    registered_files = []
    if os.path.exists(FACE_DIR):
        for filename in os.listdir(FACE_DIR):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                registered_files.append(filename)

    vlm_res = None

    # If VLM is available and we have registered people
    if (gemini_key or openrouter_key) and registered_files:
        try:
            parts = []
            
            # 1. Query Frame
            query_base64 = encode_frame_for_vision(frame)
            if not query_base64:
                return jsonify({"status": "error", "error": "Could not encode query frame."}), 500
                
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": query_base64
                }
            })
            
            # 2. Prompt describing the task and referencing the files
            prompt_text = (
                "You are a face matching assistant. "
                "The first image is the query image from the user's camera. "
                "The subsequent images are the registered face photos of known individuals. "
                "Each registered face photo has a name labeled below. "
                "Task: Compare the face(s) in the first (query) image against each of the registered faces. "
                "Check if the person in the query image is one of these registered people. "
                "You must respond ONLY with a JSON object in this format:\n"
                "{\n"
                "  \"registered\": true or false,\n"
                "  \"name\": \"Name of the person if registered, or empty string if not\",\n"
                "  \"description\": \"A brief description of the person in the query image (gender, age group, hair, clothes, expression) if not registered or if no matching name\"\n"
                "}\n"
                "Do not include any markdown styling, quotes, or code block formatting (like ```json). Just return the raw JSON object."
            )
            
            # 3. Add registered faces to the parts list and name them in prompt
            prompt_text += "\nRegistered faces list:"
            for filename in registered_files:
                name = os.path.splitext(filename)[0]
                filepath = os.path.join(FACE_DIR, filename)
                reg_img = cv2.imread(filepath)
                if reg_img is not None:
                    reg_base64 = encode_frame_for_vision(reg_img)
                    if reg_base64:
                        parts.append({
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": reg_base64
                            }
                        })
                        prompt_text += f"\n- Image index {len(parts) - 1} is registered face of: '{name}'"
            
            parts.insert(0, {"text": prompt_text})
            
            # Call Gemini/OpenRouter API
            if openrouter_key:
                messages_content = [{"type": "text", "text": prompt_text}]
                messages_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{query_base64}"}
                })
                
                for filename in registered_files:
                    name = os.path.splitext(filename)[0]
                    filepath = os.path.join(FACE_DIR, filename)
                    reg_img = cv2.imread(filepath)
                    if reg_img is not None:
                        reg_base64 = encode_frame_for_vision(reg_img)
                        if reg_base64:
                            messages_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{reg_base64}"}
                            })
                            
                url = "https://openrouter.ai/api/v1/chat/completions"
                payload = {
                    "model": openrouter_model,
                    "messages": [{"role": "user", "content": messages_content}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {openrouter_key}",
                    },
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=25) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    text = result["choices"][0]["message"]["content"].strip()
                    vlm_res = json.loads(text)
                    
            elif gemini_key:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
                payload = {
                    "contents": [{"parts": parts}],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "temperature": 0.2
                    }
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": gemini_key,
                    },
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=25) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                    vlm_res = json.loads(text)
                    
        except Exception as e:
            print(f"[Identify Face VLM Error] {e}")
            # Fall back to offline method

    # Process VLM results if we got them successfully
    if vlm_res:
        registered = vlm_res.get("registered", False)
        name = vlm_res.get("name", "")
        desc = vlm_res.get("description", "")
        
        if registered:
            reply = f"{name} is in front of you."
            if current_lang == 'hi':
                speech = f"आपके सामने {name} हैं।"
            elif current_lang == 'te':
                speech = f"మీ ముందు {name} ఉన్నారు."
            else:
                speech = reply
        elif desc:
            reply = f"The person in front of you is not registered. They look like {desc}."
            if current_lang == 'hi':
                speech = f"सामने खड़ा व्यक्ति पंजीकृत नहीं है। वे {translate_text(desc, 'hi')} लग रहे हैं।"
            elif current_lang == 'te':
                speech = f"ముందు ఉన్న వ్యక్తి నమోదు చేయబడలేదు. వారు {translate_text(desc, 'te')} గా ఉన్నారు."
            else:
                speech = reply
        else:
            reply = "I don't see anyone in front of you."
            if current_lang == 'hi':
                speech = "मुझे आपके सामने कोई व्यक्ति नहीं दिख रहा है।"
            elif current_lang == 'te':
                speech = "మీ ముందు ఎవరూ కనిపించడం లేదు."
            else:
                speech = reply
                
        return jsonify({
            "registered": registered,
            "name": name,
            "description": desc,
            "reply": reply,
            "speech": speech
        })

    # Offline Fallback (Local OpenCV Template Matching)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0 and known_face_templates:
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        face_crop = gray[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (100, 100))
        face_crop = cv2.equalizeHist(face_crop)
        
        best_name = None
        best_score = -1.0
        for name, template in known_face_templates.items():
            res = cv2.matchTemplate(face_crop, template, cv2.TM_CCOEFF_NORMED)
            score = res[0][0]
            if score > best_score:
                best_score = score
                best_name = name
                
        if best_score > 0.65:
            reply = f"{best_name} is in front of you."
            if current_lang == 'hi':
                speech = f"आपके सामने {best_name} हैं।"
            elif current_lang == 'te':
                speech = f"మీ ముందు {best_name} ఉన్నారు."
            else:
                speech = reply
            return jsonify({
                "registered": True,
                "name": best_name,
                "description": "Familiar face match",
                "reply": reply,
                "speech": speech
            })
            
    if len(faces) > 0:
        reply = "The person in front of you is not registered."
        if current_lang == 'hi':
            speech = "सामने खड़ा व्यक्ति पंजीकृत नहीं है।"
        elif current_lang == 'te':
            speech = "ముందు ఉన్న వ్యక్తి నమోదు చేయబడలేదు."
        else:
            speech = reply
        return jsonify({
            "registered": False,
            "name": "",
            "description": "an unrecognized person",
            "reply": reply,
            "speech": speech
        })
    else:
        reply = "I don't see anyone in front of you."
        if current_lang == 'hi':
            speech = "मुझे आपके सामने कोई व्यक्ति नहीं दिख रहा है।"
        elif current_lang == 'te':
            speech = "మీ ముందు ఎవరూ కనిపించడం లేదు."
        else:
            speech = reply
        return jsonify({
            "registered": False,
            "name": "",
            "description": "",
            "reply": reply,
            "speech": speech
        })

@app.route('/list_faces', methods=['GET'])
def list_faces():
    """Lists all registered face names."""
    try:
        faces = []
        if os.path.exists(FACE_DIR):
            for filename in os.listdir(FACE_DIR):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    name = os.path.splitext(filename)[0]
                    faces.append(name)
        return jsonify({"faces": faces})
    except Exception as e:
        return jsonify({"faces": [], "error": str(e)}), 500

@app.route('/scene_description', methods=['GET'])
def scene_description():
    """Generate or retrieve scene description using VLM first, falling back to YOLO descriptions."""
    frame = get_latest_camera_frame()
    
    # Try calling the VLM if configured
    if frame is not None and (config_data.get("gemini_key") or config_data.get("openrouter_key") or GEMINI_API_KEY):
        desc, err = call_vision_llm(frame, "")
        if not err and desc:
            return jsonify({"description": desc})

    # Rule-based fallback if VLM is unavailable
    if not latest_detections:
        desc = "I don't see anything around you right now."
        if current_lang == 'hi':
            desc = "मुझे अभी आपके आस-पास कुछ भी नहीं दिख रहा है।"
        elif current_lang == 'te':
            desc = "నాకు మీ చుట్టుపక్కల ఏమీ కనిపించడం లేదు."
        return jsonify({"description": desc})
    
    counts = {}
    for det in latest_detections:
        base_label = det["label"].replace("red ", "").replace("green ", "").replace(" with zebra crossing", "")
        if "familiar face" in base_label:
            name = base_label.replace("familiar face ", "")
            base_label = f"familiar face {name}"
        counts[base_label] = counts.get(base_label, 0) + 1
    
    parts = []
    if "bus" in counts or "bench" in counts or "stop sign" in counts:
        parts.append("You appear to be near a bus stop or crosswalk.")
    elif "chair" in counts or "dining table" in counts or "tv" in counts:
        parts.append("You appear to be indoors.")
        
    items = []
    for label, count in counts.items():
        if count == 1:
            items.append(f"1 {label}")
        else:
            items.append(f"{count} {label}s")
                
    if items:
        if len(items) == 1:
            parts.append(f"I see {items[0]} in front of you.")
        else:
            parts.append(f"I see {', '.join(items[:-1])}, and {items[-1]} in front of you.")
            
    description = " ".join(parts)
    if description and current_lang in ['hi', 'te']:
        description = translate_text(description, current_lang)
    return jsonify({"description": description})

@app.route('/read_text', methods=['GET'])
def read_text():
    """Extracts text using local EasyOCR, with VLM fallback and translation."""
    frame_to_process = get_latest_camera_frame()
    if frame_to_process is None:
        return jsonify({"text": "", "speech": ""})
        
    final_text = ""
    try:
        results = reader.readtext(frame_to_process)
        extracted_texts = [text for (bbox, text, prob) in results if prob > 0.3]
        final_text = " ".join(extracted_texts).strip()
    except Exception as e:
        print(f"OCR Error: {e}")
        # Try Gemini VLM as fallback
        gemini_key = config_data.get("gemini_key") or GEMINI_API_KEY
        openrouter_key = config_data.get("openrouter_key")
        if gemini_key or openrouter_key:
            try:
                desc, err = call_vision_llm(
                    frame_to_process, 
                    "Perform OCR on this image. Extract all text present in the image and return it in English. Return only the extracted text, do not add any additional explanation or wrapper."
                )
                if not err and desc:
                    final_text = desc
            except Exception as ex:
                print(f"VLM OCR fallback failed: {ex}")
                
    speech_text = final_text
    if final_text and current_lang in ['hi', 'te']:
        speech_text = translate_text(final_text, current_lang)
        
    return jsonify({"text": final_text, "speech": speech_text})

@app.route('/mickey_vision', methods=['POST'])
def mickey_vision():
    """Smart Look endpoint utilizing VLM."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        prompt = str(data.get("prompt", "")).strip()
    except Exception:
        prompt = ""

    frame = get_latest_camera_frame()
    if frame is None:
        return jsonify({
            "description": "",
            "error": "Camera is not ready yet. Please wait.",
        }), 503

    description, error = call_vision_llm(frame, prompt)
    if error:
        return jsonify({"description": "", "error": error}), 503

    return jsonify({"description": description, "error": None})

@app.route('/mickey', methods=['POST'])
def mickey_assistant():
    """Handles assistant brain commands in conversational style."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        message = str(data.get("message", "")).strip()
        lat = data.get("lat")
        lng = data.get("lng")
        active_dest = data.get("activeDestination", "")
    except Exception:
        message = ""
        lat = None
        lng = None
        active_dest = ""

    if not message:
        english_reply = RESPONSES["en"]["help"]
        speech_text = translate_text(english_reply, current_lang) if current_lang != "en" else english_reply
        return jsonify({
            "reply": english_reply,
            "speech": speech_text,
            "action": None,
        })

    # AI assistant path
    if config_data.get("gemini_key") or config_data.get("openrouter_key") or GEMINI_API_KEY:
        ai_reply, err = call_assistant_ai(message, "en", lat, lng, active_dest)
        if not err and ai_reply:
            if ai_reply.get("action") == "navigate":
                loc_desc = ai_reply.get("location_description")
                if loc_desc:
                    ai_reply["reply"] = f"{ai_reply.get('reply', '')} {loc_desc}".strip()
            
            english_reply = ai_reply.get("reply", "")
            if current_lang != "en":
                ai_reply["speech"] = translate_text(english_reply, current_lang)
            else:
                ai_reply["speech"] = english_reply
            return jsonify(ai_reply)

    # Local rule-based fallback
    rule_reply = rule_based_mickey(message, "en")
    if rule_reply.get("action") == "navigate":
        dest = rule_reply.get("destination")
        if dest:
            desc = get_location_description(dest, "en")
            rule_reply["location_description"] = desc
            rule_reply["reply"] = f"{rule_reply.get('reply', '')} {desc}".strip()
            
    english_reply = rule_reply.get("reply", "")
    if current_lang != "en":
        rule_reply["speech"] = translate_text(english_reply, current_lang)
    else:
        rule_reply["speech"] = english_reply
    return jsonify(rule_reply)

# ------------------ Cleanup on exit ------------------
def cleanup():
    global camera
    camera_stop_event.set()
    detection_stop_event.set()
    print("[CLEANUP] Shutting down — releasing camera...")
    try:
        if camera is not None and camera.isOpened():
            camera.release()
            print("[CLEANUP] Camera released.")
    except Exception as e:
        print(f"[CLEANUP] Camera release error: {e}")
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    print("[CLEANUP] Done.")

atexit.register(cleanup)

def signal_handler(sig, frame):
    cleanup()
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=5000)
    finally:
        cleanup()
