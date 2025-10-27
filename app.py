import os
# Suppress TensorFlow informational messages and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import string
import time
from PIL import Image
from deep_translator import GoogleTranslator
from gtts import gTTS
import speech_recognition as sr
import pycountry
import numpy as np
import cv2
import torch
import torch.nn as nn
import mediapipe as mp

# --- App Configuration ---
st.set_page_config(page_title="ISL Translator", page_icon="🤟", layout="wide")

# --- Constants & Model Loading ---
ISL_GIFS_PATH = "ISL_Gifs"
LETTERS_PATH = "letters"
OUTPUT_AUDIO_FILE = "output.mp3"
MODEL_PATH = "ISL.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPECTED_FEATURE_LENGTH = 128
NUM_CLASSES = 36 
FEATURES_PER_HAND = (21 * 3) + 1 # 64

# --- PyTorch Model Definition (from isl_predict.py) ---
class HandLandmarkCNN(nn.Module):
    def __init__(self, input_size=128, num_classes=NUM_CLASSES):
        super(HandLandmarkCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# --- Updated Model Loading ---
@st.cache_resource
def load_isl_model():
    """Loads the trained PyTorch model for ISL recognition."""
    try:
        model = HandLandmarkCNN(input_size=EXPECTED_FEATURE_LENGTH, num_classes=NUM_CLASSES).to(DEVICE)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading the PyTorch model: {e}")
        st.error(f"Please ensure '{MODEL_PATH}' is present and the model definition matches.")
        return None

# --- Updated Label Loading (No Hardcoding) ---
@st.cache_data
def get_class_labels():
    """
    Loads class labels by reading the subdirectory names from the 'data' folder.
    """
    try:
        data_dir = "data" 
        if not os.path.isdir(data_dir):
            st.error(f"Error: 'data' directory not found. This folder is required to get the class names.")
            return {}
        
        labels_list = sorted(os.listdir(data_dir))
        
        if not labels_list:
            st.error("Error: 'data' directory is empty. No class labels found.")
            return {}
        
        if len(labels_list) != NUM_CLASSES:
            st.warning(f"Warning: Model expects {NUM_CLASSES} classes, but found {len(labels_list)} folders in 'data'. Please update NUM_CLASSES if this is intended.")
        
        index_to_label = {i: label for i, label in enumerate(labels_list)}
        
        st.success(f"Loaded {len(labels_list)} labels (e.g., '{labels_list[0]}', '{labels_list[1]}') from 'data' directory.")
        return index_to_label

    except Exception as e:
        st.error(f"Error reading class labels from 'data' directory: {e}")
        return {}


model = load_isl_model()
class_labels = get_class_labels() # This is now a dictionary like {0: 'a', 1: 'b', ...}

# --- MediaPipe Setup (Cached) ---
@st.cache_resource
def load_mediapipe():
    """Initializes and returns MediaPipe Hands components.
    Uses THE EXACT settings from isl_predict.py"""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands_detector = mp_hands.Hands(
        static_image_mode=False, # From isl_predict.py
        max_num_hands=2, 
        min_detection_confidence=0.7 # From isl_predict.py
    )
    return mp_hands, mp_drawing, hands_detector

# --- Helper Functions (Unchanged) ---
@st.cache_data
def get_isl_gif_map():
    try:
        gif_map = {}
        for filename in os.listdir(ISL_GIFS_PATH):
            if filename.endswith('.gif'):
                phrase = os.path.splitext(filename)[0]
                gif_map[phrase.lower()] = filename
        return gif_map
    except FileNotFoundError:
        st.error(f"Error: The directory '{ISL_GIFS_PATH}' was not found.")
        return {}

def get_language_name_from_code(code):
    try:
        base_code = code.split('-')[0].lower()
        lang = pycountry.languages.get(alpha_2=base_code) or pycountry.languages.get(alpha_3=base_code)
        return lang.name if lang else None
    except Exception:
        return None

@st.cache_data
def get_supported_languages_map():
    try:
        supported_dict = GoogleTranslator().get_supported_languages(as_dict=True)
        name_to_code = {}
        for code, name_from_lib in supported_dict.items():
            full_name = get_language_name_from_code(code)
            if full_name:
                name_to_code[full_name] = code
            elif len(name_from_lib) > 3:
                 name_to_code[name_from_lib.capitalize()] = code
            else:
                 name_to_code[code.upper()] = code
        return sorted(name_to_code.keys()), name_to_code
    except Exception as e:
        st.error(f"Failed to load supported languages: {e}")
        return ["English"], {"English": "en"}

def display_sign_language_and_audio(text_to_display, output_container):
    with output_container:
        try:
            tts = gTTS(text_to_display, lang="en", tld="co.in")
            tts.save(OUTPUT_AUDIO_FILE)
            st.audio(OUTPUT_AUDIO_FILE, format='audio/mp3')
        except Exception as e:
            st.error(f"Could not generate audio: {e}")

        cleaned_text = text_to_display.lower().translate(str.maketrans('', '', string.punctuation))
        isl_gif_map = get_isl_gif_map()
        if cleaned_text in isl_gif_map:
            gif_path = os.path.join(ISL_GIFS_PATH, isl_gif_map[cleaned_text])
            st.image(gif_path, caption=f"Showing: {cleaned_text}")
        else:
            st.write("Displaying individual letters:")
            image_placeholder = st.empty()
            for char in cleaned_text:
                if 'a' <= char <= 'z':
                    letter_path = os.path.join(LETTERS_PATH, f"{char}.jpg")
                    if os.path.exists(letter_path):
                        image = Image.open(letter_path)
                        image_placeholder.image(image, caption=f"Letter: {char}", width=300)
                        time.sleep(0.8)
                elif char == ' ':
                    image_placeholder.empty()
                    time.sleep(0.5)

# --- UI Tabs ---
st.title("Indian Sign Language (ISL) Communication Hub")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Home", "🌍 Text Translator", "📝 Text to Sign", "🎤 Voice to Sign", "📹 Sign to Text"])

# --- Home Tab (Unchanged) ---
with tab1:
    st.header("Welcome to the ISL Communication Hub!")
    # ... (rest of the tab is unchanged) ...
    st.markdown("Your one-stop application for breaking down communication barriers.")
    home_image_path = "signlang.png"
    if os.path.exists(home_image_path):
        st.image(home_image_path, caption="Empowering Communication Through Technology", use_container_width=True)
    else:
        st.warning(f"**Warning:** Image '{home_image_path}' not found.")
    st.subheader("Our Features:")
    st.markdown("""
        - **🌍 Text Translator:** Translate text between various languages, using either text or voice input.
        - **📝 Text to Sign:** Type any text to see the corresponding Indian Sign Language gestures.
        - **🎤 Voice to Sign:** Speak into your microphone to translate your voice into ISL gestures.
        - **📹 Sign to Text:** Use your webcam to translate sign language into text in real-time.
    """)
    st.info("Select a tab above to get started!")

# --- Text Translator Tab (Unchanged) ---
with tab2:
    st.header("General Purpose Text Translator")
    # ... (rest of the tab is unchanged, includes the syntax fix) ...
    st.markdown("Translate text between any of the supported languages, by typing or by voice.")
    lang_names, name_to_code_map = get_supported_languages_map()
    try:
        default_source_index = lang_names.index('English')
    except ValueError:
        default_source_index = 0
    try:
        default_target_index = lang_names.index('Hindi')
    except ValueError:
        default_target_index = 1 if len(lang_names) > 1 else 0
    col1, col2 = st.columns(2)
    with col1:
        selected_source_lang_name = st.selectbox("From Language:", lang_names, index=default_source_index, key="source_lang")
    with col2:
        selected_target_lang_name = st.selectbox("To Language:", lang_names, index=default_target_index, key="target_lang")
    
    if "translator_input_text" not in st.session_state:
        st.session_state.translator_input_text = ""
        
    st.session_state.translator_input_text = st.text_area("Enter text to translate (or use the microphone):", value=st.session_state.translator_input_text, height=150, key="text_translator_area")
    button_col1, button_col2 = st.columns(2)
    with button_col1:
        if st.button("Use Microphone 🎤", key="listen_general", use_container_width=True):
            recognizer = sr.Recognizer()
            with st.spinner("Listening..."):
                try:
                    with sr.Microphone() as source:
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    recognized_text = recognizer.recognize_google(audio, language="en-IN")
                    st.session_state.translator_input_text = recognized_text
                   
                except Exception as e:
                    st.error(f"An error occurred during speech recognition: {e}")
    with button_col2:
        if st.button("Translate Text", key="translate_general", type="primary", use_container_width=True):
            if st.session_state.translator_input_text:
                try:
                    source_code = name_to_code_map[st.session_state.source_lang]
                    target_code = name_to_code_map[st.session_state.target_lang]
                    translated = GoogleTranslator(source=source_code, target=target_code).translate(st.session_state.translator_input_text)
                    st.success("Translated Text:")
                    st.text_area("Result", value=translated, height=150)
                except Exception as e:
                    st.error(f"An error occurred during translation: {e}")
            else:
                st.warning("Please enter or say something to translate.")

# --- Text to Sign Tab (Unchanged) ---
with tab3:
    st.header("Translate Text into Sign Language")
    # ... (rest of the tab is unchanged) ...
    st.markdown("Enter English text to see the corresponding ISL gestures and hear the audio.")
    col1, col2 = st.columns([1, 1])
    with col1:
        input_text_sign = st.text_area("Enter your text here:", height=150)
        if st.button("Translate to Sign", key="text_translate"):
            if input_text_sign:
                st.session_state.text_to_display = input_text_sign
            else:
                st.warning("Please enter text to translate.")
    with col2:
        st.subheader("Sign Language Output")
        text_output_container = st.container()
        if 'text_to_display' in st.session_state and st.session_state.text_to_display:
            display_sign_language_and_audio(st.session_state.text_to_display, text_output_container)
            del st.session_state.text_to_display
        else:
            text_output_container.info("The ISL gestures and audio will appear here.")

# --- Voice to Sign Tab (Unchanged) ---
with tab4:
    st.header("Translate Your Voice into Sign Language")
    # ... (rest of the tab is unchanged) ...
    st.markdown("Click the button and speak in English to see the ISL translation.")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Start Listening 🎤", key="start_listening"):
            recognizer = sr.Recognizer()
            with st.spinner("Listening..."):
                try:
                    with sr.Microphone() as source:
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    recognized_text = recognizer.recognize_google(audio, language="en-IN").lower()
                    st.info(f"You said: \"{recognized_text}\"")
                    st.session_state.voice_to_display = recognized_text
                except Exception as e:
                    st.error(f"An error occurred during speech recognition: {e}")
    with col2:
        st.subheader("Sign Language Output")
        voice_output_container = st.container()
        if 'voice_to_display' in st.session_state and st.session_state.voice_to_display:
            display_sign_language_and_audio(st.session_state.voice_to_display, voice_output_container)
            del st.session_state.voice_to_display
        else:
            voice_output_container.info("The ISL gestures will appear here after you speak.")

# =================================================================
# === SIGN TO TEXT TAB (UPDATED WITH "NO HAND" FIX) ===
# =================================================================
with tab5:
    st.header("Translate Sign Language into Text")
    st.markdown("Click 'Start Webcam' to begin. Hold a sign steady to add it. Use the buttons to edit.")
    
    if model is None:
        st.error("Model could not be loaded. This feature is unavailable.")
    elif not class_labels:
         st.error(f"Class labels could not be generated. Please ensure the 'data' folder is present and not empty.")
    else:
        mp_hands, mp_drawing, hands_detector = load_mediapipe()
        
        STABILITY_THRESHOLD = 5 

        if "run" not in st.session_state:
            st.session_state.run = False
        if "composed_text" not in st.session_state:
            st.session_state.composed_text = ""
        if "last_committed_prediction" not in st.session_state:
            st.session_state.last_committed_prediction = ""
        if "current_stable_prediction" not in st.session_state:
            st.session_state.current_stable_prediction = ""
        if "prediction_stability_counter" not in st.session_state:
            st.session_state.prediction_stability_counter = 0

        col1, col2 = st.columns([2, 1])
        
        with col1:
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                if st.button("Start Webcam", use_container_width=True):
                    st.session_state.run = True
            with b_col2:
                if st.button("Stop Webcam", use_container_width=True):
                    st.session_state.run = False
                    
            frame_placeholder = st.empty()
            
            cap = None 
            if st.session_state.run:
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("Error: Could not open camera. Please check permissions.")
                    st.session_state.run = False
                
                num_hands = 2 
                expected_len = EXPECTED_FEATURE_LENGTH
                
                # This will show the *live* (unstable) prediction
                live_prediction_text = "..." 

                while st.session_state.run:
                    ret, frame = cap.read()
                    if not ret:
                        st.write("Failed to grab frame. Stopping...")
                        st.session_state.run = False
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = hands_detector.process(frame_rgb)
                    feature = []
                    
                    # --- THIS IS THE KEY CHANGE (Part 1) ---
                    hands_detected = False

                    if result.multi_hand_landmarks:
                        hands_detected = True # <-- Set flag to True
                        for hand_landmarks in result.multi_hand_landmarks[:num_hands]: 
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            
                            x_coords = [lm.x for lm in hand_landmarks.landmark]
                            y_coords = [lm.y for lm in hand_landmarks.landmark]
                            x_min, x_max = min(x_coords), max(x_coords)
                            y_min, y_max = min(y_coords), max(y_coords)

                            for lm in hand_landmarks.landmark:
                                feature.extend([
                                    (lm.x - x_min) / (x_max - x_min + 1e-6),
                                    (lm.y - y_min) / (y_max - y_min + 1e-6),
                                    lm.z
                                ])

                            thumb_tip = hand_landmarks.landmark[4]
                            index_tip = hand_landmarks.landmark[8]
                            distance = ((thumb_tip.x - index_tip.x)**2 +
                                        (thumb_tip.y - index_tip.y)**2 +
                                        (thumb_tip.z - index_tip.z)**2) ** 0.5
                            feature.append(distance)

                    while len(feature) < expected_len:
                        feature.extend([0] * FEATURES_PER_HAND)

                    # --- THIS IS THE KEY CHANGE (Part 2) ---
                    if hands_detected:
                        # Only run prediction and stability if hands were found
                        if len(feature) == expected_len:
                            feature_tensor = torch.tensor([feature], dtype=torch.float32).to(DEVICE)
                            with torch.no_grad():
                                output = model(feature_tensor)
                                _, predicted = torch.max(output, 1)
                                
                                live_prediction_text = class_labels.get(predicted.item(), "???") 
                                
                                # --- Stability Logic ---
                                if live_prediction_text == st.session_state.current_stable_prediction:
                                    st.session_state.prediction_stability_counter += 1
                                else:
                                    st.session_state.current_stable_prediction = live_prediction_text
                                    st.session_state.prediction_stability_counter = 0

                                # --- Commit Logic ---
                                if st.session_state.prediction_stability_counter >= STABILITY_THRESHOLD:
                                    if live_prediction_text != st.session_state.last_committed_prediction and live_prediction_text != "???":
                                        st.session_state.composed_text += live_prediction_text
                                        st.session_state.last_committed_prediction = live_prediction_text
                                        st.session_state.prediction_stability_counter = 0 
                    else:
                        # --- No Hands Detected ---
                        live_prediction_text = "..."
                        
                        # --- RESET STABILITY ---
                        # This clears the counters and "unlocks" the last letter
                        st.session_state.current_stable_prediction = "..."
                        st.session_state.prediction_stability_counter = 0
                        
                        # If the last thing we added was a letter (not space),
                        # reset it. This lets us add the same letter again.
                        if st.session_state.last_committed_prediction not in ["...", " ", ""]:
                            st.session_state.last_committed_prediction = "..."
                            
                    # Show the *live* prediction on the frame
                    cv2.putText(frame, f'{live_prediction_text}', (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    
                    frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_display, channels="RGB")
                
                if cap:
                    cap.release()
                cv2.destroyAllWindows()
            
            else:
                frame_placeholder.info("Webcam is off. Click 'Start Webcam' to begin.")
        
        # --- Composed Text Column (Unchanged) ---
        with col2:
            st.subheader("Composed Text")
            st.text_area("Sentence", value=st.session_state.composed_text, height=300, key="composed_sentence")
            
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                if st.button("Add Space", use_container_width=True):
                    st.session_state.composed_text += " "
                    st.session_state.last_committed_prediction = " " 
                    
            with b_col2:
                if st.button("Backspace", use_container_width=True):
                    if len(st.session_state.composed_text) > 0:
                        st.session_state.composed_text = st.session_state.composed_text[:-1]
                        st.session_state.last_committed_prediction = "" 
                        
            
            if st.button("Clear Text", use_container_width=True):
                st.session_state.composed_text = ""
                st.session_state.last_committed_prediction = ""
                st.session_state.current_stable_prediction = ""
                st.session_state.prediction_stability_counter = 0
                