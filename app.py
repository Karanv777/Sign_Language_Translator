import os
# Suppress TensorFlow informational messages and warnings (if relevant, though not used here)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import av # Needed by streamlit-webrtc
import queue # For passing results between threads
import pydub # For audio conversion
from io import BytesIO # To handle audio in memory
import threading

# --- App Configuration ---
st.set_page_config(page_title="Voyage Voice", page_icon="🤟", layout="wide")

# --- Constants ---
ISL_GIFS_PATH = "ISL_Gifs"
LETTERS_PATH = "letters"
OUTPUT_AUDIO_FILE = "output.mp3"
MODEL_PATH = "asl1.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPECTED_FEATURE_LENGTH = 128
NUM_CLASSES = 36 # Should match the number of folders in 'data'
FEATURES_PER_HAND = (21 * 3) + 1 # 64
STABILITY_THRESHOLD = 8 # Number of frames for Sign-to-Text stability

# --- WebRTC Configuration ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun.cloudflare.com:3478"]}, # Adding Cloudflare
        # Add more public STUN servers if needed
    ]}
)

# --- PyTorch Model Definition ---
class HandLandmarkCNN(nn.Module):
    def __init__(self, input_size=EXPECTED_FEATURE_LENGTH, num_classes=NUM_CLASSES):
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

# --- Cached Resource Loaders ---
@st.cache_resource
def load_isl_model():
    """Loads the trained PyTorch model."""
    try:
        model = HandLandmarkCNN(input_size=EXPECTED_FEATURE_LENGTH, num_classes=NUM_CLASSES).to(DEVICE)
        # Ensure the model file exists
        if not os.path.exists(MODEL_PATH):
            st.error(f"Error: Model file '{MODEL_PATH}' not found. Please ensure it's in the root directory.")
            return None
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        # st.success(f"Sign language model loaded successfully on {DEVICE}.")
        return model
    except Exception as e:
        st.error(f"Error loading the PyTorch model: {e}")
        st.error(f"Please ensure '{MODEL_PATH}' is present and the model definition matches the saved state.")
        return None

@st.cache_data
def get_class_labels():
    """Loads class labels from the 'data' folder subdirectories."""
    try:
        data_dir = "data"
        if not os.path.isdir(data_dir):
            st.error(f"Error: 'data' directory not found. Needed for sign language labels.")
            return {}

        labels_list = sorted(os.listdir(data_dir))
        if not labels_list:
            st.error("Error: 'data' directory is empty. No class labels found.")
            return {}

        # Check against NUM_CLASSES constant
        if len(labels_list) != NUM_CLASSES:
            st.warning(f"Warning: Found {len(labels_list)} folders in 'data', but NUM_CLASSES is set to {NUM_CLASSES}. Predictions might be incorrect if this doesn't match the model output layer size.")

        index_to_label = {i: label for i, label in enumerate(labels_list)}
        # st.info(f"Loaded {len(labels_list)} sign labels (e.g., '{labels_list[0]}') from '{data_dir}'.")
        return index_to_label

    except Exception as e:
        st.error(f"Error reading class labels from 'data' directory: {e}")
        return {}

@st.cache_resource
def load_mediapipe():
    """Initializes MediaPipe Hands."""
    try:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        return mp_hands, mp_drawing, hands_detector
    except Exception as e:
        st.error(f"Error initializing MediaPipe Hands: {e}")
        return None, None, None

# Load resources on app start
model = load_isl_model()
class_labels = get_class_labels()
mp_hands, mp_drawing, hands_detector = load_mediapipe()

# --- Helper Functions ---
@st.cache_data
def get_isl_gif_map():
    """Creates a mapping from lowercase phrases to GIF filenames."""
    try:
        if not os.path.isdir(ISL_GIFS_PATH):
            st.warning(f"Warning: GIF directory '{ISL_GIFS_PATH}' not found. Text/Voice to Sign GIF display will be limited.")
            return {}
        gif_map = {}
        for filename in os.listdir(ISL_GIFS_PATH):
            if filename.lower().endswith('.gif'):
                phrase = os.path.splitext(filename)[0]
                gif_map[phrase.lower()] = filename
        return gif_map
    except Exception as e:
        st.error(f"Error loading GIFs from '{ISL_GIFS_PATH}': {e}")
        return {}

def get_language_name_from_code(code):
    """Gets full language name from ISO code (e.g., 'en' -> 'English')."""
    try:
        base_code = code.split('-')[0].lower()
        lang = pycountry.languages.get(alpha_2=base_code) or pycountry.languages.get(alpha_3=base_code)
        return lang.name if lang else None
    except Exception:
        return None # Handle cases where pycountry might fail

@st.cache_data
def get_supported_languages_map():
    """Gets supported languages for Google Translator."""
    try:
        supported_dict = GoogleTranslator().get_supported_languages(as_dict=True)
        name_to_code = {}
        # Try to get proper names, fallback to library names or codes
        for code, name_from_lib in supported_dict.items():
            full_name = get_language_name_from_code(code)
            display_name = full_name if full_name else (name_from_lib.capitalize() if len(name_from_lib) > 3 else code.upper())
            name_to_code[display_name] = code
        return sorted(name_to_code.keys()), name_to_code
    except Exception as e:
        st.error(f"Failed to load supported languages for translation: {e}")
        return ["English"], {"English": "en"} # Fallback

def display_sign_language_and_audio(text_to_display, output_container):
    """Displays GIFs/letters and plays audio for the given text."""
    with output_container:
        # Audio Generation
        try:
            tts = gTTS(text_to_display, lang="en", tld="co.in") # Use Indian English accent
            tts.save(OUTPUT_AUDIO_FILE)
            st.audio(OUTPUT_AUDIO_FILE, format='audio/mp3')
            # Clean up audio file after playing (optional)
            # if os.path.exists(OUTPUT_AUDIO_FILE):
            #     os.remove(OUTPUT_AUDIO_FILE)
        except Exception as e:
            st.error(f"Could not generate or play audio: {e}")

        # Sign Language Display
        cleaned_text = text_to_display.lower().translate(str.maketrans('', '', string.punctuation))
        isl_gif_map = get_isl_gif_map()

        if cleaned_text in isl_gif_map:
            # Display full phrase GIF if available
            gif_path = os.path.join(ISL_GIFS_PATH, isl_gif_map[cleaned_text])
            if os.path.exists(gif_path):
                st.image(gif_path, caption=f"Showing: {cleaned_text.capitalize()}")
            else:
                st.warning(f"GIF '{isl_gif_map[cleaned_text]}' not found.")
        elif cleaned_text:
            # Display individual letters if phrase GIF not found
            st.write("Displaying individual letters:")
            if not os.path.isdir(LETTERS_PATH):
                st.warning(f"Letter images directory '{LETTERS_PATH}' not found.")
                return

            image_placeholder = st.empty()
            for char in cleaned_text:
                if 'a' <= char <= 'z':
                    letter_path = os.path.join(LETTERS_PATH, f"{char}.jpeg")
                    if os.path.exists(letter_path):
                        try:
                            image = Image.open(letter_path)
                            image_placeholder.image(image, caption=f"Letter: {char.upper()}", width=200) # Smaller width
                            time.sleep(1.0) # Slightly faster transition
                        except Exception as e:
                            st.error(f"Error loading image {letter_path}: {e}")
                            image_placeholder.empty()
                            time.sleep(0.5)
                    else:
                        image_placeholder.warning(f"Letter image '{char}.jpeg' not found.")
                        time.sleep(0.5)
                elif char == ' ':
                    image_placeholder.empty() # Clear for space
                    time.sleep(0.3) # Shorter pause for space
            image_placeholder.empty() # Clear last letter


class AudioProcessor(VideoTransformerBase):
    def __init__(self):
        # --- State for this instance ---
        self.recognizer = sr.Recognizer()
        self.audio_buffer = []
        self.sample_rate = 0
        self.sample_width = 0
        self.channels = 0
        
        # --- Result Handling (Thread-Safe) ---
        self.result = None
        self.lock = threading.Lock()

    # This method is called by streamlit-webrtc for each audio frame
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
            nd_array = frame.to_ndarray()
            
            if not self.sample_rate:
                self.sample_rate = frame.sample_rate
                self.sample_width = nd_array.dtype.itemsize
                self.channels = frame.layout.nb_channels
                print(f"DEBUG: Audio properties set (Rate: {self.sample_rate}, Width: {self.sample_width}, Channels: {self.channels})")

            self.audio_buffer.append(nd_array.tobytes())
        except Exception as e:
            print(f"Error processing audio frame: {e}")

        return frame

    def on_ended(self):
        print("DEBUG: Audio stream ended. Processing buffer.")
        
        if not self.audio_buffer or not self.sample_rate:
            print("DEBUG: Audio buffer is empty or properties not set.")
            with self.lock:
                self.result = "No audio received."
            self.audio_buffer.clear()
            return

        result_text = "An error occurred during recognition."
        try:
            combined_audio_data = b"".join(self.audio_buffer)
            
            segment = pydub.AudioSegment(
                data=combined_audio_data,
                sample_width=self.sample_width,
                frame_rate=self.sample_rate,
                channels=self.channels
            )
            
            wav_buffer = BytesIO()
            segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)

            with sr.AudioFile(wav_buffer) as source:
                print("DEBUG: Recording from buffer...")
                audio_data = self.recognizer.record(source)
            
            print("DEBUG: Recognizing speech...")
            recognized_text = self.recognizer.recognize_google(audio_data, language="en-IN").lower()
            print(f"DEBUG: Recognized text: {recognized_text}")
            result_text = recognized_text
            
        except sr.UnknownValueError:
            print("DEBUG: SpeechRecognition could not understand audio.")
            result_text = "Audio unclear or not understood."
        except sr.RequestError as e:
            print(f"DEBUG: SpeechRecognition service error: {e}")
            result_text = f"Recognition service error; {e}"
        except Exception as e:
            print(f"DEBUG: Error during final recognition: {e}")
            result_text = "An error occurred during recognition."
        
        finally:
            # --- Store result safely ---
            with self.lock:
                self.result = result_text
            
            # --- Cleanup ---
            self.audio_buffer.clear()
            self.sample_rate = 0
            self.sample_width = 0
            self.channels = 0
            if 'wav_buffer' in locals():
                wav_buffer.close()

# (near your other session state inits)
if 'processing_audio' not in st.session_state:
    st.session_state.processing_audio = False

# --- Background Speech Recognition Callback ---
def recognition_callback(recognizer, audio):
    """Callback for background listener in Tab 4."""
    try:
        recognized_text = recognizer.recognize_google(audio, language="en-IN").lower()
        st.session_state.recognition_result = recognized_text
        st.session_state.voice_to_display = recognized_text # Trigger display
    except sr.UnknownValueError:
        st.session_state.recognition_result = "Audio unclear or not understood."
    except sr.RequestError as e:
        st.session_state.recognition_result = f"Recognition service error; {e}"
    finally:
        # Ensure state is reset
        st.session_state.listening = False
        st.session_state.stop_listening_callback = None

# --- Initialize Session State Variables ---
# Tab 4 (Voice to Sign) state
if "listening" not in st.session_state:
    st.session_state.listening = False
if "stop_listening_callback" not in st.session_state:
    st.session_state.stop_listening_callback = None
if "recognition_result" not in st.session_state:
    st.session_state.recognition_result = ""
if 'voice_to_display' not in st.session_state:
    st.session_state.voice_to_display = ""
# Tab 5 (Sign to Text) state
if "composed_text_webrtc" not in st.session_state:
    st.session_state.composed_text_webrtc = ""

# --- Sign to Text Video Processor Class ---
class SignPredictor(VideoTransformerBase):
    def __init__(self, result_queue: queue.Queue):
        # Stability logic state variables (local to this instance)
        self.last_committed_prediction = ""
        self.current_stable_prediction = ""
        self.prediction_stability_counter = 0
        self.stability_threshold = STABILITY_THRESHOLD
        self.result_queue = result_queue

        # MediaPipe (ensure it's loaded before creating instance)
        if not mp_hands or not mp_drawing or not hands_detector:
             # This should ideally not happen due to checks before creating, but is a safeguard
             raise RuntimeError("MediaPipe components failed to load.")
        self.mp_hands = mp_hands
        self.mp_drawing = mp_drawing
        self.hands_detector = hands_detector

        # Feature constants
        self.expected_len = EXPECTED_FEATURE_LENGTH
        self.features_per_hand = FEATURES_PER_HAND

    def _extract_hand_features(self, hand_landmarks):
        """Helper to extract features using the exact isl_predict.py logic."""
        feature = []
        try:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            for lm in hand_landmarks.landmark:
                feature.extend([
                    (lm.x - x_min) / (x_max - x_min + 1e-6), # Add epsilon for stability
                    (lm.y - y_min) / (y_max - y_min + 1e-6),
                    lm.z
                ])

            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP] # Use landmark names
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = np.sqrt(
                (thumb_tip.x - index_tip.x)**2 +
                (thumb_tip.y - index_tip.y)**2 +
                (thumb_tip.z - index_tip.z)**2
            )
            feature.append(distance)
        except Exception as e:
            print(f"Error extracting features: {e}") # Log error, return empty
            return []
        return feature

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Processes each frame received from the webcam."""
        img = frame.to_ndarray(format="bgr24")
        live_prediction_text = "..." # Default text on frame
        hands_detected = False

        # Pre-process frame
        img.flags.writeable = False # Improve performance
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands_detector.process(img_rgb)
        img.flags.writeable = True # Make writable again for drawing

        feature = []

        # Hand Detection and Feature Extraction
        if result.multi_hand_landmarks:
            hands_detected = True
            # Process up to 2 hands
            for hand_landmarks in result.multi_hand_landmarks[:2]:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                # Extract features
                hand_features = self._extract_hand_features(hand_landmarks)
                if hand_features: # Only extend if extraction succeeded
                    feature.extend(hand_features)

        # Padding (always done, ensures fixed length)
        while len(feature) < self.expected_len:
            feature.extend([0.0] * self.features_per_hand) # Use float for consistency
        # Truncate if somehow > expected_len (shouldn't happen with max_num_hands=2)
        feature = feature[:self.expected_len]

        # Prediction and Stability Logic
        if hands_detected and len(feature) == self.expected_len:
            try:
                feature_tensor = torch.tensor([feature], dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    if model is None: # Check if model loaded
                         raise RuntimeError("Model is not loaded.")
                    output = model(feature_tensor)
                    _, predicted_idx = torch.max(output, 1)
                    live_prediction_text = class_labels.get(predicted_idx.item(), "???")

                # Stability Check
                if live_prediction_text == self.current_stable_prediction:
                    self.prediction_stability_counter += 1
                else:
                    self.current_stable_prediction = live_prediction_text
                    self.prediction_stability_counter = 0 # Reset counter on change

                # Commit Check (add to queue if stable and new)
                if self.prediction_stability_counter >= self.stability_threshold:
                    if live_prediction_text != self.last_committed_prediction and live_prediction_text not in ["...", "???"]:
                        print(f"DEBUG: Putting '{live_prediction_text}' into queue.") # <--- ADD THIS LINE
                        self.result_queue.put(live_prediction_text) # Send to main thread
                        self.last_committed_prediction = live_prediction_text
                        self.prediction_stability_counter = 0 # Reset after committing
            except Exception as e:
                print(f"Error during prediction or stability check: {e}") # Log error
                live_prediction_text = "Error"
        else:
            # No Hands Detected: Reset stability logic
            live_prediction_text = "..."
            self.current_stable_prediction = "..."
            self.prediction_stability_counter = 0
            # Allow immediate re-signing of the same letter after hand removal
            if self.last_committed_prediction not in ["...", " ", ""]:
                 self.last_committed_prediction = "..."

        # Draw prediction text on the frame
        cv2.putText(img, f'{live_prediction_text}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

        # Return the processed frame back to WebRTC
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- UI Tabs ---
st.title("Voyage Voice 🗻")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏠 Home", "🌍 Text Translator", "📝 Text to Sign", "🎤 Voice to Sign", "📹 Sign to Text", "ℹ️ About"])

# --- Home Tab ---
with tab1:
    st.header("Welcome!")
    st.markdown("Your one-stop application for breaking down communication barriers.")
    home_image_path = "hello.jpeg" # Optional: Add a nice welcome image
    if os.path.exists(home_image_path):
        st.image(home_image_path, caption="Empowering Communication Through Technology", use_container_width=True)
    st.subheader("Features:")
    st.markdown("""
        * **🌍 Text Translator:** Translate text between various languages.
        * **📝 Text to Sign:** See English text converted into gestures and audio.
        * **🎤 Voice to Sign:** Speak in English to see the corresponding gestures and hear the audio.
        * **📹 Sign to Text:** Use your webcam to translate signs (a-z, 0-9) into text in real-time.
    """)
    st.info("Select a tab above to explore the features!")

# --- Text Translator Tab ---
with tab2:
    st.header("General Text Translator")
    st.markdown("Translate text between supported languages using text or voice input.")
    lang_names, name_to_code_map = get_supported_languages_map()

    # Default selections (e.g., English to Hindi)
    try:
        default_source_index = lang_names.index('English')
    except ValueError:
        default_source_index = 0 # Fallback to first language
    try:
        default_target_index = lang_names.index('Hindi') if 'Hindi' in lang_names else (1 if len(lang_names) > 1 else 0)
    except ValueError:
        default_target_index = 1 if len(lang_names) > 1 else 0 # Fallback

    col1, col2 = st.columns(2)
    with col1:
        selected_source_lang_name = st.selectbox("From:", lang_names, index=default_source_index, key="source_lang")
    with col2:
        selected_target_lang_name = st.selectbox("To:", lang_names, index=default_target_index, key="target_lang")

    # Use session state to preserve text area content across reruns
    if "translator_input_text" not in st.session_state:
        st.session_state.translator_input_text = ""

    st.session_state.translator_input_text = st.text_area("Enter text:", value=st.session_state.translator_input_text, height=150, key="text_translator_area")

    button_col1, button_col2 = st.columns(2)
    with button_col1:
        # Microphone input (still uses blocking listen here, simpler for this tab)
        if st.button("Use Microphone 🎤", key="listen_general", use_container_width=True):
            recognizer = sr.Recognizer()
            with st.spinner("Listening..."):
                try:
                    with sr.Microphone() as source:
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        st.info("Speak now...")
                        # Increased timeouts for potentially slower connections
                        audio = recognizer.listen(source, timeout=7, phrase_time_limit=12)
                    recognized_text = recognizer.recognize_google(audio, language="en-IN")
                    st.session_state.translator_input_text = recognized_text
                    # No st.rerun() needed here, widget value change triggers update
                except sr.WaitTimeoutError:
                    st.warning("No speech detected within the time limit.")
                except sr.UnknownValueError:
                    st.error("Could not understand the audio.")
                except sr.RequestError as e:
                    st.error(f"Could not request results from Google Speech Recognition service; {e}")
                except Exception as e:
                    st.error(f"An error occurred during speech recognition: {e}")

    with button_col2:
        # Translate button
        if st.button("Translate Text 🔀", key="translate_general", type="primary", use_container_width=True):
            if st.session_state.translator_input_text:
                source_code = name_to_code_map.get(st.session_state.source_lang, "auto") # Use get for safety
                target_code = name_to_code_map.get(st.session_state.target_lang, "en")
                if source_code and target_code:
                    try:
                        with st.spinner("Translating..."):
                            translated = GoogleTranslator(source=source_code, target=target_code).translate(st.session_state.translator_input_text)
                        st.success("Translated Text:")
                        st.text_area("Result:", value=translated or "Translation failed.", height=150, key="translation_result_area")
                    except Exception as e:
                        st.error(f"An error occurred during translation: {e}")
                else:
                    st.error("Invalid source or target language selected.")
            else:
                st.warning("Please enter or say something to translate.")

# --- Text to Sign Tab ---
with tab3:
    st.header("Text to Sign Language")
    st.markdown("Enter English text to see gestures and hear the audio.")
    col1, col2 = st.columns([1, 1]) # Keeps the 50/50 split

    with col1:
        # Added subheader for better visual structure
        st.subheader("Input Text")
        input_text_sign = st.text_area("Enter English text:", height=150, key="text_to_sign_input")
        if st.button("Show Sign & Play Audio ▶️", key="text_translate_button"):
            if input_text_sign:
                # Store text temporarily for the output column to display
                st.session_state.text_to_display = input_text_sign
            else:
                st.warning("Please enter text first.")
                st.session_state.text_to_display = "" # Clear any previous display

    with col2:
        # Added matching subheader
        st.subheader("Output")
        # Kept container for layout, removed border for cleaner look
        text_output_container = st.container(height=400, border=False)
        # Display content if text_to_display is set
        if 'text_to_display' in st.session_state and st.session_state.text_to_display:
            display_sign_language_and_audio(st.session_state.text_to_display, text_output_container)
            st.session_state.text_to_display = "" # Clear after displaying
        else:
            text_output_container.info("Signs and audio will appear here.")

# --- Voice to Sign Tab ---
# =================================================================
# === NEW: Voice to Sign Tab (using streamlit-webrtc) ===
# =================================================================
# =================================================================
# === NEW: Voice to Sign Tab (Corrected State & Rerun Logic) ===
# =================================================================
# =================================================================
# === NEW: Voice to Sign Tab (FIXED with State Trigger) ===
# =================================================================

# --- Initialize the new trigger state ---
if 'new_audio_result' not in st.session_state:
    st.session_state.new_audio_result = False

# =================================================================
# === NEW: Voice to Sign Tab (Corrected State Logic) ===
# =================================S================================

# --- Initialize session state for the queue ---
if 'audio_result_queue' not in st.session_state:
    st.session_state.audio_result_queue = queue.Queue()
audio_result_queue = st.session_state.audio_result_queue

# --- Initialize session state for the display ---
if 'voice_to_display' not in st.session_state:
    st.session_state.voice_to_display = "" # Holds the text to be displayed
if 'last_recognition_result' not in st.session_state:
    st.session_state.last_recognition_result = "" # Holds the status text

# =================================================================
# === NEW: Voice to Sign Tab (Thread-Safe) ===
# =================================================================

# --- Initialize session state for the processor and display text ---
if 'audio_processor' not in st.session_state:
    st.session_state.audio_processor = AudioProcessor()
if 'voice_to_display' not in st.session_state:
    st.session_state.voice_to_display = ""
if 'last_recognition_result' not in st.session_state:
    st.session_state.last_recognition_result = ""

# =================================================================
# === NEW: Voice to Sign Tab (FIXED with Polling Logic) ===
# =================================================================

with tab4:
    st.header("Voice 🎤 ▶️ Sign Language")
    st.markdown("Click 'Start' below to activate your microphone, speak, and then click 'Stop'.")

    col1, col2 = st.columns([1, 1])
    processor = st.session_state.audio_processor # Get the persistent processor

    # --- KEY FIX: Polling Logic ---
    # Are we currently in the "processing" state?
    if st.session_state.processing_audio:
        # Show a spinner and check for the result
        with st.spinner("Processing audio..."):
            result = None
            with processor.lock:
                if processor.result:
                    result = processor.result
                    processor.result = None # Clear it after getting
            
            if result:
                # FOUND IT! Stop polling
                st.session_state.processing_audio = False 
                st.session_state.last_recognition_result = result
                if "error" not in result.lower() and "unclear" not in result.lower() and "no audio" not in result.lower():
                    st.session_state.voice_to_display = result
                st.rerun() # Rerun one last time to show the result
            else:
                # NOT READY YET. Wait and poll again.
                time.sleep(0.25) # Wait 250ms
                st.rerun() # Rerun to check again

    # --- Main UI Rendering (if not processing) ---
    else:
        with col1:
            st.subheader("Input (Your Voice)")
            
            ctx = webrtc_streamer(
                key="voice-to-sign-webrtc",
                mode=WebRtcMode.SENDONLY, 
                audio_processor_factory=lambda: processor,
                media_stream_constraints={"video": False, "audio": True},
                rtc_configuration=RTC_CONFIGURATION,
                async_processing=True,
            )
            
            # --- Check if the user *just* clicked "Stop" ---
            # 'webrtc_is_playing' stores the *previous* state
            if 'webrtc_is_playing' not in st.session_state:
                st.session_state.webrtc_is_playing = False
            
            if not ctx.state.playing and st.session_state.webrtc_is_playing:
                # If we *were* playing, but now we're not, "Stop" was just clicked.
                # Start the "processing" state.
                st.session_state.processing_audio = True
                st.session_state.webrtc_is_playing = False
                st.rerun() # Rerun immediately to start the spinner

            # Store the current playing state for comparison next time
            st.session_state.webrtc_is_playing = ctx.state.playing

            # --- Display Status ---
            if ctx.state.playing:
                st.success("✅ Microphone is active. Speak now, then click 'Stop'.")
            else:
                st.info("ℹ️ Click 'Start' above to activate your microphone.")

            # Display the last status message (if any)
            if st.session_state.last_recognition_result:
                if "error" in st.session_state.last_recognition_result.lower() or "unclear" in st.session_state.last_recognition_result.lower():
                    st.warning(f"Status: {st.session_state.last_recognition_result}")
                else:
                    st.success(f"You said: \"{st.session_state.last_recognition_result}\"")
                # Clear the status so it's only shown once
                st.session_state.last_recognition_result = ""

        with col2:
            st.subheader("Output (Sign Language)")
            voice_output_container = st.container(height=400, border=False)
            
            # This block now runs *after* the polling logic is complete
            if st.session_state.voice_to_display:
                display_sign_language_and_audio(st.session_state.voice_to_display, voice_output_container)
                st.session_state.voice_to_display = "" # Clear it
            else:
                voice_output_container.info("Signs and audio will appear here after you speak and click 'Stop'.")

# --- Sign to Text Tab ---
with tab5:
    st.header("Sign Language to Text")
    st.markdown("Allow webcam access. {Hold a sign steady to add it}")

    # Check necessary components are loaded
    if model is None:
        st.error("Sign language model failed to load. This feature is unavailable.")
    elif not class_labels:
         st.error("Sign language labels failed to load. Cannot map predictions to characters.")
    elif mp_hands is None:
         st.error("MediaPipe Hands failed to initialize. Cannot detect hands.")
    else:
        # --- Ensure Queue is Initialized in Session State ---
        if 'result_queue' not in st.session_state:
            st.session_state.result_queue = queue.Queue()
        result_queue = st.session_state.result_queue

        # --- Initialize Session State for Collage Visibility ---
        if 'show_sign_chart' not in st.session_state:
            st.session_state.show_sign_chart = False # Hidden by default

        # --- Back to Two-column layout ---
        col1, col2 = st.columns([2, 1]) # Video feed | Composed Text

        with col1:
            st.subheader("Live Feed")
            # Start the WebRTC streamer component
            ctx = webrtc_streamer(
                key="sign-to-text-webrtc",
                video_processor_factory=lambda: SignPredictor(result_queue=result_queue),
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
                async_processing=True,
            )

            # Display status message
            if ctx.state.playing:
                st.success("✅ Webcam active")
            else:
                 st.info("ℹ️ Click 'Start' above to activate webcam.")

            st.markdown("---") # Separator

            # --- NEW: Button to toggle Sign Chart ---
            button_text = "Hide Sign Chart" if st.session_state.show_sign_chart else "Show Sign Chart"
            if st.button(button_text, use_container_width=True, key="toggle_chart"):
                st.session_state.show_sign_chart = not st.session_state.show_sign_chart
                st.rerun() # Rerun needed to update the display immediately

            # --- NEW: Display the collage image if button clicked ---
            if st.session_state.show_sign_chart:
                # *** IMPORTANT: Replace 'sign_collage.jpg' with the actual filename of your collage image ***
                collage_image_path = "signs.png"
                if os.path.exists(collage_image_path):
                    try:
                        st.image(collage_image_path, caption="Sign Chart", use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading collage image: {e}")
                else:
                    st.warning(f"Sign chart image '{collage_image_path}' not found.")


        # --- Composed Text Column (Logic is the same, just in col2 now) ---
        with col2:
            st.subheader("Composed Text")
            if "composed_text_webrtc" not in st.session_state:
                st.session_state.composed_text_webrtc = ""

            # --- Process the queue ---
            newly_added_text = ""
            while not result_queue.empty():
                try:
                    result = result_queue.get_nowait()
                    print(f"DEBUG: Got '{result}' from queue.")
                    if result and result not in ["...", "???"]: # Ensure it's a valid character
                        newly_added_text += result
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"Error getting from queue: {e}")
                    break

            # --- Update Session State ---
            if newly_added_text:
                st.session_state.composed_text_webrtc += newly_added_text
                print(f"DEBUG: Session state updated to: '{st.session_state.composed_text_webrtc}'")
                st.rerun() # Trigger rerun to update text_area

            # Display the composed text
            st.text_area("Sentence:", value=st.session_state.composed_text_webrtc, height=300, key="composed_sentence_webrtc") # Adjusted height

            # Edit Buttons
            b_col1_edit, b_col2_edit = st.columns(2)
            with b_col1_edit:
                if st.button("Add Space", use_container_width=True, key="space_webrtc"):
                    st.session_state.composed_text_webrtc += " "
                    st.rerun()
            with b_col2_edit:
                if st.button("Backspace ⌫", use_container_width=True, key="backspace_webrtc"):
                    if len(st.session_state.composed_text_webrtc) > 0:
                        st.session_state.composed_text_webrtc = st.session_state.composed_text_webrtc[:-1]
                        st.rerun()

            # Clear Button
            if st.button("Clear Text 🗑️", use_container_width=True, key="clear_webrtc"):
                st.session_state.composed_text_webrtc = ""
                # No need to clear image path anymore
                st.rerun()

# ---About---
# --- NEW: About Tab ---
with tab6:
    st.header("ℹ️ About This Application")
    st.markdown("""
        This application is designed to facilitate communication using **Sign Language**.
        It leverages modern technologies to bridge the gap between spoken/written languages and sign language.

        **Developed By:** Karan Verma (Student ID:)

        **Core Technologies Used:**
        * **Streamlit:** For creating the interactive web application interface.
        * **Python:** The primary programming language.
        * **OpenCV:** For webcam access and image processing.
        * **MediaPipe:** For real-time hand landmark detection.
        * **PyTorch:** For the Sign-to-Text deep learning model.
        * **SpeechRecognition:** For converting voice to text.
        * **gTTS (Google Text-to-Speech):** For generating audio output.
        * **deep-translator:** For general text translation.

        **Features:**
        * Translate text between various languages.
        * Convert English text into ISL gestures (visuals and audio).
        * Convert spoken English into ISL gestures (visuals and audio).
        * Translate real-time ISL signs (a-z, 0-9) captured via webcam into text.

        Feel free to explore the different tabs to utilize these features!
    """)
    st.info("Version: 1.0.0 | Last Updated: October 2025")
