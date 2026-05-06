import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

import csv
import json
import time
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty

import cv2
import face_recognition
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model

from db import UserDatabase

# ================== CONFIG ==================
CAM_INDEX = 0
FRAME_W = 640
FRAME_H = 480

ACCESS_RECORDS_FILE = "weapon_access_records.csv"
USERS_JSON_PATH = "users.json"
USERS_DB_PATH = "users.db"
KNOWN_FACES_DIRS = ["known", "known_faces"]

# ===== GESTURE MODEL (LSTM) =====
GESTURE_MODEL_PATH = "./gesture_model_az.h5"      # model neutral + A-Z
GESTURE_LABELS_PATH = "./gesture_labels_az.pkl"   # label map neutral + A-Z
NO_OF_TIMESTEPS = 20
GESTURE_CONFIDENCE = 0.65
GESTURE_STABLE_FRAMES = 5
GESTURE_VALIDATE_DELAY_SECONDS = 1.0
GESTURE_DETECT_EVERY_N = 2
NO_HAND_RESET_FRAMES = 8
GESTURE_WINDOW_SECONDS = 1.5
GESTURE_MIN_SAMPLES = 4
RESULT_POPUP_SECONDS = 4.0

# ===== FACE RECOGNITION =====
FACE_DETECT_INTERVAL_SECONDS = 0.6
FACE_SAMPLE_TARGET = 8
FACE_MATCH_TOLERANCE = 0.5
FACE_DOWNSCALE = 0.5

# ===== FLOW =====
DETECTION_INTERVAL_SECONDS = 8.0
GESTURE_COOLDOWN_SECONDS = 2.0
FACE_LOCK_SECONDS = 8.0

# ===== GPIO =====
SOLENOID_PIN = 26
SOLENOID_OPEN_SECONDS = 2.0

# ================== MediaPipe ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils

# ================== BUTTON CLASS ==================
class Button:
    def __init__(self, x, y, w, h, text, color=(100, 100, 100), text_color=(255, 255, 255)):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.color = color
        self.text_color = text_color
        self.hover_color = tuple(min(c + 40, 255) for c in color)
        self.is_hovered = False
        
    def draw(self, frame):
        color = self.hover_color if self.is_hovered else self.color
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), color, -1)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (255, 255, 255), 2)
        
        text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = self.x + (self.w - text_size[0]) // 2
        text_y = self.y + (self.h + text_size[1]) // 2
        cv2.putText(frame, self.text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 2)
    
    def is_clicked(self, mouse_x, mouse_y):
        return self.x <= mouse_x <= self.x + self.w and self.y <= mouse_y <= self.y + self.h
    
    def check_hover(self, mouse_x, mouse_y):
        self.is_hovered = self.is_clicked(mouse_x, mouse_y)

# ================== HELPER FUNCTIONS ==================
def draw_text_with_outline(img, text, pos, font, scale, color, thickness):
    x, y = pos
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

def draw_animated_popup(frame, title, subtitle, progress=1.0, status="success"):
    """Draw animated popup with progress bar."""
    h, w = frame.shape[:2]
    box_w = 800
    box_h = 300
    
    box_x = (w - box_w) // 2
    box_y = (h - box_h) // 2
    
    if status == "success":
        box_color = (0, 180, 0)
        title_color = (255, 255, 255)
        border_color = (0, 255, 0)
    elif status == "detected":
        box_color = (0, 140, 255)
        title_color = (255, 255, 255)
        border_color = (0, 200, 255)
    elif status == "failed":
        box_color = (0, 0, 200)
        title_color = (255, 255, 255)
        border_color = (0, 100, 255)
    else:
        box_color = (60, 60, 60)
        title_color = (255, 255, 255)
        border_color = (150, 150, 150)
    
    overlay = frame.copy()
    shadow_offset = 10
    cv2.rectangle(overlay, 
                  (box_x + shadow_offset, box_y + shadow_offset), 
                  (box_x + box_w + shadow_offset, box_y + box_h + shadow_offset),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), box_color, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    cv2.rectangle(frame, (box_x-2, box_y-2), (box_x + box_w+2, box_y + box_h+2), border_color, 6)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 2)
    
    icon_y = box_y + 60
    icon_x = box_x + box_w // 2
    if status == "success":
        cv2.circle(frame, (icon_x, icon_y), 35, (255, 255, 255), 4)
        cv2.line(frame, (icon_x - 15, icon_y), (icon_x - 5, icon_y + 15), (255, 255, 255), 4)
        cv2.line(frame, (icon_x - 5, icon_y + 15), (icon_x + 15, icon_y - 10), (255, 255, 255), 4)
    elif status == "detected":
        cv2.circle(frame, (icon_x, icon_y), 35, (255, 255, 255), 4)
        cv2.line(frame, (icon_x, icon_y - 15), (icon_x, icon_y + 5), (255, 255, 255), 5)
        cv2.circle(frame, (icon_x, icon_y + 15), 3, (255, 255, 255), -1)
    elif status == "failed":
        cv2.circle(frame, (icon_x, icon_y), 35, (255, 255, 255), 4)
        cv2.line(frame, (icon_x - 12, icon_y - 12), (icon_x + 12, icon_y + 12), (255, 255, 255), 5)
        cv2.line(frame, (icon_x + 12, icon_y - 12), (icon_x - 12, icon_y + 12), (255, 255, 255), 5)
    
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.8, 3)[0]
    title_x = box_x + (box_w - title_size[0]) // 2
    title_y = box_y + 140
    cv2.putText(frame, title, (title_x, title_y), 
                cv2.FONT_HERSHEY_DUPLEX, 1.8, title_color, 3, cv2.LINE_AA)
    
    subtitle_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)[0]
    subtitle_x = box_x + (box_w - subtitle_size[0]) // 2
    subtitle_y = box_y + 190
    cv2.putText(frame, subtitle, (subtitle_x, subtitle_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    
    if progress < 1.0:
        bar_y = box_y + box_h - 40
        bar_x = box_x + 50
        bar_w = box_w - 100
        bar_h = 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        progress_w = int(bar_w * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_w, bar_y + bar_h), border_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)


def normalize_gesture_label(label):
    if label is None:
        return ""
    text = str(label).strip().upper()
    for ch in text:
        if "A" <= ch <= "Z":
            return ch
    return ""

class VoiceGuide:
    STATIC_PROMPTS = [
        "Silakan tampilkan gesture pertama.",
        "Silakan tampilkan gesture kedua.",
        "Arahkan wajah ke kamera.",
        "Gesture pertama tidak valid. Ulangi gesture pertama.",
        "Gesture kedua tidak valid. Ulangi gesture kedua.",
        "Akses diterima.",
        "Akses ditolak.",
    ]

    def __init__(self, cache_dir="assets/voice_cache", language="id"):
        self._queue = Queue()
        self._gtts = None
        self._pygame = None
        self._available = False
        self._backend_name = "none"
        self._cache_dir = Path(cache_dir)
        self._language = language
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            from gtts import gTTS
            import pygame

            self._gtts = gTTS
            self._pygame = pygame
            self._pygame.mixer.init()
            self._available = True
            self._backend_name = "gTTS"
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()

            for prompt in self.STATIC_PROMPTS:
                self._ensure_audio(prompt)
        except Exception:
            self._gtts = None
            self._pygame = None
            self._available = False
            self._backend_name = "unavailable"

    @property
    def available(self):
        return self._available

    @property
    def backend_name(self):
        return self._backend_name

    def speak(self, text):
        if not self._available or not text:
            return
        self._queue.put(text)

    def stop(self):
        if self._available:
            try:
                self._pygame.mixer.music.stop()
            except Exception:
                pass
            self._queue.put(None)

    def interrupt(self):
        if not self._available:
            return

        try:
            self._pygame.mixer.music.stop()
        except Exception:
            pass

        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def _audio_path(self, text):
        key = hashlib_sha1(text)
        return self._cache_dir / f"{key}.mp3"

    def _ensure_audio(self, text):
        path = self._audio_path(text)
        if path.exists():
            return path

        speaker = self._gtts(text=text, lang=self._language, slow=False)
        speaker.save(str(path))
        return path

    def _run(self):
        while True:
            text = self._queue.get()
            if text is None:
                break

            try:
                audio_path = self._ensure_audio(text)
                self._pygame.mixer.music.load(str(audio_path))
                self._pygame.mixer.music.play()
                while self._pygame.mixer.music.get_busy():
                    time.sleep(0.05)
            except Exception:
                continue


class SolenoidController:
    def __init__(self, pin):
        self._pin = pin
        self._device = None
        self._available = False
        self._lock = threading.Lock()
        self._is_open = False
        self._close_token = 0

        self._init_device()

    def _init_device(self):
        try:
            from gpiozero import Device, LED
            from gpiozero.pins.lgpio import LGPIOFactory

            Device.pin_factory = LGPIOFactory()
            self._device = LED(self._pin, active_high=False)
            self._available = True
        except Exception as e:
            self._device = None
            self._available = False
            print("GPIO init failed:", e)

    def _ensure_device(self):
        if self._device is None or getattr(self._device, "closed", False):
            self._init_device()

    def unlock(self, duration=3):
        if not self._available:
            return

        def _run():
            with self._lock:
                self._close_token += 1
                token = self._close_token
                self._ensure_device()
                if not self._available:
                    return
                try:
                    print("Solenoid UNLOCK")
                    self._device.on()
                    self._is_open = True

                    time.sleep(3)

                    self._device.off()
                    self._is_open = False
                    print("Solenoid LOCK inside")
                except Exception as e:
                    print("GPIO on failed:", e)
                    self._available = False
                    return

            time.sleep(duration)

            with self._lock:
                if token != self._close_token:
                    return
                self._cleanup_and_reinit_locked()

        threading.Thread(target=_run, daemon=True).start()

    def _cleanup_and_reinit_locked(self):
        try:
            if self._is_open:
                self._device.off()
                print("Solenoid LOCK")
        except Exception as e:
            print("GPIO off failed:", e)
        finally:
            if self._device:
                try:
                    self._device.close()
                except Exception:
                    pass
            self._device = None
            self._is_open = False
            self._available = False
            self._init_device()

    def cleanup(self):
        with self._lock:
            self._close_token += 1
        if self._device:
            try:
                self._device.off()
            except Exception:
                pass
            self._device.close()


def hashlib_sha1(text):
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def load_gesture_model():
    if not os.path.exists(GESTURE_MODEL_PATH):
        print("[ERROR] gesture_model_az.h5 not found")
        return None, {}

    try:
        model = load_model(GESTURE_MODEL_PATH, compile=False)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return None, {}

    labels = {}
    if os.path.exists(GESTURE_LABELS_PATH):
        with open(GESTURE_LABELS_PATH, "rb") as f:
            labels.update(pickle.load(f))

    print("[GESTURE] Model OK")
    return model, labels


def ensure_access_records_file():
    if not os.path.exists(ACCESS_RECORDS_FILE):
        with open(ACCESS_RECORDS_FILE, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Tanggal",
                "Waktu",
                "Username",
                "Gesture 1",
                "Gesture 2",
                "Status",
                "Screenshot Path",
            ])


def record_access(username, gesture_1, gesture_2, status, screenshot_path):
    current_datetime = datetime.now()
    date_str = current_datetime.strftime("%Y-%m-%d")
    time_str = current_datetime.strftime("%H:%M:%S")
    with open(ACCESS_RECORDS_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            date_str,
            time_str,
            username,
            gesture_1 or "-",
            gesture_2 or "-",
            status,
            screenshot_path or "",
        ])


def take_screenshot(frame, name):
    os.makedirs("access_screenshots", exist_ok=True)
    current_datetime = datetime.now()
    date_str = current_datetime.strftime("%Y-%m-%d")
    time_str = current_datetime.strftime("%H%M%S")
    filename = f"access_{name}_{date_str}_{time_str}.jpg"
    filepath = os.path.join("access_screenshots", filename)
    cv2.imwrite(filepath, frame)
    return filepath


def load_users_registry():
    path = Path(USERS_JSON_PATH)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("Failed to read users.json:", exc)
        return []

    if isinstance(data, dict):
        users = data.get("users", [])
    elif isinstance(data, list):
        users = data
    else:
        return []

    return [user for user in users if isinstance(user, dict)]


def resolve_face_path(entry):
    face_path = entry.get("face_path") or entry.get("face")
    if face_path:
        path = Path(face_path)
        if not path.is_absolute():
            for base in KNOWN_FACES_DIRS:
                candidate = Path(base) / path
                if candidate.exists():
                    return candidate
        if path.exists():
            return path

    username = str(entry.get("username", "")).strip()
    if not username:
        return None

    for base in KNOWN_FACES_DIRS:
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = Path(base) / f"{username}{ext}"
            if candidate.exists():
                return candidate

    return None


def sync_users_from_registry(db):
    users = load_users_registry()
    if not users:
        return

    for entry in users:
        username = str(entry.get("username", "")).strip()
        gesture_1 = normalize_gesture_label(entry.get("gesture_1", ""))
        gesture_2 = normalize_gesture_label(entry.get("gesture_2", ""))

        if not username or not gesture_1 or not gesture_2:
            print("Invalid registry entry, missing fields:", entry)
            continue

        if db.user_exists(username):
            continue

        face_path = resolve_face_path(entry)
        if face_path is None:
            print(f"Face reference missing for '{username}'")
            continue

        try:
            image = face_recognition.load_image_file(str(face_path))
            encodings = face_recognition.face_encodings(image)
            if not encodings:
                print(f"No face found for '{username}'")
                continue
            success, message = db.add_user(username, gesture_1, gesture_2, encodings[0])
            if not success:
                print(message)
        except Exception as exc:
            print(f"Failed to add '{username}':", exc)


def remove_user_from_registry(username):
    path = Path(USERS_JSON_PATH)
    if not path.exists():
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("Failed to read users.json:", exc)
        return

    if isinstance(data, dict):
        users = data.get("users", [])
    elif isinstance(data, list):
        users = data
    else:
        return

    updated = [
        user
        for user in users
        if str(user.get("username", "")).strip() != username
    ]

    if len(updated) == len(users):
        return

    output = {"users": updated} if isinstance(data, dict) else updated
    try:
        path.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    except Exception as exc:
        print("Failed to update users.json:", exc)


def load_users_cache(db):
    users = db.get_all_users_with_encoding()
    for user in users:
        user["gesture_1"] = normalize_gesture_label(user.get("gesture_1"))
        user["gesture_2"] = normalize_gesture_label(user.get("gesture_2"))
    return users


def encode_faces(frame):
    small = cv2.resize(frame, (0, 0), fx=FACE_DOWNSCALE, fy=FACE_DOWNSCALE)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, locations)
    return encodings

# ================== MOUSE CALLBACK ==================
mouse_x, mouse_y = 0, 0
mouse_clicked = False

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_clicked
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True

# ================== MAIN ==================
def main():
    global mouse_clicked

    gesture_model, gesture_labels = load_gesture_model()
    if gesture_model is None:
        return

    db = UserDatabase(USERS_DB_PATH)
    sync_users_from_registry(db)
    users_cache = load_users_cache(db)
    ensure_access_records_file()

    voice = VoiceGuide()
    solenoid = SolenoidController(SOLENOID_PIN)

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        print("[ERROR] Camera failed")
        return

    print("\n" + "=" * 70)
    print("SISTEM AKSES: GESTURE + FACE (LSTM)")
    print("=" * 70)

    window_name = "SISTEM AKSES - GESTURE + FACE"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, mouse_callback)

    menu_btn_detect = Button(FRAME_W // 2 - 250, FRAME_H // 2 - 50, 200, 100, "MODE DETEKSI", (0, 120, 0))
    menu_btn_users = Button(FRAME_W // 2 + 50, FRAME_H // 2 - 50, 200, 100, "MANAGE USER", (0, 120, 200))

    menu_bg = cv2.imread("image.jpg")
    if menu_bg is not None:
        menu_bg = cv2.resize(menu_bg, (FRAME_W, FRAME_H))

    mode = None
    status_message = ""
    status_time = 0

    gesture_seq_buffer = deque(maxlen=NO_OF_TIMESTEPS)
    recent_predictions = deque()

    detect_state = {
        "stage": "idle",
        "last_detection_ts": 0.0,
        "cooldown_until": 0.0,
        "gesture_cooldown_until": 0.0,
        "frame_index": 0,
        "no_hand_frames": 0,
        "validation_pending": False,
        "validation_due": 0.0,
        "candidate_letter": None,
        "stable_letter": None,
        "stable_count": 0,
        "gesture_1": None,
        "candidate_users": [],
        "face_samples": [],
        "last_face_ts": 0.0,
        "face_start_ts": 0.0,
        "result_title": "",
        "result_subtitle": "",
        "result_status": "",
        "result_until": 0.0,
    }

    users_selected_index = 0

    def refresh_users_cache():
        nonlocal users_cache, users_selected_index
        sync_users_from_registry(db)
        users_cache = load_users_cache(db)
        if users_selected_index >= len(users_cache):
            users_selected_index = max(0, len(users_cache) - 1)

    def reset_detect_state():
        detect_state.update({
            "stage": "idle",
            "cooldown_until": 0.0,
            "gesture_cooldown_until": 0.0,
            "frame_index": 0,
            "no_hand_frames": 0,
            "validation_pending": False,
            "validation_due": 0.0,
            "candidate_letter": None,
            "stable_letter": None,
            "stable_count": 0,
            "gesture_1": None,
            "candidate_users": [],
            "face_samples": [],
            "last_face_ts": 0.0,
            "face_start_ts": 0.0,
            "result_title": "",
            "result_subtitle": "",
            "result_status": "",
            "result_until": 0.0,
        })
        gesture_seq_buffer.clear()
        recent_predictions.clear()

    def set_status(message):
        nonlocal status_message, status_time
        status_message = message
        status_time = time.time()

    def draw_status(frame):
        if status_message and (time.time() - status_time) < 3.5:
            draw_text_with_outline(frame, status_message, (30, FRAME_H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    def update_gesture_state(detected_letter, hand_present, now_ts):
        if not hand_present:
            detect_state["no_hand_frames"] += 1
            if detect_state["no_hand_frames"] >= NO_HAND_RESET_FRAMES:
                detect_state["stable_letter"] = None
                detect_state["stable_count"] = 0
                detect_state["validation_pending"] = False
                recent_predictions.clear()
            return

        detect_state["no_hand_frames"] = 0
        if detected_letter and detected_letter != "neutral" and detected_letter != "-":
            recent_predictions.append((now_ts, detected_letter))

        while recent_predictions and (now_ts - recent_predictions[0][0]) > GESTURE_WINDOW_SECONDS:
            recent_predictions.popleft()

        if len(recent_predictions) < GESTURE_MIN_SAMPLES:
            detect_state["stable_letter"] = None
            detect_state["stable_count"] = 0
            return

        counts = {}
        for _, letter in recent_predictions:
            counts[letter] = counts.get(letter, 0) + 1

        stable_letter = max(counts, key=counts.get)
        stable_count = counts[stable_letter]
        detect_state["stable_letter"] = stable_letter
        detect_state["stable_count"] = min(stable_count, GESTURE_STABLE_FRAMES)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        current_time = time.monotonic()
        h, w = frame.shape[:2]

        if mode is None:
            if menu_bg is not None:
                frame = menu_bg.copy()
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            draw_text_with_outline(frame, "PILIH MODE OPERASI", (w // 2 - 250, h // 2 - 150), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
            menu_btn_detect.draw(frame)
            menu_btn_users.draw(frame)
            menu_btn_detect.check_hover(mouse_x, mouse_y)
            menu_btn_users.check_hover(mouse_x, mouse_y)

            if mouse_clicked:
                if menu_btn_detect.is_clicked(mouse_x, mouse_y):
                    mode = "detect"
                    reset_detect_state()
                    voice.interrupt()
                    voice.speak("Silakan tampilkan gesture pertama.")
                    print("\nMode: DETEKSI")
                elif menu_btn_users.is_clicked(mouse_x, mouse_y):
                    mode = "users"
                    refresh_users_cache()
                    print("\nMode: MANAGE USER")
                mouse_clicked = False

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        if mode == "users":
            frame[:] = (25, 25, 25)
            draw_text_with_outline(frame, "MANAGE USER", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2)

            list_top = 100
            list_left = 30
            list_width = w - 60
            list_height = h - 220
            cv2.rectangle(frame, (list_left, list_top), (list_left + list_width, list_top + list_height), (40, 40, 40), -1)
            cv2.rectangle(frame, (list_left, list_top), (list_left + list_width, list_top + list_height), (0, 255, 255), 2)

            if not users_cache:
                draw_text_with_outline(frame, "Belum ada user", (list_left + 20, list_top + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            else:
                visible_count = 8
                start_idx = max(0, users_selected_index - visible_count + 1)
                for i in range(start_idx, min(start_idx + visible_count, len(users_cache))):
                    user = users_cache[i]
                    y = list_top + 40 + (i - start_idx) * 45
                    color = (0, 255, 0) if i == users_selected_index else (220, 220, 220)
                    text = f"{user['username']}  |  {user['gesture_1']} -> {user['gesture_2']}"
                    cv2.putText(frame, text, (list_left + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            btn_up = Button(30, h - 100, 100, 60, "UP", (70, 70, 70))
            btn_down = Button(150, h - 100, 120, 60, "DOWN", (70, 70, 70))
            btn_delete = Button(290, h - 100, 160, 60, "DELETE", (150, 50, 50))
            btn_refresh = Button(470, h - 100, 160, 60, "REFRESH", (0, 120, 200))
            btn_sync = Button(650, h - 100, 180, 60, "SYNC JSON", (0, 120, 200))
            btn_back = Button(w - 220, 20, 190, 60, "KEMBALI", (100, 100, 100))

            for btn in [btn_up, btn_down, btn_delete, btn_refresh, btn_sync, btn_back]:
                btn.check_hover(mouse_x, mouse_y)
                btn.draw(frame)

            if mouse_clicked:
                if btn_up.is_clicked(mouse_x, mouse_y):
                    users_selected_index = max(0, users_selected_index - 1)
                elif btn_down.is_clicked(mouse_x, mouse_y):
                    users_selected_index = min(len(users_cache) - 1, users_selected_index + 1) if users_cache else 0
                elif btn_delete.is_clicked(mouse_x, mouse_y) and users_cache:
                    username = users_cache[users_selected_index]["username"]
                    success, message = db.delete_user(username)
                    if success:
                        remove_user_from_registry(username)
                    refresh_users_cache()
                    set_status(message)
                elif btn_refresh.is_clicked(mouse_x, mouse_y):
                    refresh_users_cache()
                    set_status("Data user diperbarui")
                elif btn_sync.is_clicked(mouse_x, mouse_y):
                    sync_users_from_registry(db)
                    refresh_users_cache()
                    set_status("users.json disinkronkan")
                elif btn_back.is_clicked(mouse_x, mouse_y):
                    mode = None
                mouse_clicked = False

            draw_status(frame)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        if mode == "detect":
            draw_text_with_outline(frame, "MODE: DETEKSI", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
            back_btn = Button(w - 220, 20, 200, 60, "KEMBALI", (100, 100, 100))
            back_btn.check_hover(mouse_x, mouse_y)
            back_btn.draw(frame)

            if mouse_clicked and back_btn.is_clicked(mouse_x, mouse_y):
                mode = None
                reset_detect_state()
                voice.interrupt()
                mouse_clicked = False

            if not users_cache:
                draw_text_with_outline(frame, "Belum ada user. Tambahkan via users.json", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            if detect_state["stage"] == "idle":
                if current_time - detect_state["last_detection_ts"] < DETECTION_INTERVAL_SECONDS:
                    draw_text_with_outline(frame, "Standby...", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                else:
                    detect_state["last_detection_ts"] = current_time
                    detect_state["stage"] = "gesture_1"
                    detect_state["stable_letter"] = None
                    detect_state["stable_count"] = 0
                    detect_state["validation_pending"] = False
                    detect_state["candidate_letter"] = None
                    voice.interrupt()
                    voice.speak("Silakan tampilkan gesture pertama.")

            detected_letter = "-"
            gesture_confidence = 0.0
            hand_present = False
            if detect_state["stage"] in ("gesture_1", "gesture_2"):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_res = hands.process(rgb_frame)
                if hand_res.multi_hand_landmarks:
                    hand_present = True
                    hlm = hand_res.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

                    feat = []
                    for lm in hlm.landmark:
                        feat.extend([lm.x, lm.y, lm.z])

                    if sum(feat) != 0:
                        gesture_seq_buffer.append(feat)

                    if len(gesture_seq_buffer) == NO_OF_TIMESTEPS:
                        if detect_state.get("frame_index", 0) % GESTURE_DETECT_EVERY_N == 0:
                            X = np.array(gesture_seq_buffer).reshape(1, NO_OF_TIMESTEPS, -1)
                            prob = gesture_model.predict(X, verbose=0)[0]
                            cls = int(np.argmax(prob))
                            gesture_confidence = float(np.max(prob))
                            if gesture_confidence >= GESTURE_CONFIDENCE:
                                detected_letter = gesture_labels.get(cls, "Unknown")
                                detected_letter = normalize_gesture_label(detected_letter)
                                if not detected_letter:
                                    detected_letter = "-"
                    detect_state["frame_index"] = detect_state.get("frame_index", 0) + 1
                else:
                    gesture_seq_buffer.clear()

            if detect_state["stage"] in ("gesture_1", "gesture_2"):
                step_num = 1 if detect_state["stage"] == "gesture_1" else 2
                draw_text_with_outline(frame, f"STEP {step_num}/3", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                draw_text_with_outline(frame, f"Gesture terbaca: {detected_letter}", (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                if not hand_present:
                    draw_text_with_outline(frame, "Tangan belum terdeteksi", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                    update_gesture_state("-", hand_present, current_time)
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    continue

                if current_time < detect_state.get("gesture_cooldown_until", 0.0):
                    draw_text_with_outline(frame, "Cooldown...", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                else:
                    letter_for_stable = detected_letter if gesture_confidence >= GESTURE_CONFIDENCE else "-"
                    update_gesture_state(letter_for_stable, hand_present, current_time)
                    draw_text_with_outline(frame, f"Stabil: {detect_state['stable_count']}/{GESTURE_STABLE_FRAMES}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                    if detect_state["stable_count"] >= GESTURE_STABLE_FRAMES and not detect_state["validation_pending"] and detect_state["stable_letter"]:
                        detect_state["validation_pending"] = True
                        detect_state["validation_due"] = current_time + GESTURE_VALIDATE_DELAY_SECONDS
                        detect_state["candidate_letter"] = detect_state["stable_letter"]

                if detect_state["validation_pending"]:
                    remaining = max(0.0, detect_state["validation_due"] - current_time)
                    draw_text_with_outline(frame, f"Validasi: {remaining:.1f}s", (30, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    if remaining <= 0:
                        candidate = detect_state["candidate_letter"]
                        candidate = normalize_gesture_label(candidate)
                        detect_state["validation_pending"] = False
                        detect_state["candidate_letter"] = None
                        detect_state["stable_letter"] = None
                        detect_state["stable_count"] = 0
                        gesture_seq_buffer.clear()

                        if detect_state["stage"] == "gesture_1":
                            matched = [u for u in users_cache if u["gesture_1"] == candidate]
                            if not matched:
                                draw_animated_popup(frame, "GESTURE 1 GAGAL", "Gesture pertama tidak cocok", 0.0, "failed")
                                voice.interrupt()
                                voice.speak("Gesture pertama tidak valid. Ulangi gesture pertama.")
                                detect_state["gesture_cooldown_until"] = current_time + GESTURE_COOLDOWN_SECONDS
                                continue

                            detect_state["gesture_1"] = candidate
                            detect_state["candidate_users"] = matched
                            detect_state["stage"] = "gesture_2"
                            detect_state["gesture_cooldown_until"] = current_time + GESTURE_COOLDOWN_SECONDS
                            voice.interrupt()
                            voice.speak("Gesture pertama valid.")
                            voice.speak("Silakan tampilkan gesture kedua.")
                        else:
                            matched = [u for u in detect_state["candidate_users"] if u["gesture_2"] == candidate]
                            if not matched:
                                draw_animated_popup(frame, "GESTURE 2 GAGAL", "Gesture kedua tidak cocok", 0.0, "failed")
                                voice.interrupt()
                                voice.speak("Gesture kedua tidak valid. Ulangi gesture kedua.")
                                detect_state["gesture_cooldown_until"] = current_time + GESTURE_COOLDOWN_SECONDS
                                continue

                            detect_state["candidate_users"] = matched
                            detect_state["stage"] = "face"
                            detect_state["face_samples"] = []
                            detect_state["last_face_ts"] = 0.0
                            detect_state["face_start_ts"] = current_time
                            voice.interrupt()
                            voice.speak("Gesture kedua valid.")
                            voice.speak("Arahkan wajah ke kamera.")

            if detect_state["stage"] == "face":
                draw_text_with_outline(frame, "STEP 3/3", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                if current_time - detect_state["face_start_ts"] > FACE_LOCK_SECONDS:
                    draw_animated_popup(frame, "WAJAH GAGAL", "Waktu habis", 0.0, "failed")
                    voice.interrupt()
                    voice.speak("Akses ditolak.")
                    detect_state["stage"] = "cooldown"
                    detect_state["cooldown_until"] = current_time + 2.0
                elif current_time - detect_state["last_face_ts"] >= FACE_DETECT_INTERVAL_SECONDS:
                    detect_state["last_face_ts"] = current_time
                    encodings = encode_faces(frame)
                    if encodings:
                        detect_state["face_samples"].append(encodings[0])

                draw_text_with_outline(frame, f"Sampel wajah: {len(detect_state['face_samples'])}/{FACE_SAMPLE_TARGET}", (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                if len(detect_state["face_samples"]) >= FACE_SAMPLE_TARGET:
                    face_encoding = np.mean(np.stack(detect_state["face_samples"]), axis=0)
                    candidates = detect_state["candidate_users"]
                    encs = [u["face_encoding"] for u in candidates]
                    distances = face_recognition.face_distance(encs, face_encoding)
                    best_idx = int(np.argmin(distances))
                    best_distance = distances[best_idx]
                    matched_user = None
                    if best_distance < FACE_MATCH_TOLERANCE:
                        matched_user = candidates[best_idx]

                    if matched_user:
                        username = matched_user["username"]
                        screenshot_path = take_screenshot(frame, username)
                        record_access(username, detect_state["gesture_1"], matched_user["gesture_2"], "GRANTED", screenshot_path)
                        voice.interrupt()
                        voice.speak("Akses diterima.")
                        solenoid.unlock()
                        detect_state["result_title"] = "AKSES DIIZINKAN"
                        detect_state["result_subtitle"] = f"Selamat datang, {username}"
                        detect_state["result_status"] = "success"
                    else:
                        screenshot_path = take_screenshot(frame, "unknown")
                        record_access("unknown", detect_state["gesture_1"], "-", "DENIED", screenshot_path)
                        voice.interrupt()
                        voice.speak("Akses ditolak.")
                        detect_state["result_title"] = "AKSES DITOLAK"
                        detect_state["result_subtitle"] = "Wajah tidak cocok"
                        detect_state["result_status"] = "failed"

                    detect_state["stage"] = "cooldown"
                    detect_state["result_until"] = current_time + RESULT_POPUP_SECONDS
                    detect_state["cooldown_until"] = current_time + RESULT_POPUP_SECONDS

            if detect_state["stage"] == "cooldown":
                if detect_state.get("result_title") and current_time < detect_state.get("result_until", 0.0):
                    draw_animated_popup(
                        frame,
                        detect_state["result_title"],
                        detect_state["result_subtitle"],
                        0.0,
                        detect_state["result_status"],
                    )
                if current_time >= detect_state["cooldown_until"]:
                    reset_detect_state()

            draw_status(frame)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    voice.stop()
    solenoid.cleanup()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
