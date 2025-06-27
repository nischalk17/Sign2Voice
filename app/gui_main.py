import cv2, json, time, numpy as np, tensorflow as tf
import mediapipe as mp
from collections import deque, Counter
import tkinter as tk
from PIL import Image, ImageTk
from app.tts import speak

# ── Load trained model & class labels ────────────────────────────────
model = tf.keras.models.load_model("models/landmark_cnn.h5")
class_names = json.load(open("models/landmark_classes.json"))

# ── MediaPipe Hands setup (keyword args avoid type error) ───────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
mp_draw = mp.solutions.drawing_utils

# ── Tkinter GUI  ─────────────────────────────────────────────────────
root = tk.Tk()
root.title("Sign2Voice")

# Video panel
video_panel = tk.Label(root)
video_panel.grid(row=0, column=0, columnspan=3)

# Sentence + current sign variables
current_var = tk.StringVar(value="Current: _")
sentence_var = tk.StringVar(value="Sentence:")
tk.Label(root, textvariable=current_var, font=("Arial", 14)
         ).grid(row=1, column=0, sticky="w")
tk.Label(root, textvariable=sentence_var, font=("Arial", 14)
         ).grid(row=2, column=0, columnspan=3, sticky="w")

# ── Helper callbacks BEFORE buttons ─────────────────────────────────
sentence = ""

def clear_sentence():
    global sentence
    sentence = ""
    sentence_var.set("Sentence:")

def speak_sentence():
    speak(sentence)

# ── Buttons with color + icons ──────────────────────────────────────
BTN_STYLE = {"font": ("Arial", 12, "bold"), "width": 12, "relief": "raised"}

clear_btn = tk.Button(root, text="🗑️  Clear", bg="#ffb3b3", fg="black",
                      command=clear_sentence, **BTN_STYLE)
speak_btn = tk.Button(root, text="🗣 Speak", bg="#b3e6ff", fg="black",
                      command=speak_sentence, **BTN_STYLE)
exit_btn  = tk.Button(root, text="❌ Exit",  bg="#ff6666", fg="white",
                      command=root.destroy, **BTN_STYLE)

clear_btn.grid(row=1, column=1, padx=5, pady=5)
speak_btn.grid(row=1, column=2, padx=5, pady=5)
exit_btn.grid(row=2, column=2, padx=5, pady=5, sticky="e")

# ── Webcam & prediction buffers ─────────────────────────────────────
cap = cv2.VideoCapture(0)
pred_buffer = deque(maxlen=10)
buffer_vote  = 6
last_added   = ""

def update_frame():
    global sentence, last_added
    ok, frame = cap.read()
    if not ok:
        root.after(10, update_frame)
        return

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        lm_vec = [c for p in hand.landmark for c in (p.x, p.y, p.z)]
        preds  = model.predict(np.expand_dims(lm_vec, 0), verbose=0)[0]
        conf   = preds.max()
        letter = class_names[int(preds.argmax())]

        if conf >= 0.8:
            pred_buffer.append(letter)
            vote_letter, vote_count = Counter(pred_buffer).most_common(1)[0]
            if vote_count >= buffer_vote and vote_letter != last_added:
                if vote_letter == "space":
                    sentence += " "
                elif vote_letter == "del":
                    sentence = sentence[:-1]
                else:
                    sentence += vote_letter
                last_added = vote_letter
                pred_buffer.clear()

            current_var.set(f"Current: {letter} ({conf:.2f})")

        # Draw landmarks & overlay letter on frame
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, f"{letter} ({conf:.2f})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 255, 0), 3)
    else:
        pred_buffer.clear()
        last_added = ""
        current_var.set("Current: _")

    sentence_var.set(f"Sentence: {sentence}")

    # Convert frame to Tkinter image
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk   = ImageTk.PhotoImage(image=img_pil)
    video_panel.imgtk = imgtk
    video_panel.configure(image=imgtk)

    root.after(10, update_frame)

# Start the loop
update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
