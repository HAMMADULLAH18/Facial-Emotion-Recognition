import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
import csv
from datetime import datetime
from collections import defaultdict, deque
from fer import FER

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="FER — Facial Emotion Recognition",
    page_icon="😄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&display=swap');

  .main { background: #0a0a0f; }
  .block-container { padding-top: 1.5rem; }

  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #7c3aed, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }
  .hero-sub {
    color: #6b6b85;
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
  }
  .emotion-card {
    background: #111118;
    border: 1px solid #2a2a3a;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    margin: 0.3rem 0;
  }
  .emotion-label {
    font-size: 1.5rem;
    font-weight: 800;
    font-family: 'Syne', sans-serif;
  }
  .metric-card {
    background: #111118;
    border: 1px solid #2a2a3a;
    border-radius: 8px;
    padding: 0.8rem 1rem;
  }
  .stButton > button {
    background: linear-gradient(135deg, #7c3aed, #06b6d4);
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    width: 100%;
  }
  .stButton > button:hover {
    opacity: 0.85;
  }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
EMOTION_COLORS = {
    'happy':    '#22c55e',
    'sad':      '#3b82f6',
    'angry':    '#ef4444',
    'neutral':  '#94a3b8',
    'surprise': '#f59e0b',
    'fear':     '#a855f7',
    'disgust':  '#84cc16'
}
EMOTION_EMOJI = {
    'happy': '😄', 'sad': '😢', 'angry': '😠',
    'neutral': '😐', 'surprise': '😲',
    'fear': '😨', 'disgust': '🤢'
}
DETECT_EVERY_N = 2
SMOOTH_WINDOW  = 5
FRAME_SCALE    = 0.6

os.makedirs("logs", exist_ok=True)

# ── Session state ─────────────────────────────────────────────
if 'running'       not in st.session_state: st.session_state.running       = False
if 'log_rows'      not in st.session_state: st.session_state.log_rows      = []
if 'counts'        not in st.session_state: st.session_state.counts        = defaultdict(int)
if 'frame_count'   not in st.session_state: st.session_state.frame_count   = 0
if 'smooth_buffer' not in st.session_state: st.session_state.smooth_buffer = deque(maxlen=SMOOTH_WINDOW)
if 'last_results'  not in st.session_state: st.session_state.last_results  = []
if 'fps_history'   not in st.session_state: st.session_state.fps_history   = []

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    detect_every = st.slider("Detect every N frames", 1, 5, 2,
        help="Higher = faster FPS, lower = more responsive")
    smooth_win   = st.slider("Smoothing window", 1, 10, 5,
        help="Frames to average predictions over")
    frame_scale  = st.slider("Frame scale for detection", 0.3, 1.0, 0.6,
        help="Smaller = faster, larger = more accurate")

    st.markdown("---")
    st.markdown("## 🧬 Psychology Notes")
    st.markdown("""
    The **7 emotions** detected here are Ekman's
    *universal emotions* — biologically hardwired
    expressions recognized across all human cultures.

    - 😄 **Happy** — dopamine, reward
    - 😢 **Sad** — loss, empathy signal
    - 😠 **Angry** — threat response
    - 😨 **Fear** — amygdala activation
    - 😲 **Surprise** — attention reset
    - 🤢 **Disgust** — avoidance response
    - 😐 **Neutral** — cognitive baseline

    > Expressions occur in under **200ms** —
    > faster than conscious thought.
    """)

    st.markdown("---")
    st.markdown("## 📁 Session Logs")
    log_files = sorted(os.listdir("logs")) if os.path.exists("logs") else []
    if log_files:
        for f in log_files[-3:]:
            st.markdown(f"📄 `{f}`")
    else:
        st.markdown("*No logs yet*")

# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Facial Emotion Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Real-Time AI · Computer Vision · Human Psychology</div>', unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────
col_cam, col_stats = st.columns([3, 2], gap="large")

with col_cam:
    st.markdown("### 📷 Live Feed")
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        start_btn = st.button("▶ Start", key="start")
    with btn_col2:
        stop_btn  = st.button("⏹ Stop",  key="stop")
    with btn_col3:
        save_btn  = st.button("💾 Save Log", key="save")

with col_stats:
    st.markdown("### 📊 Live Stats")
    emotion_placeholder  = st.empty()
    bars_placeholder     = st.empty()
    session_placeholder  = st.empty()

# ── Button handlers ───────────────────────────────────────────
if start_btn:
    st.session_state.running       = True
    st.session_state.log_rows      = []
    st.session_state.counts        = defaultdict(int)
    st.session_state.frame_count   = 0
    st.session_state.smooth_buffer = deque(maxlen=smooth_win)
    st.session_state.last_results  = []
    st.session_state.fps_history   = []

if stop_btn:
    st.session_state.running = False

if save_btn and st.session_state.log_rows:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"logs/emotion_log_{ts}.csv"
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=st.session_state.log_rows[0].keys())
        writer.writeheader()
        writer.writerows(st.session_state.log_rows)
    st.success(f"Saved: {path}")

# ── Drawing helper ────────────────────────────────────────────
def draw_overlay(frame, box, emotion, confidence, all_emotions):
    x, y, w, h = box
    fh, fw = frame.shape[:2]
    x, y   = max(0,x), max(0,y)
    w, h   = min(w, fw-x), min(h, fh-y)

    # Color from hex to BGR
    hex_col = EMOTION_COLORS.get(emotion, '#ffffff').lstrip('#')
    r,g,b   = tuple(int(hex_col[i:i+2],16) for i in (0,2,4))
    color   = (b, g, r)

    cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
    label = f"{emotion.upper()} {confidence*100:.0f}%"
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y-th-12), (x+tw+8,y), color, -1)
    cv2.putText(frame, label, (x+4,y-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    bar_w = int(w*confidence)
    cv2.rectangle(frame,(x,y+h+4),(x+w,y+h+12),(40,40,40),-1)
    cv2.rectangle(frame,(x,y+h+4),(x+bar_w,y+h+12),color,-1)

# ── Main loop ─────────────────────────────────────────────────
if st.session_state.running:
    detector = FER(mtcnn=False)
    cap      = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    status_placeholder.info("🎥 Camera active — click Stop to end session")
    prev_time = time.time()

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            status_placeholder.error("❌ Camera not found")
            break

        st.session_state.frame_count += 1
        fc = st.session_state.frame_count

        # Detection with frame skipping
        if fc % detect_every == 0:
            small = cv2.resize(frame, (0,0), fx=frame_scale, fy=frame_scale)
            raw   = detector.detect_emotions(small)
            st.session_state.last_results = []
            for face in raw:
                x,y,w,h = face['box']
                st.session_state.last_results.append({
                    'box': [int(x/frame_scale), int(y/frame_scale),
                            int(w/frame_scale), int(h/frame_scale)],
                    'emotions': face['emotions']
                })

        current_emotion    = None
        current_confidence = 0
        current_all        = {}

        for face in st.session_state.last_results:
            emotions = face['emotions']
            st.session_state.smooth_buffer.append(emotions)
            avg = {e: float(np.mean([f[e] for f in st.session_state.smooth_buffer]))
                   for e in emotions}

            emotion    = max(avg, key=avg.get)
            confidence = avg[emotion]

            current_emotion    = emotion
            current_confidence = confidence
            current_all        = avg

            draw_overlay(frame, face['box'], emotion, confidence, avg)
            st.session_state.counts[emotion] += 1
            st.session_state.log_rows.append({
                'timestamp':  datetime.now().isoformat(),
                'emotion':    emotion,
                'confidence': round(confidence, 4),
                **{k: round(v,4) for k,v in avg.items()}
            })

        # FPS
        now       = time.time()
        fps       = 1.0 / (now - prev_time + 1e-6)
        prev_time = now
        st.session_state.fps_history.append(fps)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Frames: {fc}", (10,52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

        # Show frame (RGB for Streamlit)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Live stats panel
        with emotion_placeholder.container():
            if current_emotion:
                emoji = EMOTION_EMOJI.get(current_emotion, '')
                color = EMOTION_COLORS.get(current_emotion, '#fff')
                st.markdown(f"""
                <div class="emotion-card">
                  <div style="font-size:3rem">{emoji}</div>
                  <div class="emotion-label" style="color:{color}">
                    {current_emotion.upper()}
                  </div>
                  <div style="color:#888;font-size:0.8rem;margin-top:0.3rem">
                    Confidence: {current_confidence*100:.1f}%
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with bars_placeholder.container():
            if current_all:
                st.markdown("**All Emotion Scores**")
                for emo, score in sorted(current_all.items(),
                                         key=lambda x: -x[1]):
                    emoji = EMOTION_EMOJI.get(emo, '')
                    st.progress(float(score),
                                text=f"{emoji} {emo}: {score*100:.1f}%")

        with session_placeholder.container():
            counts = st.session_state.counts
            if counts:
                total = sum(counts.values())
                top   = max(counts, key=counts.get)
                st.markdown("**Session Summary**")
                avg_fps = np.mean(st.session_state.fps_history[-30:]) \
                          if st.session_state.fps_history else 0
                c1, c2, c3 = st.columns(3)
                c1.metric("Frames", fc)
                c2.metric("FPS", f"{avg_fps:.1f}")
                c3.metric("Dominant", EMOTION_EMOJI.get(top,'') + top[:3])
                df_counts = pd.DataFrame({
                    'Emotion': list(counts.keys()),
                    'Frames':  list(counts.values())
                }).sort_values('Frames', ascending=False)
                st.bar_chart(df_counts.set_index('Emotion'))

    cap.release()
    status_placeholder.success(
        f"Session ended — {st.session_state.frame_count} frames processed")

else:
    # Idle state
    frame_placeholder.markdown("""
    <div style="background:#111118;border:1px solid #2a2a3a;border-radius:8px;
                padding:3rem;text-align:center;color:#4a4a6a;min-height:300px;
                display:flex;flex-direction:column;align-items:center;
                justify-content:center">
      <div style="font-size:3rem;margin-bottom:1rem">📷</div>
      <div style="font-size:1rem;color:#6b6b85">
        Click <strong style="color:#7c3aed">▶ Start</strong> to begin
      </div>
      <div style="font-size:0.75rem;margin-top:0.5rem;color:#4a4a6a">
        Make sure your webcam is connected
      </div>
    </div>
    """, unsafe_allow_html=True)

    with emotion_placeholder.container():
        st.markdown("""
        <div class="emotion-card">
          <div style="font-size:2.5rem">🎭</div>
          <div style="color:#4a4a6a;font-size:0.85rem;margin-top:0.5rem">
            Waiting for camera...
          </div>
        </div>
        """, unsafe_allow_html=True)
