import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
import csv
from datetime import datetime
from collections import defaultdict, deque
from PIL import Image
from fer.fer import FER

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="FER — Facial Emotion Recognition",
    page_icon="😄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&display=swap');
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #7c3aed, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }
  .hero-sub { color: #6b6b85; font-size: 0.85rem; margin-bottom: 1rem; }
  .emotion-card {
    background: #111118; border: 1px solid #2a2a3a;
    border-radius: 8px; padding: 1.2rem; text-align: center;
  }
  .mode-badge {
    display: inline-block; padding: 0.3rem 1rem;
    border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    margin-bottom: 1rem;
  }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
EMOTION_COLORS = {
    'happy':    '#22c55e', 'sad':      '#3b82f6',
    'angry':    '#ef4444', 'neutral':  '#94a3b8',
    'surprise': '#f59e0b', 'fear':     '#a855f7',
    'disgust':  '#84cc16'
}
EMOTION_EMOJI = {
    'happy': '😄', 'sad': '😢', 'angry': '😠',
    'neutral': '😐', 'surprise': '😲',
    'fear': '😨', 'disgust': '🤢'
}

os.makedirs("logs", exist_ok=True)

@st.cache_resource
def load_detector():
    return FER(mtcnn=False)

detector = load_detector()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Mode")
    mode = st.radio("Select Mode",
                    ["Photo Upload", "Live Webcam"],
                    help="Photo Upload works on Streamlit Cloud. Webcam is local only.")

    st.markdown("---")
    st.markdown("## Settings")
    smooth_win   = st.slider("Smoothing window",      1, 10, 5)
    detect_every = st.slider("Detect every N frames", 1,  5, 2)
    frame_scale  = st.slider("Frame scale",         0.3, 1.0, 0.5)

    st.markdown("---")
    st.markdown("## Psychology Notes")
    st.markdown("""
    **7 Universal Emotions** — Ekman's biologically
    hardwired expressions, identical across all cultures.

    - 😄 **Happy** — dopamine, reward
    - 😢 **Sad** — loss processing
    - 😠 **Angry** — threat response
    - 😨 **Fear** — amygdala activation
    - 😲 **Surprise** — attention reset
    - 🤢 **Disgust** — avoidance response
    - 😐 **Neutral** — cognitive baseline

    > Expressions fire in under **200ms** —
    > faster than conscious thought.
    """)

# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Facial Emotion Recognition</div>',
            unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Real-Time AI · Computer Vision · Human Psychology</div>',
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# MODE A — PHOTO UPLOAD (Streamlit Cloud compatible)
# ═══════════════════════════════════════════════════
if mode == "Photo Upload":

    st.markdown("""
    <div class="mode-badge"
         style="background:rgba(6,182,212,0.15);color:#06b6d4;
                border:1px solid rgba(6,182,212,0.3)">
      Photo Mode — works on Streamlit Cloud
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### Upload or Capture")
        uploaded  = st.file_uploader("Upload a face photo",
                                     type=["jpg","jpeg","png","webp"])
        cam_photo = st.camera_input("Or take a snapshot")

    with col2:
        st.markdown("### Results")
        result_slot = st.empty()
        bars_slot   = st.empty()

    img_source = cam_photo if cam_photo else uploaded

    if img_source is not None:
        pil_img = Image.open(img_source).convert("RGB")
        img_np  = np.array(pil_img)
        frame   = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing..."):
            results = detector.detect_emotions(frame)

        if results:
            for face in results:
                x,y,w,h    = face['box']
                emotions   = face['emotions']
                emotion    = max(emotions, key=emotions.get)
                confidence = emotions[emotion]
                hex_col    = EMOTION_COLORS.get(emotion,'#ffffff').lstrip('#')
                r,g,b      = tuple(int(hex_col[i:i+2],16) for i in (0,2,4))
                color      = (b,g,r)
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,3)
                label      = f"{emotion.upper()} {confidence*100:.0f}%"
                (tw,th),_  = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame,(x,y-th-14),(x+tw+8,y),color,-1)
                cv2.putText(frame,label,(x+4,y-7),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)

            st.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),
                     caption="Detected emotions", use_column_width=True)

            face     = results[0]
            emotions = face['emotions']
            emotion  = max(emotions, key=emotions.get)
            conf     = emotions[emotion]
            emoji    = EMOTION_EMOJI.get(emotion,'')
            color    = EMOTION_COLORS.get(emotion,'#fff')

            with result_slot.container():
                st.markdown(f"""
                <div class="emotion-card">
                  <div style="font-size:3.5rem">{emoji}</div>
                  <div style="font-family:'Syne',sans-serif;font-size:2rem;
                              font-weight:800;color:{color}">
                    {emotion.upper()}
                  </div>
                  <div style="color:#888;font-size:0.85rem;margin-top:0.4rem">
                    Confidence: {conf*100:.1f}%
                  </div>
                </div>
                """, unsafe_allow_html=True)

            with bars_slot.container():
                st.markdown("**All Emotion Scores**")
                for emo,score in sorted(emotions.items(),
                                         key=lambda x:-x[1]):
                    em = EMOTION_EMOJI.get(emo,'')
                    st.progress(float(score),
                                text=f"{em} {emo}: {score*100:.1f}%")

            if len(results) > 1:
                st.markdown(f"**{len(results)} faces detected**")
                cols = st.columns(len(results))
                for i,(face,col) in enumerate(zip(results,cols)):
                    emo  = max(face['emotions'],key=face['emotions'].get)
                    conf = face['emotions'][emo]
                    col.metric(f"Face {i+1}",
                               f"{EMOTION_EMOJI.get(emo,'')} {emo}",
                               f"{conf*100:.0f}%")
        else:
            st.warning("No face detected. Try a clearer front-facing photo.")
            st.image(pil_img, use_column_width=True)

# ═══════════════════════════════════════════════════
# MODE B — LIVE WEBCAM (local only)
# ═══════════════════════════════════════════════════
else:
    st.markdown("""
    <div class="mode-badge"
         style="background:rgba(124,58,237,0.15);color:#a78bfa;
                border:1px solid rgba(124,58,237,0.3)">
      Webcam Mode — local only: streamlit run app.py
    </div>
    """, unsafe_allow_html=True)

    for key,val in [('running',False),('log_rows',[]),
                    ('counts',defaultdict(int)),('frame_count',0),
                    ('smooth_buffer',deque(maxlen=smooth_win)),
                    ('last_results',[]),('fps_history',[])]:
        if key not in st.session_state:
            st.session_state[key] = val

    col_cam, col_stats = st.columns([3,2], gap="large")

    with col_cam:
        st.markdown("### Live Feed")
        frame_slot  = st.empty()
        status_slot = st.empty()
        b1,b2,b3    = st.columns(3)
        start_btn   = b1.button("Start")
        stop_btn    = b2.button("Stop")
        save_btn    = b3.button("Save Log")

    with col_stats:
        st.markdown("### Live Stats")
        emo_slot     = st.empty()
        bars_slot    = st.empty()
        session_slot = st.empty()

    if start_btn:
        st.session_state.update({
            'running': True, 'log_rows': [],
            'counts': defaultdict(int), 'frame_count': 0,
            'smooth_buffer': deque(maxlen=smooth_win),
            'last_results': [], 'fps_history': []
        })

    if stop_btn:
        st.session_state.running = False

    if save_btn and st.session_state.log_rows:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"logs/emotion_log_{ts}.csv"
        with open(path,'w',newline='') as f:
            writer = csv.DictWriter(
                f,fieldnames=st.session_state.log_rows[0].keys())
            writer.writeheader()
            writer.writerows(st.session_state.log_rows)
        st.success(f"Saved: {path}")

    def draw_box(frame, box, emotion, confidence):
        x,y,w,h   = box
        fh,fw     = frame.shape[:2]
        x,y       = max(0,x),max(0,y)
        w,h       = min(w,fw-x),min(h,fh-y)
        hex_col   = EMOTION_COLORS.get(emotion,'#ffffff').lstrip('#')
        r,g,b     = tuple(int(hex_col[i:i+2],16) for i in (0,2,4))
        color     = (b,g,r)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        label     = f"{emotion.upper()} {confidence*100:.0f}%"
        (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
        cv2.rectangle(frame,(x,y-th-12),(x+tw+8,y),color,-1)
        cv2.putText(frame,label,(x+4,y-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
        bw = int(w*confidence)
        cv2.rectangle(frame,(x,y+h+4),(x+w,y+h+12),(40,40,40),-1)
        cv2.rectangle(frame,(x,y+h+4),(x+bw,y+h+12),color,-1)

    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        status_slot.info("Camera active — click Stop to end")
        prev_time = time.time()

        while st.session_state.running:
            ret,frame = cap.read()
            if not ret:
                status_slot.error("Camera not found")
                break

            st.session_state.frame_count += 1
            fc = st.session_state.frame_count

            if fc % detect_every == 0:
                small = cv2.resize(frame,(0,0),
                                   fx=frame_scale,fy=frame_scale)
                raw   = detector.detect_emotions(small)
                st.session_state.last_results = [{
                    'box':[int(f['box'][0]/frame_scale),
                           int(f['box'][1]/frame_scale),
                           int(f['box'][2]/frame_scale),
                           int(f['box'][3]/frame_scale)],
                    'emotions':f['emotions']
                } for f in raw]

            cur_emo,cur_conf,cur_all = None,0,{}

            for face in st.session_state.last_results:
                emo_dict = face['emotions']
                st.session_state.smooth_buffer.append(emo_dict)
                avg  = {e:float(np.mean([f[e] for f in
                         st.session_state.smooth_buffer]))
                        for e in emo_dict}
                emo  = max(avg,key=avg.get)
                conf = avg[emo]
                cur_emo,cur_conf,cur_all = emo,conf,avg
                draw_box(frame,face['box'],emo,conf)
                st.session_state.counts[emo] += 1
                st.session_state.log_rows.append({
                    'timestamp': datetime.now().isoformat(),
                    'emotion': emo, 'confidence': round(conf,4),
                    **{k:round(v,4) for k,v in avg.items()}
                })

            now = time.time()
            fps = 1.0/(now-prev_time+1e-6)
            prev_time = now
            st.session_state.fps_history.append(fps)
            cv2.putText(frame,f"FPS: {fps:.1f}",(10,28),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

            frame_slot.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),
                             channels="RGB",use_column_width=True)

            if cur_emo:
                emoji = EMOTION_EMOJI.get(cur_emo,'')
                color = EMOTION_COLORS.get(cur_emo,'#fff')
                with emo_slot.container():
                    st.markdown(f"""
                    <div class="emotion-card">
                      <div style="font-size:3rem">{emoji}</div>
                      <div style="font-family:'Syne',sans-serif;
                                  font-size:1.8rem;font-weight:800;
                                  color:{color}">{cur_emo.upper()}</div>
                      <div style="color:#888;font-size:0.8rem">
                        Confidence: {cur_conf*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                with bars_slot.container():
                    st.markdown("**All Emotion Scores**")
                    for emo,score in sorted(cur_all.items(),
                                            key=lambda x:-x[1]):
                        st.progress(float(score),
                            text=f"{EMOTION_EMOJI.get(emo,'')} "
                                 f"{emo}: {score*100:.1f}%")

            counts = st.session_state.counts
            if counts:
                with session_slot.container():
                    total   = sum(counts.values())
                    top     = max(counts,key=counts.get)
                    avg_fps = np.mean(st.session_state.fps_history[-30:])
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Frames",fc)
                    c2.metric("FPS",f"{avg_fps:.1f}")
                    c3.metric("Mood",
                              EMOTION_EMOJI.get(top,'')+top[:3])
                    st.bar_chart(pd.DataFrame({
                        'Emotion':list(counts.keys()),
                        'Frames':list(counts.values())
                    }).sort_values('Frames',ascending=False)
                     .set_index('Emotion'))

        cap.release()
        status_slot.success(
            f"Done — {st.session_state.frame_count} frames")
    else:
        frame_slot.markdown("""
        <div style="background:#111118;border:1px solid #2a2a3a;
                    border-radius:8px;padding:4rem;text-align:center;
                    min-height:280px">
          <div style="font-size:3rem;margin-bottom:1rem">📷</div>
          <div style="color:#6b6b85">
            Click <strong style="color:#7c3aed">Start</strong> to begin
          </div>
        </div>""", unsafe_allow_html=True)
