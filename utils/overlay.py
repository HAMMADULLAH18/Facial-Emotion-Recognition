import cv2

EMOTION_COLORS = {
    'happy':    (50,  205, 50),
    'sad':      (220, 100, 30),
    'angry':    (30,  30,  220),
    'neutral':  (180, 180, 180),
    'surprise': (30,  200, 220),
    'fear':     (180, 60,  220),
    'disgust':  (60,  180, 30),
}

def draw_face(frame, box, emotion, confidence, all_emotions):
    x, y, w, h = box
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))

    fh, fw = frame.shape[:2]
    x, y   = max(0, x), max(0, y)
    w, h   = min(w, fw-x), min(h, fh-y)

    # Bounding box
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Label
    label = f"{emotion.upper()}  {confidence*100:.1f}%"
    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y-th-12), (x+tw+8, y), color, -1)
    cv2.putText(frame, label, (x+4, y-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    # Confidence bar
    bar_w = int(w * confidence)
    cv2.rectangle(frame, (x, y+h+4), (x+w,     y+h+14), (50,50,50), -1)
    cv2.rectangle(frame, (x, y+h+4), (x+bar_w, y+h+14), color,      -1)

    # All 7 emotion mini bars
    bar_x = x + w + 12
    for i, (emo, score) in enumerate(all_emotions.items()):
        bc   = EMOTION_COLORS.get(emo, (200, 200, 200))
        blen = int(score * 90)
        by   = y + i * 20
        cv2.rectangle(frame, (bar_x, by), (bar_x+blen, by+13), bc, -1)
        cv2.putText(frame, emo[:3], (bar_x-32, by+11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, bc, 1)
        cv2.putText(frame, f"{score*100:.0f}%", (bar_x+blen+4, by+11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, bc, 1)

def draw_hud(frame, fps, frame_count, counts):
    h, w = frame.shape[:2]
    fps_color = (0,255,0) if fps > 15 else (0,215,255) if fps > 8 else (0,0,255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
    cv2.putText(frame, f"Frames: {frame_count}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    if counts:
        top   = max(counts, key=counts.get)
        total = sum(counts.values())
        pct   = counts[top] / total * 100
        color = EMOTION_COLORS.get(top, (255,255,255))
        cv2.putText(frame, f"Session: {top.upper()} ({pct:.0f}%)",
                    (10, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)