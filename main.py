from fer import FER
import cv2
import time
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
from utils.overlay import draw_face, draw_hud
from utils.logger import EmotionLogger

# Settings
DETECT_EVERY_N_FRAMES = 2
SMOOTH_WINDOW         = 5
FRAME_SCALE           = 0.6

def main():
    detector = FER(mtcnn=False)
    cap      = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    logger        = EmotionLogger()
    counts        = defaultdict(int)
    smooth_buffer = deque(maxlen=SMOOTH_WINDOW)
    last_results  = []
    frame_count   = 0
    prev_time     = time.time()

    print("Starting FER... Press Q in popup window to stop")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            small       = cv2.resize(frame, (0,0),
                                     fx=FRAME_SCALE, fy=FRAME_SCALE)
            raw         = detector.detect_emotions(small)
            last_results = []
            for face in raw:
                x, y, w, h = face['box']
                last_results.append({
                    'box': [
                        int(x/FRAME_SCALE), int(y/FRAME_SCALE),
                        int(w/FRAME_SCALE), int(h/FRAME_SCALE)
                    ],
                    'emotions': face['emotions']
                })

        for face in last_results:
            emotions = face['emotions']
            smooth_buffer.append(emotions)
            avg = {e: float(np.mean([f[e] for f in smooth_buffer]))
                   for e in emotions}

            emotion    = max(avg, key=avg.get)
            confidence = avg[emotion]

            draw_face(frame, face['box'], emotion, confidence, avg)
            counts[emotion] += 1
            logger.log(emotion, confidence, avg)

        now       = time.time()
        fps       = 1.0 / (now - prev_time + 1e-6)
        prev_time = now
        draw_hud(frame, fps, frame_count, counts)

        cv2.imshow("FER Live | Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.save()
    print(f"Done — {frame_count} frames | {dict(counts)}")

if __name__ == "__main__":
    main()