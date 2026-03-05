import csv
import json
from datetime import datetime
from collections import defaultdict
import os

os.makedirs("logs", exist_ok=True)

class EmotionLogger:
    def __init__(self):
        self.rows   = []
        self.counts = defaultdict(int)

    def log(self, emotion, confidence, all_scores):
        self.counts[emotion] += 1
        self.rows.append({
            'timestamp':  datetime.now().isoformat(),
            'emotion':    emotion,
            'confidence': round(confidence, 4),
            **{k: round(float(v), 4) for k, v in all_scores.items()}
        })

    def save(self):
        if not self.rows:
            print("No data to save")
            return

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"logs/emotion_log_{ts}.csv"

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)

        total = sum(self.counts.values())
        summary = {
            k: {'count': v, 'pct': round(v/total*100, 1)}
            for k, v in self.counts.items()
        }
        with open('logs/session_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Saved: {path}")
        print(f"Total frames: {total}")
        print(f"Summary: {summary}")