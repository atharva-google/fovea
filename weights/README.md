# weights/

Model weights live here. They are **not** committed to git — they're downloaded
automatically on first run, or up-front with:

```bash
python -m fovea download        # core models
python -m fovea download --all  # + whisper/demucs for --transcribe
python -m fovea weights         # show what's present
```

Expected layout:

```
weights/
├── yolo26s.pt                      # Ultralytics (auto-download from GitHub releases)
├── yolo26s-pose.pt                 # Ultralytics pose model (only for --face-backend pose)
├── blaze_face_full_range.tflite    # MediaPipe (Google Cloud Storage)
├── pretrain_TalkSet.model          # TalkNet-ASD (Google Drive via gdown)
├── easyocr/                        # EasyOCR detector + recognizer (auto)
└── whisper-medium/                 # faster-whisper (Hugging Face, optional)
```

If you already have these files, just copy them in with the names above.
Use `FOVEA_WEIGHTS_DIR=/some/path` to keep them elsewhere (e.g. Google Drive on Colab).
