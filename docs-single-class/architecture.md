# معماری نسخه دمو

```text
Camera2 rear-camera frames
          ↓
YUV frame preprocessing
          ↓
Letterbox to 320 × 320
          ↓
FP32 TensorFlow Lite inference
          ↓
Confidence filtering and NMS
          ↓
Temporal stabilization
          ↓
Bounding boxes on live preview
          ↓
Local session report
```

تمام مراحل تحلیل مدل در نسخه فعلی روی دستگاه Android انجام می‌شوند.
