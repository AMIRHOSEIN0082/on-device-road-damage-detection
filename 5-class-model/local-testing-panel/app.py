import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import io
import time
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

st.set_page_config(page_title="RoadScan RDD - Realtime Panel", layout="wide")

# ---------- Persian class names ----------
CLASS_NAMES_FA = {
    0: "ترک طولی",
    1: "ترک عرضی",
    2: "ترک شبکه‌ای",
    3: "وصله/ترمیم",
    4: "چاله",
}
CLASS_NAMES_EN = {
    0: "longitudinal_crack",
    1: "transverse_crack",
    2: "alligator_crack",
    3: "repair",
    4: "pothole",
}

CLASS_COLORS = {
    0: (255, 128, 0),
    1: (0, 200, 255),
    2: (0, 0, 255),
    3: (0, 255, 0),
    4: (255, 0, 255),
}

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")

# Longer side (px) a captured detection frame is downscaled to before it's held in memory for
# the PDF report. Frames are already annotated by the time they're captured, so this only trims
# resolution, never re-runs detection; without it, a handful of 4K positive frames could balloon
# to gigabytes of RAM over a long video.
PDF_FRAME_MAX_DIM = 1000
PDF_PAGE_SIZE = (1240, 1754)  # ~A4 at 150dpi
# Pillow embeds each PDF page as a JPEG (DCTDecode) and honors this quality on save() — dropping
# it from Pillow's PDF default of 75 to 60 measurably shrinks the file with no visible loss at
# the page sizes this report actually prints at.
PDF_JPEG_QUALITY = 60


def get_font(size, bold=False):
    # cv2.putText can't render Persian, and Pillow's bundled fonts have the same gap — so PDF
    # text (like the on-image labels) stays in English. matplotlib ships DejaVu Sans, which is
    # already installed as an ultralytics/plotting dependency, so it's used here instead of
    # Pillow's tiny built-in bitmap font.
    try:
        import matplotlib
        base = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return ImageFont.truetype(os.path.join(base, name), size)
    except Exception:
        return ImageFont.load_default()


def resize_for_pdf(frame_bgr):
    h, w = frame_bgr.shape[:2]
    scale = PDF_FRAME_MAX_DIM / max(h, w)
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame_bgr


def find_weight_files():
    weights = {}
    if os.path.isdir(WEIGHTS_DIR):
        for f in sorted(os.listdir(WEIGHTS_DIR)):
            if f.endswith(".pt") or f.endswith(".tflite"):
                weights[f] = os.path.join(WEIGHTS_DIR, f)
    return weights


def open_video_writer(path, fps, width, height):
    """mp4v (MPEG-4 Part 2) is what cv2.VideoWriter_fourcc examples default to, but it's a far
    less efficient codec than H.264 — roughly 5x larger for the same visual quality in testing
    on this machine. avc1 asks OpenCV's ffmpeg backend for real H.264, with mp4v as a fallback
    for a build/platform where that codec isn't available.
    """
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    if writer.isOpened():
        return writer
    writer.release()
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


def letterbox_to_square(frame_bgr, size):
    """Resizes+pads frame_bgr into a size x size canvas preserving aspect ratio, matching the
    Android app's LetterboxTransform (pad color 114, the standard YOLO gray). Returns the
    canvas plus the scale/pad needed to map model-space boxes back to the original frame.
    """
    h, w = frame_bgr.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


class TFLiteYoloModel:
    """Runs the exact .tflite file shipped in the Android app (roadscan_rdd_5class_fp32_320.tflite)
    through the same decode this panel uses to sanity-check the .pt weights, so results here are
    directly comparable to what the phone actually does — not Ultralytics' own postprocessing.

    Mirrors ir.roadscan.nativeapp.DetectionMath.decodeAndNms(): output is [1, 9, anchors], channels
    0-3 are box center/size already in the model's own input-pixel space (not normalized 0-1),
    channels 4-8 are five raw per-class scores (no separate objectness channel) — the highest one
    wins the anchor's class. NMS is class-aware (only suppresses within the same class).
    """

    NUM_CLASSES = 5

    def __init__(self, path):
        from ai_edge_litert.interpreter import Interpreter
        self.interpreter = Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        in_detail = self.interpreter.get_input_details()[0]
        out_detail = self.interpreter.get_output_details()[0]
        self.input_index = in_detail["index"]
        self.output_index = out_detail["index"]
        _, _, self.input_h, self.input_w = in_detail["shape"]  # NCHW

    def infer(self, frame_bgr, conf_thres, iou_thres, max_detections=100):
        h, w = frame_bgr.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox_to_square(frame_bgr, self.input_w)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)
        self.interpreter.set_tensor(self.input_index, chw)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_index)[0]  # [9, anchors]

        boxes_xy = output[0:4]
        class_scores = output[4:4 + self.NUM_CLASSES]
        best_cls = np.argmax(class_scores, axis=0)
        best_score = class_scores[best_cls, np.arange(class_scores.shape[1])]

        candidates = []
        for i in range(output.shape[1]):
            conf = float(best_score[i])
            if conf < conf_thres:
                continue
            cx, cy, bw, bh = boxes_xy[:, i]
            if bw <= 0 or bh <= 0:
                continue
            left = (cx - bw / 2 - pad_x) / scale
            top = (cy - bh / 2 - pad_y) / scale
            right = (cx + bw / 2 - pad_x) / scale
            bottom = (cy + bh / 2 - pad_y) / scale
            left, right = max(0.0, min(left, w)), max(0.0, min(right, w))
            top, bottom = max(0.0, min(top, h)), max(0.0, min(bottom, h))
            if right <= left or bottom <= top:
                continue
            candidates.append((int(best_cls[i]), conf, (left, top, right, bottom)))

        return self._nms(candidates, iou_thres, max_detections)

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _nms(self, candidates, iou_thres, max_detections):
        kept = []
        for cls_id in {c[0] for c in candidates}:
            group_kept = []
            for cand in sorted((c for c in candidates if c[0] == cls_id), key=lambda c: -c[1]):
                if all(self._iou(cand[2], k[2]) <= iou_thres for k in group_kept):
                    group_kept.append(cand)
            kept.extend(group_kept)
        kept.sort(key=lambda c: -c[1])
        return kept[:max_detections]


@st.cache_resource(show_spinner=False)
def load_model(path):
    if path.endswith(".tflite"):
        return TFLiteYoloModel(path)
    return YOLO(path)


def infer_boxes(model, frame, conf_thres, iou_thres, imgsz):
    """Returns [(cls_id, confidence, (x1, y1, x2, y2)), ...] for one frame, regardless of
    whether `model` is an Ultralytics YOLO (.pt) or the TFLiteYoloModel (.tflite) wrapper above —
    isolates run_stream() from the two backends' very different native APIs.

    Dispatches on hasattr(model, "infer") rather than isinstance(model, TFLiteYoloModel):
    st.cache_resource keeps a model instance alive across Streamlit reruns, but each rerun
    re-executes this whole script and so redefines TFLiteYoloModel as a brand-new class object —
    isinstance against that fresh class then silently fails for an instance cached from an
    earlier rerun, which is exactly what caused this to fall through to the .pt/Ultralytics path
    and raise AttributeError: 'TFLiteYoloModel' object has no attribute 'predict'.
    """
    if hasattr(model, "infer"):
        return model.infer(frame, conf_thres, iou_thres)
    results = model.predict(frame, conf=conf_thres, iou=iou_thres, imgsz=imgsz, verbose=False)
    r = results[0]
    boxes = []
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            xyxy = tuple(box.xyxy[0].cpu().numpy().tolist())
            boxes.append((int(box.cls[0]), float(box.conf[0]), xyxy))
    return boxes


def draw_box(frame, xyxy, cls_id, conf):
    x1, y1, x2, y2 = map(int, xyxy)
    color = CLASS_COLORS.get(cls_id, (255, 255, 255))
    # cv2.putText cannot render Persian/Unicode glyphs (Hershey fonts are Latin-only),
    # so the on-image label must stay in English. Persian names are shown in the side tables.
    label = f"{CLASS_NAMES_EN.get(cls_id, str(cls_id))} {conf*100:.0f}%"

    # Scale font/line thickness to the actual frame resolution so labels stay legible
    # regardless of whether the source video is 480p or 4K.
    h, w = frame.shape[:2]
    scale_ref = max(w, h) / 1600.0
    font_scale = max(0.45, 0.6 * scale_ref)
    thickness = max(1, round(1.4 * scale_ref))
    box_thickness = max(1, round(1.6 * scale_ref))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pad = max(4, round(6 * scale_ref))
    ly1 = max(y1 - th - baseline - pad, 0)
    cv2.rectangle(frame, (x1, ly1), (x1 + tw + 2 * pad, ly1 + th + baseline + pad), color, -1)
    cv2.putText(frame, label, (x1 + pad, ly1 + th + pad // 2), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return frame


def render_counts_table(total_detections, max_per_frame):
    table_md = "| کلاس | تعداد کل تشخیص | حداکثر هم‌زمان در یک فریم |\n|---|---|---|\n"
    for cid in sorted(CLASS_NAMES_FA.keys()):
        table_md += (
            f"| {CLASS_NAMES_FA[cid]} ({CLASS_NAMES_EN[cid]}) "
            f"| {total_detections.get(cid, 0)} | {max_per_frame.get(cid, 0)} |\n"
        )
    return table_md


def render_pdf_download(positive_frames, total_detections, max_per_frame, source_label, processed_frames, max_pdf_frames):
    st.subheader("📄 خروجی PDF فریم‌های تشخیص‌داده‌شده")
    if not positive_frames:
        st.info("هیچ فریم مثبتی برای این اجرا ثبت نشد (یا ذخیره‌سازی برای PDF از تنظیمات کنار خاموش بود).")
        return
    if len(positive_frames) >= max_pdf_frames:
        st.caption(f"⚠️ تعداد فریم‌های مثبت به سقف {max_pdf_frames} رسید؛ PDF فقط شامل {max_pdf_frames} فریم اول است.")
    pdf_bytes = build_pdf_report(positive_frames, total_detections, max_per_frame, source_label, processed_frames)
    st.download_button(
        f"⬇️ دانلود PDF ({len(positive_frames)} فریم تشخیص‌دار)",
        data=pdf_bytes,
        file_name="roadscan_detections.pdf",
        mime="application/pdf",
    )


def build_pdf_report(positive_frames, total_detections, max_per_frame, source_label, processed_frames):
    """Builds a multi-page PDF: a summary page with the per-class table, followed by one page
    per captured positive frame showing the annotated image and which classes were found on it.
    """
    W, H = PDF_PAGE_SIZE
    title_font = get_font(38, bold=True)
    heading_font = get_font(22, bold=True)
    body_font = get_font(18)
    small_font = get_font(15)
    margin = 60

    summary = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(summary)
    y = 80
    draw.text((margin, y), "RoadScan RDD - Detection Report", font=title_font, fill=(15, 20, 25))
    y += 60
    for line in (
        f"Source: {source_label}",
        f"Processed frames: {processed_frames}",
        f"Positive frames in this report: {len(positive_frames)}",
        f"Total detections: {sum(total_detections.values())}",
    ):
        draw.text((margin, y), line, font=body_font, fill=(60, 70, 76))
        y += 32
    y += 30
    draw.text((margin, y), "Detections by class", font=heading_font, fill=(230, 110, 20))
    y += 42
    col_class, col_total, col_max = margin, margin + 480, margin + 700
    draw.text((col_class, y), "Class", font=small_font, fill=(130, 130, 130))
    draw.text((col_total, y), "Total", font=small_font, fill=(130, 130, 130))
    draw.text((col_max, y), "Max / frame", font=small_font, fill=(130, 130, 130))
    y += 26
    draw.line((margin, y, W - margin, y), fill=(210, 214, 218), width=1)
    y += 16
    for cid in sorted(CLASS_NAMES_FA.keys()):
        draw.text((col_class, y), f"{CLASS_NAMES_EN[cid]} ({CLASS_NAMES_FA[cid]})", font=body_font, fill=(30, 35, 40))
        draw.text((col_total, y), str(total_detections.get(cid, 0)), font=body_font, fill=(30, 35, 40))
        draw.text((col_max, y), str(max_per_frame.get(cid, 0)), font=body_font, fill=(30, 35, 40))
        y += 34
        draw.line((margin, y - 8, W - margin, y - 8), fill=(235, 237, 239), width=1)
    pages = [summary]

    caption_h = 90
    avail_w = W - 2 * margin
    avail_h = H - 2 * margin - caption_h
    for i, pf in enumerate(positive_frames):
        img = pf["image"]
        scale = min(avail_w / img.width, avail_h / img.height)
        resized = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        page = Image.new("RGB", (W, H), "white")
        page.paste(resized, ((W - resized.width) // 2, margin))
        draw = ImageDraw.Draw(page)
        cap_y = margin + resized.height + 20
        header = f"Frame {i + 1}/{len(positive_frames)}  |  frame #{pf['frame_idx']}  |  t={pf['time_sec']:.1f}s"
        draw.text((margin, cap_y), header, font=body_font, fill=(20, 20, 20))
        classes_str = ", ".join(
            f"{CLASS_NAMES_EN[cid]} x{count}" for cid, count in sorted(pf["classes"].items())
        )
        draw.text((margin, cap_y + 30), f"Detected: {classes_str}", font=body_font, fill=(190, 60, 0))
        pages.append(page)

    buffer = io.BytesIO()
    pages[0].save(buffer, format="PDF", save_all=True, append_images=pages[1:], quality=PDF_JPEG_QUALITY)
    return buffer.getvalue()


def run_stream(model, cap, conf_thres, iou_thres, imgsz, frame_skip,
                frame_placeholder, counts_placeholder, fps_placeholder,
                stop_flag_key, total_frames=None, progress_bar=None,
                writer=None, video_fps=None, capture_for_pdf=True,
                max_pdf_frames=200):
    total_detections = Counter()
    max_per_frame = Counter()
    # Every processed frame that had at least one detection, kept (annotated, downscaled) for
    # the PDF export below. Capped at max_pdf_frames so a long video full of positive frames
    # can't grow this into a multi-gigabyte in-memory list.
    positive_frames = []
    frame_idx = 0
    last_t = time.time()
    smoothed_fps = 0.0
    start_wall = time.time()

    while True:
        if stop_flag_key is not None and st.session_state.get(stop_flag_key):
            break

        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            frame_class_count = Counter()
            for cls_id, conf, xyxy in infer_boxes(model, frame, conf_thres, iou_thres, imgsz):
                frame = draw_box(frame, xyxy, cls_id, conf)
                total_detections[cls_id] += 1
                frame_class_count[cls_id] += 1
            for cid, c in frame_class_count.items():
                if c > max_per_frame[cid]:
                    max_per_frame[cid] = c
            if capture_for_pdf and frame_class_count and len(positive_frames) < max_pdf_frames:
                pdf_frame = resize_for_pdf(frame.copy())
                time_sec = (frame_idx / video_fps) if video_fps else (time.time() - start_wall)
                positive_frames.append({
                    "frame_idx": frame_idx,
                    "time_sec": time_sec,
                    "image": Image.fromarray(cv2.cvtColor(pdf_frame, cv2.COLOR_BGR2RGB)),
                    "classes": Counter(frame_class_count),
                })

        if writer is not None:
            writer.write(frame)

        # live FPS estimate
        now = time.time()
        dt = now - last_t
        last_t = now
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        smoothed_fps = smoothed_fps * 0.9 + inst_fps * 0.1 if smoothed_fps else inst_fps

        # --- live update every processed frame (real-time feel) ---
        frame_placeholder.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True,
        )
        counts_placeholder.markdown(render_counts_table(total_detections, max_per_frame))
        fps_placeholder.metric("سرعت پردازش زنده (FPS)", f"{smoothed_fps:.1f}")

        if progress_bar is not None and total_frames:
            progress_bar.progress(min((frame_idx + 1) / total_frames, 1.0),
                                   text=f"فریم {frame_idx+1}/{total_frames}")

        frame_idx += 1

    return total_detections, max_per_frame, frame_idx, positive_frames


def main():
    st.title("🛣️ RoadScan RDD — پنل Realtime تشخیص خرابی راه")
    st.caption("مدل YOLO11n پنج‌کلاسه — پخش و پردازش زنده فریم‌به‌فریم")

    weights = find_weight_files()
    if not weights:
        st.error(
            f"هیچ فایل .pt در پوشه `{WEIGHTS_DIR}` پیدا نشد. "
            "فایل‌های best.pt / last.pt را در این پوشه قرار دهید."
        )
        return

    with st.sidebar:
        st.header("تنظیمات")
        model_name = st.selectbox("انتخاب وزن مدل", list(weights.keys()))
        is_tflite_model = model_name.endswith(".tflite")
        if is_tflite_model:
            st.caption("📱 مدل موبایل (TFLite) — دقیقاً همون فایل و همون decode داخل اپ اندروید.")
        conf_thres = st.slider("آستانه اطمینان (confidence)", 0.05, 0.95, 0.25, 0.05)
        iou_thres = st.slider("آستانه IoU (NMS)", 0.1, 0.9, 0.45, 0.05)
        imgsz = st.selectbox(
            "اندازه ورودی مدل (imgsz)", [512, 640, 320], index=0,
            disabled=is_tflite_model,
            help="برای مدل TFLite نادیده گرفته می‌شود؛ ورودی این مدل ثابت روی ۳۲۰×۳۲۰ است." if is_tflite_model else None,
        )
        frame_skip = st.number_input(
            "پردازش هر چند فریم یکبار (۱ = بدون رد کردن فریم، بیشترین دقت)",
            min_value=1, max_value=10, value=1
        )
        st.markdown("---")
        capture_for_pdf = st.checkbox("ذخیره فریم‌های تشخیص‌دار برای خروجی PDF", value=True)
        max_pdf_frames = st.number_input(
            "حداکثر تعداد فریم در PDF خروجی", min_value=10, max_value=1000, value=200, step=10,
            disabled=not capture_for_pdf,
        )
        st.markdown("---")
        source_mode = st.radio("منبع ورودی", ["📼 آپلود ویدیو", "📷 دوربین زنده (وبکم)"])

        if source_mode == "📼 آپلود ویدیو":
            video_file = st.file_uploader("آپلود ویدیو", type=["mp4", "mov", "avi", "mkv"])
        else:
            video_file = None
            cam_index = st.number_input("شماره دوربین (معمولاً 0)", min_value=0, max_value=5, value=0)

        run_btn = st.button("🚀 شروع پردازش زنده", type="primary", use_container_width=True)
        stop_btn = st.button("⏹ توقف", use_container_width=True)

    if "stop_stream" not in st.session_state:
        st.session_state["stop_stream"] = False
    if stop_btn:
        st.session_state["stop_stream"] = True

    model = load_model(weights[model_name])

    col1, col2 = st.columns([2, 1])
    with col1:
        frame_placeholder = st.empty()
        progress_bar = st.progress(0, text="آماده") if source_mode == "📼 آپلود ویدیو" else None
    with col2:
        fps_placeholder = st.empty()
        counts_placeholder = st.empty()

    if not run_btn:
        if source_mode == "📼 آپلود ویدیو" and video_file:
            st.video(video_file)
        st.info("برای شروع پخش و پردازش زنده، دکمه «شروع پردازش زنده» را بزنید.")
        return

    st.session_state["stop_stream"] = False

    if source_mode == "📷 دوربین زنده (وبکم)":
        cap = cv2.VideoCapture(int(cam_index))
        if not cap.isOpened():
            st.error("دوربین باز نشد. شماره دوربین را بررسی کنید یا دسترسی مرورگر/سیستم را چک کنید.")
            return
        total_detections, max_per_frame, n_frames, positive_frames = run_stream(
            model, cap, conf_thres, iou_thres, imgsz, frame_skip,
            frame_placeholder, counts_placeholder, fps_placeholder,
            stop_flag_key="stop_stream",
            capture_for_pdf=capture_for_pdf, max_pdf_frames=max_pdf_frames,
        )
        cap.release()
        st.success(f"پخش زنده متوقف شد. {n_frames} فریم پردازش شد.")
        st.subheader("📊 نتیجه نهایی")
        st.markdown(render_counts_table(total_detections, max_per_frame))
        render_pdf_download(positive_frames, total_detections, max_per_frame, f"webcam #{cam_index}", n_frames, max_pdf_frames)
        return

    # ---- uploaded video, but played/processed live frame-by-frame ----
    if not video_file:
        st.warning("یک ویدیو آپلود کنید.")
        return

    in_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1])
    in_tmp.write(video_file.read())
    in_tmp.flush()
    in_path = in_tmp.name

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        st.error("باز کردن فایل ویدیو ناموفق بود.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = in_path + "_annotated.mp4"
    writer = open_video_writer(out_path, fps, width, height)

    total_detections, max_per_frame, n_frames, positive_frames = run_stream(
        model, cap, conf_thres, iou_thres, imgsz, frame_skip,
        frame_placeholder, counts_placeholder, fps_placeholder,
        stop_flag_key="stop_stream",
        total_frames=total_frames, progress_bar=progress_bar,
        writer=writer, video_fps=fps,
        capture_for_pdf=capture_for_pdf, max_pdf_frames=max_pdf_frames,
    )

    cap.release()
    writer.release()

    st.success("پردازش ویدیو تمام شد ✅" if n_frames >= total_frames else "پردازش متوقف شد.")

    st.subheader("📊 نتیجه نهایی تشخیص")
    st.markdown(render_counts_table(total_detections, max_per_frame))
    st.metric("مجموع کل تشخیص‌ها", sum(total_detections.values()))

    render_pdf_download(positive_frames, total_detections, max_per_frame, video_file.name, n_frames, max_pdf_frames)

    st.subheader("🎬 دانلود ویدیوی خروجی (با bounding box)")
    with open(out_path, "rb") as f:
        video_bytes = f.read()
    st.download_button(
        "⬇️ دانلود ویدیوی نتیجه",
        data=video_bytes,
        file_name="roadscan_result.mp4",
        mime="video/mp4",
    )

    try:
        os.remove(in_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
