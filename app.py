"""Stable Streamlit demo for YOLOv8 mask detection.

Functions:
1. single image detection
2. batch image detection
3. uploaded video detection

Design choices:
- Use explicit buttons to trigger inference.
- Use uploader_key increments to reset uploaders safely.
- Use static Markdown tables instead of st.dataframe to avoid frontend DataFrame module issues.
- Load YOLO lazily and cache the model with st.cache_resource.
"""

from __future__ import annotations

import csv
import hashlib
import io
import sys
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parent
TEMP_UPLOADS = ROOT / "temp_uploads"
TEMP_RESULTS = ROOT / "temp_results"
WEIGHT_CANDIDATES = [
    ROOT / "runs/paper/exp6_dataset_v3/weights/best.pt",
    ROOT / "runs/detect/runs/paper/exp6_dataset_v3/weights/best.pt",
]
CLASS_NAMES = {
    0: "with_mask",
    1: "without_mask",
    2: "mask_weared_incorrect",
}
APP_VERSION = "2026-06-15-stable-table-reset"


st.set_page_config(page_title="口罩佩戴检测系统", layout="wide")
st.title("基于 YOLOv8 的口罩佩戴检测系统")
st.info("页面启动成功。上传文件后请点击检测按钮，系统不会自动反复推理。")
st.caption(f"页面版本：{APP_VERSION}")
st.markdown("检测类别：`with_mask`、`without_mask`、`mask_weared_incorrect`。")


@st.cache_resource(show_spinner=False)
def load_model(weight_path: str):
    from ultralytics import YOLO

    return YOLO(str(weight_path))


def find_weight() -> Path | None:
    for candidate in WEIGHT_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def initialize_state() -> None:
    defaults: dict[str, Any] = {
        "single_uploader_key": 0,
        "single_file_id": None,
        "single_result_rows": [],
        "single_result_image_path": None,
        "single_debug": {},
        "batch_uploader_key": 0,
        "batch_file_id": None,
        "batch_results": [],
        "batch_summary": [],
        "batch_selector_key": 0,
        "video_uploader_key": 0,
        "video_file_id": None,
        "video_result_path": None,
        "video_statistics": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def ensure_temp_dirs() -> None:
    TEMP_UPLOADS.mkdir(parents=True, exist_ok=True)
    TEMP_RESULTS.mkdir(parents=True, exist_ok=True)


def safe_filename(filename: str) -> str:
    path = Path(filename)
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in path.stem)
    suffix = path.suffix.lower()
    return f"{stem or 'upload'}{suffix}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_identifier(filename: str, file_bytes: bytes) -> str:
    return f"{filename}_{sha256_bytes(file_bytes)}"


def save_upload(filename: str, file_bytes: bytes, prefix: str) -> Path:
    ensure_temp_dirs()
    digest = sha256_bytes(file_bytes)[:12]
    target = TEMP_UPLOADS / f"{prefix}_{digest}_{safe_filename(filename)}"
    target.write_bytes(file_bytes)
    return target


def display_image(image: Any, caption: str) -> None:
    try:
        st.image(image, caption=caption, width="stretch")
    except TypeError:
        st.image(image, caption=caption, use_container_width=True)


def display_video(video: Any) -> None:
    st.video(video)


def model_class_name(model: Any, class_id: int) -> str:
    names = getattr(model, "names", CLASS_NAMES)
    if isinstance(names, dict):
        return str(names.get(class_id, CLASS_NAMES.get(class_id, str(class_id))))
    try:
        return str(names[class_id])
    except Exception:
        return CLASS_NAMES.get(class_id, str(class_id))


def extract_detections(result: Any, model: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return rows

    for index, box in enumerate(boxes, start=1):
        xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
        conf = float(box.conf[0].detach().cpu().item())
        cls_id = int(box.cls[0].detach().cpu().item())
        rows.append(
            {
                "序号": index,
                "类别 ID": cls_id,
                "类别名称": model_class_name(model, cls_id),
                "置信度": f"{conf:.4f}",
                "x1": f"{float(xyxy[0]):.2f}",
                "y1": f"{float(xyxy[1]):.2f}",
                "x2": f"{float(xyxy[2]):.2f}",
                "y2": f"{float(xyxy[3]):.2f}",
            }
        )
    return rows


def get_annotated_image(result: Any):
    annotated = result.plot()
    return annotated[:, :, ::-1].copy()


def save_result_image(image: Any, filename: str) -> Path:
    from PIL import Image

    ensure_temp_dirs()
    target = TEMP_RESULTS / filename
    Image.fromarray(image).save(target)
    return target


def predict_path(model: Any, source: Path, conf: float, imgsz: int, device: str):
    results = model.predict(
        source=str(source),
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    if not results:
        raise RuntimeError("模型未返回任何结果对象。")
    return results[0]


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def render_detection_table(rows: list[dict[str, Any]], empty_message: str) -> None:
    if not rows:
        st.warning(empty_message)
        return
    st.markdown(markdown_table(rows))


def rows_to_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def get_model_or_error(weight_path: Path | None):
    if weight_path is None:
        st.error("未找到模型权重，无法检测。")
        st.write("已检查以下路径：")
        for candidate in WEIGHT_CANDIDATES:
            st.code(str(candidate))
        return None
    try:
        with st.spinner("正在加载模型，首次加载可能需要稍候..."):
            return load_model(str(weight_path))
    except Exception as exc:
        st.error("模型加载失败。")
        st.exception(exc)
        return None


def clear_single_results() -> None:
    st.session_state.single_result_rows = []
    st.session_state.single_result_image_path = None
    st.session_state.single_debug = {}


def clear_single_upload() -> None:
    st.session_state.single_uploader_key += 1
    st.session_state.single_file_id = None
    clear_single_results()


def clear_batch_results() -> None:
    st.session_state.batch_results = []
    st.session_state.batch_summary = []
    st.session_state.batch_selector_key += 1


def clear_batch_upload() -> None:
    st.session_state.batch_uploader_key += 1
    st.session_state.batch_file_id = None
    clear_batch_results()


def clear_video_results() -> None:
    st.session_state.video_result_path = None
    st.session_state.video_statistics = None


def clear_video_upload() -> None:
    st.session_state.video_uploader_key += 1
    st.session_state.video_file_id = None
    clear_video_results()


def reset_all_state() -> None:
    st.cache_resource.clear()
    clear_single_upload()
    clear_batch_upload()
    clear_video_upload()


def single_image_tab(weight_path: Path | None, conf: float, imgsz: int, device: str) -> None:
    st.subheader("单张图片检测")

    reset_cols = st.columns(2)
    if reset_cols[0].button("清空当前图片", key="single_clear_upload"):
        try:
            clear_single_upload()
            st.success("当前图片已清空，可以重新上传。")
            st.rerun()
        except Exception as exc:
            st.error("清空当前图片失败。")
            st.exception(exc)
    if reset_cols[1].button("清空检测结果", key="single_clear_result"):
        try:
            clear_single_results()
            st.success("检测结果已清空。")
            st.rerun()
        except Exception as exc:
            st.error("清空检测结果失败。")
            st.exception(exc)

    uploaded_file = st.file_uploader(
        "上传单张图片",
        type=["jpg", "jpeg", "png"],
        key=f"single_image_uploader_{st.session_state.single_uploader_key}",
    )

    if uploaded_file is None:
        if st.session_state.single_file_id is not None:
            st.session_state.single_file_id = None
            clear_single_results()
        st.info("请上传一张图片。上传后只显示原图，点击“开始检测”才会推理。")
        return

    try:
        from PIL import Image

        file_bytes = uploaded_file.getvalue()
        current_id = file_identifier(uploaded_file.name, file_bytes)
        if current_id != st.session_state.single_file_id:
            st.session_state.single_file_id = current_id
            clear_single_results()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        display_image(image, caption=f"原始图片：{uploaded_file.name}")
    except Exception as exc:
        st.error("图片读取失败。")
        st.exception(exc)
        return

    detect_clicked = st.button("开始检测", type="primary", key="single_detect")
    if detect_clicked:
        model = get_model_or_error(weight_path)
        if model is not None:
            try:
                image_path = save_upload(uploaded_file.name, file_bytes, "single")
                with st.spinner("模型正在检测中，请稍候..."):
                    result = predict_path(model, image_path, conf, imgsz, device)
                rows = extract_detections(result, model)
                annotated = get_annotated_image(result)
                result_name = f"single_{sha256_bytes(file_bytes)[:12]}.png"
                result_path = save_result_image(annotated, result_name)
                boxes = getattr(result, "boxes", None)
                st.session_state.single_result_rows = rows
                st.session_state.single_result_image_path = str(result_path)
                st.session_state.single_debug = {
                    "当前 Python 路径": sys.executable,
                    "当前权重路径": str(weight_path),
                    "当前设备": device,
                    "当前上传文件名": uploaded_file.name,
                    "当前 file_id": current_id,
                    "检测框数量": 0 if boxes is None else len(boxes),
                    "当前 result_rows 长度": len(rows),
                    "当前 uploader_key": st.session_state.single_uploader_key,
                    "推理文件": str(image_path),
                    "模型类别映射": str(getattr(model, "names", CLASS_NAMES)),
                }
                st.success("检测完成。")
            except Exception as exc:
                clear_single_results()
                st.error("单张图片推理失败。")
                st.exception(exc)

    if st.session_state.single_result_image_path:
        result_path = Path(st.session_state.single_result_image_path)
        if result_path.is_file():
            display_image(str(result_path), caption="检测结果")
        else:
            st.warning("检测结果图片文件不存在，请重新检测。")

    if st.session_state.single_result_rows is not None:
        st.subheader("检测结果表格")
        render_detection_table(
            st.session_state.single_result_rows,
            "模型正常运行，但未检测到口罩相关目标。",
        )
        with st.expander("检测调试信息", expanded=False):
            debug_rows = [{"项目": key, "值": value} for key, value in st.session_state.single_debug.items()]
            if debug_rows:
                st.markdown(markdown_table(debug_rows))
            else:
                st.caption("尚未执行检测。")


def batch_image_tab(weight_path: Path | None, conf: float, imgsz: int, device: str) -> None:
    st.subheader("批量图片检测")

    reset_cols = st.columns(2)
    if reset_cols[0].button("清空批量图片", key="batch_clear_upload"):
        try:
            clear_batch_upload()
            st.success("批量图片已清空，可以重新上传。")
            st.rerun()
        except Exception as exc:
            st.error("清空批量图片失败。")
            st.exception(exc)
    if reset_cols[1].button("清空批量结果", key="batch_clear_result"):
        try:
            clear_batch_results()
            st.success("批量检测结果已清空。")
            st.rerun()
        except Exception as exc:
            st.error("清空批量结果失败。")
            st.exception(exc)

    uploaded_files = st.file_uploader(
        "批量上传图片",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"batch_image_uploader_{st.session_state.batch_uploader_key}",
    )

    if not uploaded_files:
        if st.session_state.batch_file_id is not None:
            st.session_state.batch_file_id = None
            clear_batch_results()
        st.info("请上传一张或多张图片。")
        return

    payloads = [(item, item.getvalue()) for item in uploaded_files]
    current_id = "|".join(file_identifier(item.name, data) for item, data in payloads)
    if current_id != st.session_state.batch_file_id:
        st.session_state.batch_file_id = current_id
        clear_batch_results()

    st.write(f"已选择 {len(uploaded_files)} 张图片。")
    if st.button("开始批量检测", type="primary", key="batch_detect"):
        model = get_model_or_error(weight_path)
        if model is not None:
            results_store: list[dict[str, Any]] = []
            summary_store: list[dict[str, Any]] = []
            progress = st.progress(0)
            for index, (uploaded_file, file_bytes) in enumerate(payloads, start=1):
                item: dict[str, Any] = {"filename": uploaded_file.name, "error": None}
                try:
                    image_path = save_upload(uploaded_file.name, file_bytes, "batch")
                    result = predict_path(model, image_path, conf, imgsz, device)
                    rows = extract_detections(result, model)
                    annotated = get_annotated_image(result)
                    digest = sha256_bytes(file_bytes)[:12]
                    annotated_path = save_result_image(annotated, f"batch_{digest}.png")
                    item.update(
                        {
                            "original_path": str(image_path),
                            "annotated_path": str(annotated_path),
                            "rows": rows,
                        }
                    )
                    counts = {name: 0 for name in CLASS_NAMES.values()}
                    numeric_confidences: list[float] = []
                    for row in rows:
                        name = str(row["类别名称"])
                        if name in counts:
                            counts[name] += 1
                        try:
                            numeric_confidences.append(float(row["置信度"]))
                        except Exception:
                            pass
                    summary_store.append(
                        {
                            "图片名": uploaded_file.name,
                            "检测目标数量": len(rows),
                            "with_mask 数量": counts["with_mask"],
                            "without_mask 数量": counts["without_mask"],
                            "mask_weared_incorrect 数量": counts["mask_weared_incorrect"],
                            "最高置信度": f"{max(numeric_confidences, default=0.0):.4f}",
                        }
                    )
                except Exception as exc:
                    item["error"] = repr(exc)
                    summary_store.append(
                        {
                            "图片名": uploaded_file.name,
                            "检测目标数量": 0,
                            "with_mask 数量": 0,
                            "without_mask 数量": 0,
                            "mask_weared_incorrect 数量": 0,
                            "最高置信度": "0.0000",
                        }
                    )
                results_store.append(item)
                progress.progress(index / len(payloads))
            st.session_state.batch_results = results_store
            st.session_state.batch_summary = summary_store
            st.success("批量检测完成。")

    if st.session_state.batch_results:
        st.subheader("逐张查看批量结果")
        labels = [f"{idx + 1}. {item['filename']}" for idx, item in enumerate(st.session_state.batch_results)]
        selected_label = st.selectbox(
            "选择要查看的图片",
            labels,
            key=f"batch_result_selector_{st.session_state.batch_selector_key}",
        )
        selected_index = labels.index(selected_label)
        item = st.session_state.batch_results[selected_index]
        st.caption(f"当前查看第 {selected_index + 1} 张，共 {len(labels)} 张。")
        if item.get("error"):
            st.error(f"该图片检测失败：{item['error']}")
        else:
            original_path = Path(item["original_path"])
            annotated_path = Path(item["annotated_path"])
            if original_path.is_file() and annotated_path.is_file():
                left, right = st.columns(2)
                with left:
                    display_image(str(original_path), caption="原始图片")
                with right:
                    display_image(str(annotated_path), caption="检测结果")
                render_detection_table(
                    item.get("rows", []),
                    "模型正常运行，但该图片未检测到口罩相关目标。",
                )
            else:
                st.warning("该图片的临时结果文件不存在，请重新执行批量检测。")

    if st.session_state.batch_summary:
        st.subheader("批量检测汇总")
        render_detection_table(st.session_state.batch_summary, "暂无批量汇总结果。")
        csv_bytes = rows_to_csv_bytes(st.session_state.batch_summary)
        if csv_bytes:
            st.download_button(
                "下载汇总 CSV",
                data=csv_bytes,
                file_name="batch_detection_summary.csv",
                mime="text/csv",
            )


def process_video(
    model: Any,
    input_path: Path,
    output_path: Path,
    conf: float,
    imgsz: int,
    device: str,
    frame_interval: int,
    progress: Any,
) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件：{input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("无法读取视频尺寸。")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频：{output_path}")

    total_frames = 0
    detected_frames = 0
    class_counts = {name: 0 for name in CLASS_NAMES.values()}
    confidences: list[float] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total_frames += 1
            output_frame = frame
            if (total_frames - 1) % frame_interval == 0:
                results = model.predict(
                    source=frame,
                    conf=conf,
                    imgsz=imgsz,
                    device=device,
                    verbose=False,
                )
                if results:
                    result = results[0]
                    detected_frames += 1
                    output_frame = result.plot()
                    for row in extract_detections(result, model):
                        name = str(row["类别名称"])
                        if name in class_counts:
                            class_counts[name] += 1
                        try:
                            confidences.append(float(row["置信度"]))
                        except Exception:
                            pass
            writer.write(output_frame)
            if total_frames_hint > 0:
                progress.progress(min(total_frames / total_frames_hint, 1.0))
    finally:
        cap.release()
        writer.release()
    progress.progress(1.0)

    return {
        "总帧数": total_frames,
        "实际检测帧数": detected_frames,
        "with_mask 次数": class_counts["with_mask"],
        "without_mask 次数": class_counts["without_mask"],
        "mask_weared_incorrect 次数": class_counts["mask_weared_incorrect"],
        "平均置信度": f"{(sum(confidences) / len(confidences)):.4f}" if confidences else "0.0000",
    }


def video_tab(weight_path: Path | None, conf: float, imgsz: int, device: str, frame_interval: int) -> None:
    st.subheader("上传视频检测")

    reset_cols = st.columns(2)
    if reset_cols[0].button("清空当前视频", key="video_clear_upload"):
        try:
            clear_video_upload()
            st.success("当前视频已清空，可以重新上传。")
            st.rerun()
        except Exception as exc:
            st.error("清空当前视频失败。")
            st.exception(exc)
    if reset_cols[1].button("清空视频结果", key="video_clear_result"):
        try:
            clear_video_results()
            st.success("视频检测结果已清空。")
            st.rerun()
        except Exception as exc:
            st.error("清空视频结果失败。")
            st.exception(exc)

    uploaded_video = st.file_uploader(
        "上传视频文件",
        type=["mp4", "avi", "mov"],
        key=f"video_uploader_{st.session_state.video_uploader_key}",
    )

    if uploaded_video is None:
        st.info("请上传短视频。视频不会自动处理，点击“开始视频检测”后才会推理。")
        return

    video_bytes = uploaded_video.getvalue()
    current_id = file_identifier(uploaded_video.name, video_bytes)
    if current_id != st.session_state.video_file_id:
        st.session_state.video_file_id = current_id
        clear_video_results()

    display_video(video_bytes)
    if st.button("开始视频检测", type="primary", key="video_detect"):
        model = get_model_or_error(weight_path)
        if model is not None:
            try:
                input_path = save_upload(uploaded_video.name, video_bytes, "video")
                digest = sha256_bytes(video_bytes)[:12]
                output_path = TEMP_RESULTS / f"detected_{digest}.mp4"
                progress = st.progress(0)
                with st.spinner("视频正在检测中，请稍候..."):
                    stats = process_video(
                        model,
                        input_path,
                        output_path,
                        conf,
                        imgsz,
                        device,
                        frame_interval,
                        progress,
                    )
                st.session_state.video_result_path = str(output_path)
                st.session_state.video_statistics = stats
                st.success("视频检测完成。")
            except Exception as exc:
                clear_video_results()
                st.error("视频检测失败。")
                st.exception(exc)

    if st.session_state.video_result_path:
        output_path = Path(st.session_state.video_result_path)
        if output_path.is_file():
            st.subheader("处理后视频")
            display_video(str(output_path))
            st.download_button(
                "下载处理后视频",
                data=output_path.read_bytes(),
                file_name=output_path.name,
                mime="video/mp4",
            )
        else:
            st.warning("视频结果文件不存在，请重新检测。")

    if st.session_state.video_statistics:
        st.subheader("视频检测统计")
        render_detection_table([st.session_state.video_statistics], "暂无视频统计结果。")


def usage_tab() -> None:
    st.subheader("使用说明")
    st.markdown(
        """
1. 页面支持单张图片检测、批量图片检测和上传视频检测。
2. 上传文件后不会自动推理，必须点击对应的检测按钮。
3. 如果 GPU 推理卡住，请在侧边栏选择 `cpu`。
4. 如果 Streamlit 自动打开旧页面，请手动访问 `http://127.0.0.1:8502`。
5. 本地摄像头实时检测为可选扩展功能。如果当前环境无法打开摄像头，请使用上传视频检测进行演示。
"""
    )


def main() -> None:
    initialize_state()
    weight_path = find_weight()

    st.sidebar.header("推理设置")
    selected_module = st.sidebar.radio(
        "功能模块",
        ["单张图片检测", "批量图片检测", "上传视频检测", "使用说明"],
        index=0,
    )
    st.sidebar.write("当前权重路径")
    st.sidebar.code(str(weight_path) if weight_path else "未找到可用权重")
    device_option = st.sidebar.selectbox("推理设备", ["cpu", "0"], index=0)
    st.sidebar.caption("如果 GPU 推理卡住，请选择 CPU。")
    conf_threshold = st.sidebar.slider("置信度阈值", 0.05, 1.0, 0.25, 0.05)
    imgsz = st.sidebar.slider("图片尺寸", 320, 1280, 640, 32)
    frame_interval = st.sidebar.slider("视频抽帧间隔", 1, 30, 5, 1)

    if st.sidebar.button("清空所有页面状态和模型缓存"):
        try:
            reset_all_state()
            st.success("缓存和页面状态已清空。")
            st.rerun()
        except Exception as exc:
            st.error("清空缓存失败。")
            st.exception(exc)

    if weight_path is None:
        st.warning("未找到模型权重，请确认 best.pt 是否存在。")
        st.write("已检查以下路径：")
        for candidate in WEIGHT_CANDIDATES:
            st.code(str(candidate))
    else:
        st.success("已找到模型权重。模型将在点击检测按钮时加载，并使用缓存避免重复加载。")

    if selected_module == "单张图片检测":
        single_image_tab(weight_path, conf_threshold, imgsz, device_option)
    elif selected_module == "批量图片检测":
        batch_image_tab(weight_path, conf_threshold, imgsz, device_option)
    elif selected_module == "上传视频检测":
        video_tab(weight_path, conf_threshold, imgsz, device_option, frame_interval)
    else:
        usage_tab()


try:
    main()
except Exception as exc:
    st.error("页面发生未处理异常。")
    st.exception(exc)
