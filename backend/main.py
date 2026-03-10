from io import BytesIO
import base64
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import threading
import uuid
from typing import Callable

import mediapipe as mp
import numpy as np
import torch
from diffusers import AutoPipelineForInpainting
from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from transformers import OwlViTForObjectDetection, OwlViTProcessor, SamModel, SamProcessor

try:
    import cv2
except Exception:
    cv2 = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def to_device(batch: dict[str, torch.Tensor], target_device: str) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if not hasattr(value, "to"):
            converted[key] = value
            continue
        tensor = value
        if (
            target_device == "mps"
            and isinstance(tensor, torch.Tensor)
            and tensor.is_floating_point()
            and tensor.dtype == torch.float64
        ):
            tensor = tensor.to(dtype=torch.float32)
        converted[key] = tensor.to(target_device)
    return converted


def encode_pil_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def safe_pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def now_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def is_fingernail_target(target: str) -> bool:
    normalized = target.strip().lower()
    allowed = {
        "fingernails",
        "fingernail",
        "nail",
        "nails",
        "nail beds",
        "white fingernails",
    }
    return normalized in allowed


def ensure_same_size_rgb(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if image.size != size:
        image = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
    return image.convert("RGB")


def ensure_same_size_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    if mask.size != size:
        mask = ImageOps.fit(mask, size, method=Image.Resampling.LANCZOS)
    return mask.convert("L")


def build_mask_with_mediapipe_sam(image: Image.Image) -> tuple[Image.Image, int]:
    width, height = image.size
    mp_result = hands.process(np.array(image))

    if not mp_result.multi_hand_landmarks:
        raise ValueError("Could not find hands in the image.")

    fingertip_boxes: list[list[float]] = []
    finger_joints = [(4, 3), (8, 7), (12, 11), (16, 15), (20, 19)]

    for hand_landmarks in mp_result.multi_hand_landmarks:
        for tip_id, joint_id in finger_joints:
            tip = hand_landmarks.landmark[tip_id]
            joint = hand_landmarks.landmark[joint_id]

            t_x, t_y = tip.x * width, tip.y * height
            j_x, j_y = joint.x * width, joint.y * height

            dx = j_x - t_x
            dy = j_y - t_y
            segment_length = np.hypot(dx, dy)

            # Extend from fingertip toward cuticle along the finger axis.
            base_x = t_x + dx * 0.45
            base_y = t_y + dy * 0.45
            padding = max(2, int(segment_length * 0.35))

            min_x = max(0, int(min(t_x, base_x) - padding))
            min_y = max(0, int(min(t_y, base_y) - padding))
            max_x = min(width - 1, int(max(t_x, base_x) + padding))
            max_y = min(height - 1, int(max(t_y, base_y) + padding))

            if max_x <= min_x or max_y <= min_y:
                continue

            x1, y1, x2, y2 = min_x, min_y, max_x, max_y
            fingertip_boxes.append([float(x1), float(y1), float(x2), float(y2)])

    if not fingertip_boxes:
        raise ValueError("Could not build fingertip boxes from detected hands.")

    sam_inputs = sam_processor(
        image,
        input_boxes=[fingertip_boxes],
        return_tensors="pt",
    )
    sam_inputs = to_device(sam_inputs, device)

    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)

    processed_masks = sam_processor.image_processor.post_process_masks(
        sam_outputs.pred_masks.detach().cpu(),
        sam_inputs["original_sizes"].detach().cpu(),
        sam_inputs["reshaped_input_sizes"].detach().cpu(),
    )

    mask_tensor = processed_masks[0]
    if mask_tensor.ndim == 4:
        mask_tensor = mask_tensor[:, 0, :, :]

    combined_mask = torch.any(mask_tensor > 0, dim=0).numpy()
    mask_array = combined_mask.astype(np.uint8) * 255
    mask_image = Image.fromarray(mask_array, mode="L").convert("RGB")

    hand_count = len(mp_result.multi_hand_landmarks)
    return mask_image, hand_count


def build_mask_with_owlvit_sam(
    image: Image.Image,
    target: str,
    detection_threshold: float,
) -> tuple[Image.Image, int, float]:
    owl_inputs = owl_processor(text=[[target]], images=image, return_tensors="pt")
    owl_inputs = to_device(owl_inputs, device)

    with torch.no_grad():
        owl_outputs = owl_model(**owl_inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    fallback_thresholds = [detection_threshold, max(0.03, detection_threshold * 0.5), 0.02]
    boxes = None
    threshold_used = detection_threshold
    for threshold in fallback_thresholds:
        detections = owl_processor.post_process_grounded_object_detection(
            owl_outputs,
            threshold=threshold,
            target_sizes=target_sizes,
        )[0]
        candidate_boxes = detections["boxes"]
        if candidate_boxes.numel() > 0:
            boxes = candidate_boxes
            threshold_used = threshold
            break

    if boxes is None:
        raise ValueError(
            f"Could not find '{target}' in image. Try target text like 'fingernails', 'nails', or lower threshold."
        )

    sam_inputs = sam_processor(
        image,
        input_boxes=[boxes.detach().cpu().tolist()],
        return_tensors="pt",
    )
    sam_inputs = to_device(sam_inputs, device)

    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)

    processed_masks = sam_processor.image_processor.post_process_masks(
        sam_outputs.pred_masks.detach().cpu(),
        sam_inputs["original_sizes"].detach().cpu(),
        sam_inputs["reshaped_input_sizes"].detach().cpu(),
    )

    mask_tensor = processed_masks[0]
    if mask_tensor.ndim == 4:
        mask_tensor = mask_tensor[:, 0, :, :]

    combined_mask = torch.any(mask_tensor > 0, dim=0).numpy()
    mask_array = combined_mask.astype(np.uint8) * 255
    mask_image = Image.fromarray(mask_array, mode="L").convert("RGB")
    return mask_image, int(boxes.shape[0]), threshold_used


def refine_mask(mask_l: Image.Image) -> tuple[Image.Image, Image.Image]:
    mask_np = np.array(mask_l, dtype=np.uint8)

    if cv2 is not None:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel_close)

        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        hard_np = cv2.dilate(closed_np, kernel_dilate, iterations=1)
        soft_np = cv2.GaussianBlur(hard_np, (9, 9), 0)

        hard_mask = Image.fromarray(hard_np, mode="L")
        soft_mask = Image.fromarray(soft_np, mode="L")
    else:
        # Fallback if OpenCV isn't installed.
        hard_mask = mask_l.filter(ImageFilter.MaxFilter(7))
        soft_mask = hard_mask.filter(ImageFilter.GaussianBlur(radius=2.5))

    return hard_mask, soft_mask


def run_pipeline(
    init_image: Image.Image,
    resolved_prompt: str,
    resolved_target: str,
    detection_threshold: float,
    feedback: str,
    progress_hook: Callable[[int, str], None] | None = None,
    log_hook: Callable[[str], None] | None = None,
) -> dict:
    if progress_hook is not None:
        progress_hook(1, "Step 1/3: Using OpenCV & SAM to generate perfect masks...")
    if log_hook is not None:
        log_hook(
            f"Pipeline start: target='{resolved_target}', prompt='{resolved_prompt}', "
            f"detection_threshold={detection_threshold:.2f}"
        )

    if is_fingernail_target(resolved_target) and hands is not None:
        mask_image, item_count = build_mask_with_mediapipe_sam(init_image)
        detector_used = "mediapipe"
        threshold_used = None
    else:
        mask_image, item_count, threshold_used = build_mask_with_owlvit_sam(
            image=init_image,
            target=resolved_target,
            detection_threshold=detection_threshold,
        )
        detector_used = "owlvit_sam"
    if log_hook is not None:
        log_hook(f"Detector program: {detector_used}, detected_items={item_count}")

    mask_l = mask_image.convert("L")
    hard_mask, soft_mask = refine_mask(mask_l)
    if log_hook is not None:
        log_hook(f"Mask refinement program: {'opencv-python' if cv2 is not None else 'Pillow fallback'}")

    hard_mask_rgb = hard_mask.convert("RGB")
    soft_mask_rgb = soft_mask.convert("RGB")

    mask_image.save(debug_mask_path)
    hard_mask_rgb.save(debug_expanded_mask_path)
    soft_mask_rgb.save(debug_feathered_mask_path)

    init_rgb = init_image.convert("RGB")
    soft_mask_l = ensure_same_size_mask(soft_mask, init_image.size)
    hard_mask_l = ensure_same_size_mask(hard_mask, init_image.size)
    width, height = init_image.size

    soft_mask_np = np.array(soft_mask_l, dtype=np.uint8)
    if cv2 is not None:
        # Group nearby nail regions into hand-level blobs for per-hand HD processing.
        kernel_group = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (150, 150))
        grouped_mask = cv2.dilate(soft_mask_np, kernel_group)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grouped_mask, connectivity=8)
        components: list[tuple[int, int, int, int, int]] = []
        for label in range(1, num_labels):
            x, y, w_box, h_box, area = stats[label]
            if area < 500:
                continue
            components.append((int(x), int(y), int(w_box), int(h_box), int(area)))
    else:
        # Fallback: single component from non-zero mask bounds.
        y_indices, x_indices = np.where(soft_mask_np > 0)
        components = []
        if len(x_indices) > 0 and len(y_indices) > 0:
            x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
            y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
            components.append((x_min, y_min, x_max - x_min + 1, y_max - y_min + 1, int(len(x_indices))))

    if log_hook is not None:
        log_hook(f"Connected hand regions detected: {len(components)}")

    if not components:
        raise ValueError("No valid hand regions found after mask refinement.")

    working_image = init_rgb.copy()
    total_regions = len(components)

    for idx, (x, y, w_box, h_box, area) in enumerate(components, start=1):
        if progress_hook is not None:
            progress_hook(2, f"Step 2/3: Processing hand {idx} of {total_regions}...")

        padding = 50
        crop_x1 = max(0, x - padding)
        crop_y1 = max(0, y - padding)
        crop_x2 = min(width, x + w_box + padding)
        crop_y2 = min(height, y + h_box + padding)

        crop_img = working_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        crop_hard_mask = hard_mask_l.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        crop_soft_mask = soft_mask_l.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        crop_soft_mask = crop_soft_mask.filter(ImageFilter.GaussianBlur(radius=3))
        original_crop_size = crop_img.size

        ai_input_img = crop_img.resize((512, 512), Image.Resampling.LANCZOS)
        ai_input_mask = crop_hard_mask.resize((512, 512), Image.Resampling.LANCZOS).convert("RGB")
        ai_soft_mask = crop_soft_mask.resize((512, 512), Image.Resampling.LANCZOS).convert("L")
        if log_hook is not None:
            log_hook(
                f"Region {idx}/{total_regions}: bbox=({x},{y},{w_box},{h_box}), area={area}, "
                f"crop=({crop_x1},{crop_y1})-({crop_x2},{crop_y2}), ai_input_size=(512,512)"
            )

        prompt_pass1 = (
            f"bright {resolved_prompt} fingernail polish, glossy metallic texture, professional cosmetic photography, 8k"
            if resolved_prompt
            else "bright silver fingernail polish, glossy metallic texture, professional cosmetic photography, 8k"
        )
        negative_pass1 = "black, void, holes, dark spots, ugly, blurry, poorly drawn"
        if log_hook is not None:
            log_hook("Pass1 program: Dreamshaper 8 inpaint (per-hand HD crop)")
            log_hook(f"Pass1 prompt: {prompt_pass1}")
            log_hook(f"Pass1 negative_prompt: {negative_pass1}")
            log_hook("Pass1 params: num_inference_steps=30, guidance_scale=7.5, strength=0.75")

        pass1_result = pipe(
            prompt=prompt_pass1,
            negative_prompt=negative_pass1,
            image=ai_input_img,
            mask_image=ai_input_mask,
            num_inference_steps=30,
            guidance_scale=7.5,
            strength=0.75,
        ).images[0]

        pass1_result = ensure_same_size_rgb(pass1_result, (512, 512))
        pass1_composite = Image.composite(pass1_result, ai_input_img.convert("RGB"), ai_soft_mask)

        if progress_hook is not None:
            progress_hook(3, f"Step 3/3: Refining highlights for hand {idx} of {total_regions}...")

        refinement_focus = (
            feedback.strip()
            if feedback and feedback.strip()
            else "add bright soft white specular highlights, cohesive room lighting, flawless shiny manicure"
        )
        prompt_pass2 = f"Fix image inconsistencies, {refinement_focus}"
        negative_pass2 = "dark reflections, artifacts, unpolished, matte"
        if log_hook is not None:
            log_hook("Pass2 program: Dreamshaper 8 inpaint (per-hand HD crop)")
            log_hook(f"Pass2 prompt: {prompt_pass2}")
            log_hook(f"Pass2 negative_prompt: {negative_pass2}")
            log_hook("Pass2 params: num_inference_steps=20, guidance_scale=5.0, strength=0.35")

        pass2_result = pipe(
            prompt=prompt_pass2,
            negative_prompt=negative_pass2,
            image=pass1_composite,
            mask_image=ai_soft_mask,
            num_inference_steps=20,
            guidance_scale=5.0,
            strength=0.35,
        ).images[0]

        finished_crop = ensure_same_size_rgb(pass2_result, (512, 512)).resize(
            original_crop_size, Image.Resampling.LANCZOS
        )
        working_image.paste(finished_crop, (crop_x1, crop_y1), crop_soft_mask.convert("L"))

    final_image = working_image
    if log_hook is not None:
        log_hook("Final compositing program: per-hand crop reintegration with soft alpha mask")

    return {
        "status": "success",
        "image_base64": encode_pil_to_base64(final_image),
        "mask_base64": encode_pil_to_base64(mask_image),
        "expanded_mask_base64": encode_pil_to_base64(hard_mask_rgb),
        "feathered_mask_base64": encode_pil_to_base64(soft_mask_rgb),
        "count": item_count,
        "detector_used": detector_used,
        "detection_threshold": threshold_used,
    }


device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32
print(f"Using device: {device}")

print("Loading OwlViT detector (fallback path)...")
owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
owl_model.eval()

print("Initializing MediaPipe hand tracker...")
hands = None
mediapipe_init_error = None
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.01,
    )
except Exception as exc:
    mediapipe_init_error = str(exc)
    print(f"MediaPipe disabled, falling back to OwlViT+SAM: {exc}")

print("Loading SAM segmenter...")
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
sam_model.eval()

print("Loading Dreamshaper 8 Inpaint Pipeline (Mac Optimized fp16)...")
pipe = AutoPipelineForInpainting.from_pretrained(
    "Lykon/dreamshaper-8-inpainting",
    torch_dtype=torch.float16,
    variant="fp16",
).to(device)
pipe.enable_attention_slicing()

print("All models loaded.")
debug_mask_path = Path(__file__).with_name("DEBUG_SAM_MASK.png")
debug_expanded_mask_path = Path(__file__).with_name("DEBUG_EXPANDED_MASK.png")
debug_feathered_mask_path = Path(__file__).with_name("DEBUG_FEATHERED_MASK.png")

active_jobs: dict[str, dict] = {}
active_jobs_lock = threading.Lock()


def append_job_log(job_id: str, message: str) -> None:
    stamped = f"[{now_utc()}] {message}"
    with active_jobs_lock:
        job = active_jobs.setdefault(
            job_id,
            {"stage": 0, "message": "Initializing...", "result": None, "logs": []},
        )
        logs = job.setdefault("logs", [])
        logs.append(stamped)
        if len(logs) > 500:
            del logs[:-500]


def set_job_status(job_id: str, stage: int, message: str, result: dict | None = None) -> None:
    with active_jobs_lock:
        job = active_jobs.setdefault(
            job_id,
            {"stage": 0, "message": "Initializing...", "result": None, "logs": []},
        )
        job["stage"] = stage
        job["message"] = message
        if result is not None:
            job["result"] = result


def process_edit_pipeline(
    job_id: str,
    img_bytes: bytes,
    prompt: str,
    target: str,
    feedback: str,
    detection_threshold: float,
) -> None:
    try:
        append_job_log(job_id, "Background task started")
        append_job_log(job_id, "Loading input image bytes into PIL")
        init_image = Image.open(BytesIO(img_bytes)).convert("RGB")

        def progress(stage: int, message: str) -> None:
            set_job_status(job_id, stage, message)
            append_job_log(job_id, message)

        def log(message: str) -> None:
            append_job_log(job_id, message)

        result = run_pipeline(
            init_image=init_image,
            resolved_prompt=prompt,
            resolved_target=target,
            detection_threshold=detection_threshold,
            feedback=feedback,
            progress_hook=progress,
            log_hook=log,
        )
        set_job_status(
            job_id,
            4,
            "Done!",
            result={"image_base64": result["image_base64"], "mask_base64": result["feathered_mask_base64"]},
        )
        append_job_log(job_id, "Background task completed successfully")
    except Exception as exc:
        set_job_status(job_id, -1, f"Error: {exc}")
        append_job_log(job_id, f"Background task failed: {exc}")


@app.on_event("startup")
async def log_routes() -> None:
    routes = sorted(
        [
            f"{','.join(sorted(getattr(route, 'methods', [])))} {route.path}"
            for route in app.routes
            if getattr(route, "methods", None)
        ]
    )
    print("Registered API routes:")
    for route in routes:
        print(f"  - {route}")


@app.post("/generate-edit")
@app.post("/generate-edit/")
@app.post("/auto-edit")
@app.post("/auto-edit/")
async def generate_edit(
    image: UploadFile,
    prompt: str | None = Form(None),
    target: str | None = Form(None),
    edit_prompt: str | None = Form(None),
    target_object: str | None = Form(None),
    detection_threshold: float = Form(0.03),
):
    resolved_prompt = (prompt or edit_prompt or "").strip()
    resolved_target = (target or target_object or "fingernails").strip()

    if not resolved_prompt:
        raise HTTPException(status_code=400, detail="Missing prompt/edit_prompt form field.")
    if not resolved_target:
        raise HTTPException(status_code=400, detail="Missing target/target_object form field.")

    try:
        init_image = Image.open(BytesIO(await image.read())).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid input image file.") from exc

    try:
        result = run_pipeline(
            init_image=init_image,
            resolved_prompt=resolved_prompt,
            resolved_target=resolved_target,
            detection_threshold=detection_threshold,
            feedback="",
        )
    except ValueError as exc:
        return {
            "status": "error",
            "message": str(exc),
            "count": 0,
        }

    return {
        "status": "success",
        "image_base64": result["image_base64"],
        "mask_base64": result["mask_base64"],
        "expanded_mask_base64": result["expanded_mask_base64"],
        "feathered_mask_base64": result["feathered_mask_base64"],
        "hand_count": result["count"],
        "detector_used": result["detector_used"],
        "detection_threshold": result["detection_threshold"],
        "mediapipe_available": hands is not None,
        "mediapipe_error": mediapipe_init_error,
        "debug_mask_path": str(debug_mask_path),
        "debug_expanded_mask_path": str(debug_expanded_mask_path),
        "debug_feathered_mask_path": str(debug_feathered_mask_path),
    }


@app.post("/start-edit")
@app.post("/start-edit/")
async def start_edit(
    background_tasks: BackgroundTasks,
    image: UploadFile,
    prompt: str = Form("silver"),
    feedback: str = Form(""),
    target: str = Form("fingernails"),
    detection_threshold: float = Form(0.03),
):
    try:
        img_bytes = await image.read()
        if not img_bytes:
            raise HTTPException(status_code=400, detail="Empty input image file.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid input image file.") from exc

    job_id = str(uuid.uuid4())
    runtime_meta = (
        f"Runtime: python={platform.python_version()}, platform={platform.platform()}, device={device}, "
        f"torch={safe_pkg_version('torch')}, diffusers={safe_pkg_version('diffusers')}, "
        f"transformers={safe_pkg_version('transformers')}, mediapipe={safe_pkg_version('mediapipe')}, "
        f"opencv-python={safe_pkg_version('opencv-python')}, pillow={safe_pkg_version('Pillow')}"
    )
    with active_jobs_lock:
        active_jobs[job_id] = {"stage": 0, "message": "Initializing...", "result": None, "logs": []}
    set_job_status(job_id, 0, "Initializing...")
    append_job_log(job_id, f"Job created: id={job_id}")
    append_job_log(job_id, runtime_meta)
    append_job_log(
        job_id,
        f"Input: prompt='{prompt.strip() or 'silver'}', target='{target.strip() or 'fingernails'}', "
        f"feedback='{feedback}', detection_threshold={detection_threshold:.2f}",
    )
    append_job_log(job_id, "Programs: MediaPipe/OwlViT detector, SAM segmentation, Kandinsky inpaint, Pillow compositing")
    background_tasks.add_task(
        process_edit_pipeline,
        job_id,
        img_bytes,
        prompt.strip() or "silver",
        target.strip() or "fingernails",
        feedback,
        detection_threshold,
    )
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    with active_jobs_lock:
        job = active_jobs.get(job_id, {"stage": 0, "message": "Initializing...", "result": None, "logs": []})
        return {
            "stage": job.get("stage", 0),
            "message": job.get("message", "Initializing..."),
            "result": job.get("result"),
            "logs": job.get("logs", []),
            "logs_text": "\n".join(job.get("logs", [])),
        }
