import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2

# CONFIG & SETUP
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "best_model.pth"
DATASET_ROOT = PROJECT_ROOT / "finalised_dataset" / "splits" / "test"

CLASS_NAMES = ["Dustin", "Eleven", "Hopper", "Lucas", "Steve"]
NUM_CLASSES = 5
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
STRICT_SINGLE_FACE = True

# Device detection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# LOAD MODEL
from resnet50_model import FaceResNet50

model = FaceResNet50(num_classes=NUM_CLASSES)
try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Could not load model: {e}")

model = model.to(DEVICE)
model.eval()

# MTCNN FACE DETECTOR
mtcnn = MTCNN(keep_all=True, device=DEVICE)
print(f"MTCNN detector initialized on {DEVICE}")

# PREPROCESSING
def preprocess_face(face_img):
    """Match validation preprocessing: resize, normalize, then tensor conversion."""
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])
    face_rgb = face_img.convert("RGB")
    face_np = np.array(face_rgb)
    return transform(image=face_np)["image"]


def clip_box(box, width, height):
    left = max(0, int(np.floor(box[0])))
    top = max(0, int(np.floor(box[1])))
    right = min(width, int(np.ceil(box[2])))
    bottom = min(height, int(np.ceil(box[3])))

    if right <= left or bottom <= top:
        return None

    return left, top, right, bottom


# PREDICTION & VISUALIZATION
def select_best_face(boxes: np.ndarray, probs: np.ndarray | None) -> int:
    if probs is not None and len(probs) == len(boxes):
        return int(np.argmax(probs))

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(areas))


def predict_characters(image, threshold, allow_multiple_faces, force_label):
    """
    Main inference function.
    Returns: (annotated_image, result_text, probability_chart)
    """
    if image is None:
        return None, "No image uploaded", None

    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image.astype('uint8')).convert("RGB")
    else:
        pil_image = image.convert("RGB")

    # Detect faces
    try:
        detection = mtcnn.detect(pil_image, landmarks=False)
    except:
        # No faces detected
        return pil_image, "No faces detected in image", None

    if detection is None:
        return pil_image, "No faces detected in image", None

    if len(detection) == 2:
        boxes, probs = detection
    else:
        boxes, probs, _landmarks = detection

    # Handle detection results
    if boxes is None:
        return pil_image, "No faces detected in image", None

    original_face_count = len(boxes)
    selected_note = ""
    if not allow_multiple_faces and len(boxes) > 1:
        best_idx = select_best_face(boxes, probs)
        boxes = boxes[best_idx:best_idx + 1]
        if probs is not None and len(probs) == original_face_count:
            probs = probs[best_idx:best_idx + 1]
        selected_note = "Multiple faces found; showing best match.\n\n"

    # Ensure boxes is 2D
    if len(boxes.shape) == 1:
        boxes = boxes.reshape(1, -1)
    if isinstance(probs, np.ndarray) and len(probs.shape) == 0:
        probs = np.array([probs])

    num_faces = len(boxes)
    
    # RUN INFERENCE ON EACH FACE 
    results = []
    all_probs = []

    for face_idx, (box, conf) in enumerate(zip(boxes, probs if probs is not None else [1.0] * num_faces)):
        clipped = clip_box(np.asarray(box, dtype=float), pil_image.width, pil_image.height)
        if clipped is None:
            continue

        x1, y1, x2, y2 = clipped
        face_crop = pil_image.crop(clipped)

        # Preprocess
        face_tensor = preprocess_face(face_crop).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            logits = model(face_tensor)
            probs_face = F.softmax(logits, dim=1)[0].cpu().numpy()

        all_probs.append(probs_face)
        max_prob = probs_face.max()
        pred_class = probs_face.argmax()

        # Check confidence
        if max_prob < threshold:
            if force_label:
                label = f"{CLASS_NAMES[pred_class]} (low confidence)"
            else:
                label = "Uncertain"
            confidence_pct = max_prob * 100
        else:
            label = CLASS_NAMES[pred_class]
            confidence_pct = max_prob * 100

        results.append({
            "box": (x1, y1, x2, y2),
            "label": label,
            "confidence": confidence_pct,
            "pred_class": pred_class,
            "max_prob": max_prob,
        })

    if not results:
        return pil_image, "No valid face crops detected", None

    # DRAW BOUNDING BOXES & LABELS
    annotated = pil_image.copy()
    draw = ImageDraw.Draw(annotated)

    for result in results:
        x1, y1, x2, y2 = result["box"]
        label_text = f"{result['label']} ({result['confidence']:.1f}%)"

        # Draw box
        box_color = (0, 255, 0) if result["max_prob"] >= threshold else (255, 0, 0)
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

        # Draw label background
        text_bbox = draw.textbbox((x1, y1 - 25), label_text)
        draw.rectangle(text_bbox, fill=box_color)
        draw.text((x1, y1 - 25), label_text, fill=(255, 255, 255))

    # GENERATE RESULT TEXT 
    if num_faces == 1:
        result_text = (
            f"{selected_note}Found 1 face\n\n"
            f"{results[0]['label']} ({results[0]['confidence']:.1f}%)"
        )
    else:
        result_text = f"{selected_note}Found {num_faces} faces:\n\n"
        for i, r in enumerate(results):
            result_text += f"Face {i+1}: {r['label']} ({r['confidence']:.1f}%)\n"

    # GENERATE PROBABILITY BAR CHART
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Average probabilities across all faces
    avg_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    
    colors = ['#1f77b4' if p >= threshold else '#ff7f0e' for p in avg_probs]
    bars = ax.bar(CLASS_NAMES, avg_probs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel("Probability", fontsize=12, fontweight='bold')
    ax.set_title("Character Probabilities (Red = Below Confidence Threshold)", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax.legend()
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()

    return annotated, result_text, fig


# COLLECT EXAMPLE IMAGES
def get_examples():
    """Collect example images from finalised_dataset/splits/test/."""
    examples = []
    if DATASET_ROOT.exists():
        for character_dir in DATASET_ROOT.iterdir():
            if character_dir.is_dir():
                image_files = list(character_dir.glob("*.jpg")) + list(character_dir.glob("*.png"))
                # Take first 2 examples per character
                for img_file in image_files[:2]:
                    examples.append(str(img_file))
    return examples


# GRADIO INTERFACE
examples = get_examples()

with gr.Blocks(title="Stranger Things Character Identifier") as demo:
    gr.Markdown("# 🎬 Stranger Things Character Identifier")
    gr.Markdown(
        "Upload an image and detect which Stranger Things character appears in it using **MTCNN + ResNet-50**."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Image")
            image_input = gr.Image(
                label="Input Image",
                type="pil",
                sources=["upload", "clipboard"]
            )
            threshold_input = gr.Slider(
                label="Confidence Threshold",
                minimum=0.3,
                maximum=0.9,
                value=DEFAULT_CONFIDENCE_THRESHOLD,
                step=0.05,
            )
            allow_multi_input = gr.Checkbox(
                label="Allow multiple faces",
                value=True,
            )
            force_label_input = gr.Checkbox(
                label="Show label even if below threshold",
                value=True,
            )
            submit_btn = gr.Button("🔍 Detect & Identify", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Results")
            result_text = gr.Textbox(
                label="Prediction",
                interactive=False,
                lines=5
            )

    with gr.Row():
        annotated_image = gr.Image(
            label="Detected Faces (Green=Confident, Red=Uncertain)",
            interactive=False
        )
        probability_chart = gr.Plot(label="Class Probabilities")

    gr.Markdown("### Example Images (Click to Test)")
    gr.Examples(
        examples=examples,
        inputs=[image_input],
        label="Test Set Samples"
    )

    # Connect submit button
    submit_btn.click(
        predict_characters,
        inputs=[image_input, threshold_input, allow_multi_input, force_label_input],
        outputs=[annotated_image, result_text, probability_chart]
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
