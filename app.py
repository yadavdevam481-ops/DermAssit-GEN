from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from PIL import Image
import gradio as gr

from src.models.classifier import EfficientNetClassifier
from src.data.dataset import CLASS_NAMES, ID_TO_CLASS
from generate_diffusion import PROMPTS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
CHECKPOINT = Path("checkpoints/baseline/best.pt")

if CHECKPOINT.exists():
    try:
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        MODEL = EfficientNetClassifier(7, pretrained=False).to(DEVICE)
        MODEL.load_state_dict(ckpt["model_state"])
        MODEL.eval()
    except Exception:
        MODEL = None

LABEL_TEXT = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def classify(image):
    if image is None:
        return {"error": "Upload an image."}
    if MODEL is None:
        return {"status": "No trained classifier checkpoint found. Run train_classifier.py first."}

    im = Image.fromarray(image).convert("RGB").resize((224, 224))
    x = np.asarray(im).astype("float32") / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.softmax(MODEL(x), dim=1)[0].cpu().numpy()

    order = np.argsort(-prob)
    return {LABEL_TEXT[ID_TO_CLASS[i]]: float(prob[i]) for i in order}


def generate_message(class_name):
    if not torch.cuda.is_available():
        return "Diffusion generation is intended for a CUDA-capable research environment. Use generate_diffusion.py from a GPU machine."
    return (
        f"Generation prompt: {PROMPTS[class_name]}\n\n"
        "Use generate_diffusion.py with your trained LoRA checkpoint. "
        "Synthetic output is for research only and is not clinically validated."
    )


with gr.Blocks(title="DermAssist-GEN Research Demo") as demo:
    gr.Markdown("# DermAssist-GEN Research Demo")
    gr.Markdown(
        "Research-only interface for minority-class synthesis and image classification. "
        "Do not use this interface for diagnosis or patient care."
    )
    with gr.Tab("Classifier"):
        img = gr.Image(type="numpy", label="Dermoscopic research image")
        btn = gr.Button("Classify")
        out = gr.Label(num_top_classes=7, label="Predicted class probabilities")
        btn.click(classify, inputs=img, outputs=out)

    with gr.Tab("Generator"):
        cls = gr.Dropdown(CLASS_NAMES, value="df", label="Target class")
        msg = gr.Textbox()
        gr.Button("Show generation recipe").click(generate_message, inputs=cls, outputs=msg)

if __name__ == "__main__":
    demo.launch()
