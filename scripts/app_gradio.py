"""
Stage 9: Gradio Web UI Demo for YOLOv4 Potato Leaf Disease Detection
Interactive interface with real-time detection, agronomic diagnosis,
targeted fungicide recommendations, and disclaimer banners.
"""

from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.infer import predict_image

AGRONOMIC_GUIDE = {
    "potato_early_blight": {
        "title": "Potato Early Blight (*Alternaria solani*)",
        "severity": "Moderate to High",
        "symptoms": "Dark brown to black necrotic spots with characteristic concentric rings ('target-board' pattern), often surrounded by a chlorotic yellow halo. Primarily attacks older lower foliage first.",
        "cultural_control": [
            "Maintain 2-3 year crop rotation away from Solanaceae (tomatoes, peppers, eggplants).",
            "Avoid overhead sprinkler irrigation; irrigate early in the morning so foliage dries rapidly.",
            "Destroy or deeply till crop debris post-harvest to reduce overwintering fungal inocula.",
            "Ensure balanced nitrogen and potassium fertilization; stressed plants are more vulnerable.",
        ],
        "chemical_control": [
            "Protectant fungicides: Chlorothalonil, Mancozeb (apply preventively prior to canopy closure).",
            "Systemic / Translaminar options: Azoxystrobin, Pyraclostrobin, or Difenoconazole at first lesion onset.",
            "Rotate FRAC groups (Group 11 with Groups 3 or 7) to prevent fungicide resistance development.",
        ],
    },
    "potato_late_blight": {
        "title": "Potato Late Blight (*Phytophthora infestans*)",
        "severity": "CRITICAL / EMERGENCY",
        "symptoms": "Water-soaked irregular pale to dark lesions rapidly expanding across leaves and stems. In humid/wet conditions, delicate white fungal sporulation appears on the underside of leaves. Capable of total field destruction within days.",
        "cultural_control": [
            "URGENT: Immediately rogue out and destroy severely infected plants in sealed bags.",
            "Cease all overhead watering immediately; eliminate standing water and improve field drainage.",
            "Monitor weather forecasts for cool, wet periods (relative humidity > 90%, 10-20°C).",
            "Plant certified disease-free seed tubers and resistant cultivars where available.",
        ],
        "chemical_control": [
            "Apply specialized oomycete fungicides immediately: Metalaxyl/Mefenoxam (FRAC 4), Mandipropamid (FRAC 40), Fluopicolide (FRAC 43), or Cymoxanil.",
            "Tank-mix systemic fungicides with broad-spectrum protectants (Mancozeb or Copper hydroxide).",
            "Spray on a strict 5-7 day schedule during active outbreaks until disease progression halts.",
        ],
    },
    "potato_healthy": {
        "title": "Healthy Potato Foliage",
        "severity": "Normal / Healthy",
        "symptoms": "Vigorous, uniformly green leaves with intact cuticle and no visible fungal or bacterial lesions.",
        "cultural_control": [
            "Continue standard agronomic monitoring and scouting weekly.",
            "Maintain optimal drip irrigation and nutrient schedule.",
            "Scout perimeter rows for early insect vectors (aphids, flea beetles).",
        ],
        "chemical_control": [
            "No chemical fungicide application required at this stage.",
            "Consider preventive biological control (Bacillus subtilis) if high humidity is anticipated.",
        ],
    },
}


def diagnose_leaf(image, conf_thresh):
    if image is None:
        return None, "Please upload or capture a potato leaf image."

    # Gradio provides RGB, cv2 expects BGR
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    res = predict_image(img_bgr, conf_thresh=conf_thresh)
    annotated_rgb = cv2.cvtColor(res["annotated_image"], cv2.COLOR_BGR2RGB)

    top_class = res["top_class"]
    top_conf = res["top_confidence"]

    if top_class == "no_detection":
        report = (
            "### ⚠️ No Disease Lesions Detected\n\n"
            f"- **Confidence**: Below threshold ({conf_thresh:.2f})\n"
            "- **Suggestion**: Ensure the leaf is centered and well-lit, or slightly lower the confidence slider."
        )
        return annotated_rgb, report

    info = AGRONOMIC_GUIDE.get(top_class, {})
    title = info.get("title", top_class)
    severity = info.get("severity", "Unknown")
    symptoms = info.get("symptoms", "")
    cult = "\n".join([f"- {c}" for c in info.get("cultural_control", [])])
    chem = "\n".join([f"- {c}" for c in info.get("chemical_control", [])])

    report = f"""
## Diagnosis: **{title}**
- **Model Confidence**: `{top_conf * 100:.2f}%`
- **Disease Severity Rating**: **`{severity}`**
- **Detected Bounding Boxes**: `{len(res['boxes'])}`

---
### 🔍 Visual Symptomatology
{symptoms}

---
### 🌱 Recommended Cultural Management
{cult}

---
### 💊 Chemical & Fungicide Interventions
{chem}

---
> ⚠️ **Agricultural Disclaimer**: This diagnostic tool is an AI decision-support prototype reproducing the methodology of *Aldakheel et al. (2024)*. Confirmatory testing by a certified plant pathologist or university agricultural extension service is recommended before applying chemical sprays.
"""
    return annotated_rgb, report


def build_app():
    import gradio as gr

    # Load 3 representative test samples as examples
    examples = []
    test_txt = PROJECT_ROOT / "data" / "test.txt"
    if test_txt.exists():
        lines = [PROJECT_ROOT / l.strip() for l in test_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
        for target in ["Early_blight", "Late_blight", "healthy"]:
            for p in lines:
                if target in p.name and p.exists():
                    examples.append([str(p), 0.25])
                    break

    with gr.Blocks(title="AgriBot - Potato Leaf Disease Detector (YOLOv4)", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🥔 AgriBot: Precision Potato Leaf Disease Detection
            ### Real-Time Detection and Identification using YOLOv4
            *Based on the methodology of Aldakheel, Zakariah, & Alabdalall (2024), Frontiers in Plant Science.*
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="numpy", label="Upload Potato Leaf Image / Use Webcam")
                conf_slider = gr.Slider(
                    minimum=0.10, maximum=1.00, value=0.25, step=0.05, label="Detection Confidence Threshold"
                )
                submit_btn = gr.Button("🔍 Run Disease Diagnosis", variant="primary")

                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=[input_image, conf_slider],
                        label="Click to Test Benchmark Samples",
                    )

            with gr.Column(scale=1):
                output_image = gr.Image(type="numpy", label="YOLOv4 Detection & Bounding Boxes")
                output_report = gr.Markdown(label="Agronomic Diagnostic Report")

        submit_btn.click(
            fn=diagnose_leaf,
            inputs=[input_image, conf_slider],
            outputs=[output_image, output_report],
        )

    return demo


if __name__ == "__main__":
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Install with: pip install gradio")
        sys.exit(1)

    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
