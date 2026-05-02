"""Streamlit demo for Pokemon classification.

Run:
    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_paste_button import paste_image_button

from src.predict import load_checkpoint, predict_topk


PROJECT_ROOT = Path(__file__).resolve().parent
CKPT_DIR = PROJECT_ROOT / "checkpoints"


st.set_page_config(page_title="Pokemon Classifier", page_icon=":sparkles:", layout="wide")
st.title("Pokemon Classifier — Transfer Learning Demo")
st.caption(
    "Upload a Pokemon image and a fine-tuned CNN will return its top-5 predictions. "
    "Trained on the *7,000 Labeled Pokemon* dataset (150 classes)."
)


@st.cache_resource(show_spinner="Loading model...")
def _load(ckpt_path: str):
    return load_checkpoint(ckpt_path)


def list_checkpoints() -> list[Path]:
    if not CKPT_DIR.exists():
        return []
    return sorted(CKPT_DIR.glob("*.pth"))


with st.sidebar:
    st.header("Model")
    ckpts = list_checkpoints()
    if not ckpts:
        st.warning(
            "No checkpoint found in `checkpoints/`. Train a model first:\n\n"
            "```\npython -m experiments.run_all --data-dir data/PokemonData\n```"
        )
        st.stop()
    selected = st.selectbox(
        "Checkpoint",
        options=[c.name for c in ckpts],
        index=0,
        help="Pick a trained experiment.",
    )
    ckpt_path = CKPT_DIR / selected
    top_k = st.slider("Top-K predictions", min_value=1, max_value=10, value=5)

model, class_names, transform, device = _load(str(ckpt_path))
st.sidebar.success(f"Loaded **{selected}**\n\nDevice: `{device}` · Classes: {len(class_names)}")


col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("Input image")
    uploaded = st.file_uploader(
        "Drop a Pokemon image (jpg/png/webp).",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
    )
    pasted = paste_image_button(
        label="📋 클립보드에서 붙여넣기",
        key="paste_btn",
        errors="ignore",
    )
    image = None
    caption = None
    if pasted.image_data is not None:
        image = pasted.image_data
        caption = "Pasted from clipboard"
    elif uploaded is not None:
        image = Image.open(uploaded)
        caption = uploaded.name
    if image is not None:
        st.image(image, caption=caption, use_container_width=True)

with col_output:
    st.subheader("Predictions")
    if image is None:
        st.info("Upload an image on the left to see predictions.")
    else:
        results = predict_topk(image, model, class_names, transform, device, k=top_k)
        df = pd.DataFrame(results, columns=["Pokemon", "Confidence"])
        df["Confidence (%)"] = (df["Confidence"] * 100).round(2)
        top_label, top_conf = results[0]
        st.metric(label="Top-1 prediction", value=top_label, delta=f"{top_conf*100:.2f}%")
        st.bar_chart(df.set_index("Pokemon")["Confidence"])
        st.dataframe(
            df[["Pokemon", "Confidence (%)"]],
            hide_index=True,
            use_container_width=True,
        )

with st.expander("About the experiments"):
    st.markdown(
        "This app loads checkpoints produced by `experiments/run_all.py`. "
        "Five settings are compared: ResNet18 frozen, ResNet18 full fine-tune, "
        "ResNet50 full fine-tune, ResNet18 from scratch, MobileNetV2 full fine-tune. "
        "See `README.md` for the result table and learning curves."
    )
