import streamlit as st
from PIL import Image, ImageOps
import io, base64, requests

# Config
OLLAMA_BASE = "http://127.0.0.1:11434"  # change if needed
MODEL_NAME = "qwen2.5vl:3b"                # you already pulled it

st.set_page_config(page_title="Photo Preview", page_icon="I", layout="centered")
st.title("Photo Caption (Ollama / Qwen)")
st.write("Upload a photo and I’ll show it nicely here. Click the button to caption it locally with model.")

uploaded = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "webp", "heic", "heif"],
    accept_multiple_files=False,
    help="PNG, JPG, JPEG, WEBP, HEIC/HEIF"
)

if uploaded is not None:
    # --- Read once ---
    raw_bytes = uploaded.read()

    # --- Open & fix orientation ---
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        st.error(f"Could not read that image: {e}")
        st.stop()

    # --- Display info & image ---
    w, h = img.size
    fmt = img.format or "Unknown"
    st.caption(f"**{uploaded.name}** — {w}×{h}px — {fmt}")
    st.image(img, use_container_width=True)

    # --- Optional: basic EXIF peek ---
    with st.expander("Show basic metadata"):
        meta = getattr(img, "getexif", lambda: {})()
        if meta:
            items = []
            for k, v in dict(meta).items():
                if k in (256, 257, 271, 272, 274, 306, 36867):
                    items.append((k, v))
            if items:
                for k, v in items:
                    st.write(f"{k}: {v}")
            else:
                st.write("No common EXIF fields found.")
        else:
            st.write("No EXIF metadata found.")

    # --- Prep image for the model (use the fixed orientation) ---
    img_buf = io.BytesIO()
    img.convert("RGB").save(img_buf, format="JPEG", quality=95)
    img_b64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")

    # --- Prompt + Button ---
    prompt = st.text_input(
        "Caption prompt",
        value="Describe this photo in one vivid, specific sentence.",
        help="Change this to steer style/length."
    )

    if st.button("Generate caption with Model"):
        with st.spinner("Thinking…"):
            try:
                r = requests.post(
                    f"{OLLAMA_BASE}/api/generate",
                    json={
                        "model": MODEL_NAME,
                        "prompt": prompt,
                        "images": [img_b64],
                        "stream": False
                    },
                    timeout=120,
                )
                r.raise_for_status()
                caption = (r.json().get("response") or "").strip()
                if caption:
                    st.subheader("Caption")
                    st.write(caption)
                else:
                    st.warning("No caption returned.")
            except requests.RequestException as e:
                st.error(f"Ollama request failed: {e}")

else:
    st.info("Drop an image above or click to browse. iPhone/Android rotations are fixed automatically.")
