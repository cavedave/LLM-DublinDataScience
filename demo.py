import json
import io
import requests
import streamlit as st
import pandas as pd

# ---------- Config ----------
OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.1:8b"

st.set_page_config(page_title="Local Text Classifier (Ollama)", layout="wide")
st.title("Local Text Classifier (Ollama)")

# Helpers
def classify_text(model: str, text: str, label_list: list[str], few_shots: list[dict] | None = None):
    """
    Calls Ollama /api/generate with a few-shot prompt and requests strict JSON output:
    {"label":"<one>","confidence":0-1}
    Returns (label, confidence, raw_text_response).
    """
    if few_shots is None:
        few_shots = []

    system = (
        "You are an expert email text classifier.\n"
        f"Choose exactly one label from {label_list}.\n"
        "Respond ONLY as compact JSON: {\"label\":\"<one>\",\"confidence\":0-1}.\n"
        "Do not include extra keys or explanations"
    )
    shots = "\n".join(
        [f"Text: {s['text']}\nLabel: {s['label']}" for s in few_shots]
    )
    user = f"Text: {text}\nLabel:"
    prompt = f"<<SYS>>{system}<<SYS>>\n\n{shots}\n\n{user}".strip()

    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "options": {"temperature": 0.0, "num_ctx": 2048},
                "stream": False,
            },
            timeout=120,
        )
        if resp.status_code == 404:
            raise RuntimeError(
                "Ollama API path not found (404). Is the server running at "
                f"{OLLAMA_BASE}? Try `ollama serve` or set OLLAMA_BASE."
            )
        resp.raise_for_status()
    except Exception as e:
        return "ERROR", 0.0, f"Request failed: {e}"

    out = resp.json().get("response", "").strip()

    # Robust JSON parse with fallback
    try:
        data = json.loads(out)
        label = str(data.get("label", "Other"))
        confidence = float(data.get("confidence", 0.5))
    except Exception:
        # Heuristic fallback if not strict JSON
        label = "Other"
        confidence = 0.3
        low = out.lower()
        for L in label_list:
            if L.lower() in low:
                label, confidence = L, 0.6
                break

    return label, confidence, out


# Sidebar config
with st.sidebar:
    st.header("Model & Labels")
    model = st.text_input("Text model", value=DEFAULT_MODEL, help="e.g. llama3.1:8b, mistral:7b")
    labels_csv = st.text_input("Labels (comma-separated)", value="Billing,Technical,Sales,Spam,Other")
    label_list = [x.strip() for x in labels_csv.split(",") if x.strip()]

    st.caption("Optional: Add 1–3 few-shot examples to stabilize outputs.")
    with st.expander("Few-shot examples (optional)"):
        fs1_t = st.text_input("Shot 1 text", value="Invoice 4421 overdue, please remit payment.")
        fs1_l = st.text_input("Shot 1 label", value="Billing")
        fs2_t = st.text_input("Shot 2 text", value="The app crashes when I export a report.")
        fs2_l = st.text_input("Shot 2 label", value="Technical")
        fs3_t = st.text_input("Shot 3 text", value="Win a free prize! Click here now!!!")
        fs3_l = st.text_input("Shot 3 label", value="Spam")
    FEW_SHOTS = []
    if fs1_t and fs1_l: FEW_SHOTS.append({"text": fs1_t, "label": fs1_l})
    if fs2_t and fs2_l: FEW_SHOTS.append({"text": fs2_t, "label": fs2_l})
    if fs3_t and fs3_l: FEW_SHOTS.append({"text": fs3_t, "label": fs3_l})

tab1, tab2 = st.tabs(["🧪 Paste & Classify (single)", "📑 Bulk Paste → CSV"])

# ---------------------- Tab 1: single text ----------------------
with tab1:
    st.subheader("Paste a single text and get a classification")
    txt = st.text_area("Text to classify", height=180, placeholder="Paste your text here...")
    colA, colB = st.columns([1,1])
    with colA:
        run_single = st.button("Classify")
    with colB:
        clear_single = st.button("Clear")

    if clear_single:
        st.experimental_rerun()

    if run_single and txt.strip():
        label, conf, raw = classify_text(model, txt.strip(), label_list, FEW_SHOTS)
        st.success("Classification complete")
        st.markdown(f"**Label:** `{label}` &nbsp;&nbsp; **Confidence:** `{conf:.2f}`")
        with st.expander("Raw model output"):
            st.code(raw)


# ---------------------- Tab 2: manual label & save ----------------------
with tab2:
    st.subheader("Paste email to reeducate model")

    # Where to store labels
    csv_path = st.text_input("Output CSV path", value="labeled_manual.csv", help="File will be created if missing.")

    # Email text input
    email_text = st.text_area("Email text", height=240, placeholder="Paste the email body (or entire message) here...")

    # Pick a label from the sidebar-defined list
    chosen_label = st.selectbox("Choose label", options=label_list, index=0 if label_list else None)

    # (Optional) Add metadata fields if you want
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        subj = st.text_input("Subject (optional)", value="")
    with col_meta2:
        sender = st.text_input("From (optional)", value="")

    col_btn1, col_btn2 = st.columns([1,1])
    with col_btn1:
        save_btn = st.button("💾 Save item")
    with col_btn2:
        clear_btn = st.button("Clear form")

    if clear_btn:
        st.experimental_rerun()

    # Append to CSV
    if save_btn:
        if not email_text.strip():
            st.error("Please paste some text before saving.")
        elif not chosen_label:
            st.error("Please select a label.")
        else:
            import os, csv
            # ensure file exists with header
            new_file = not os.path.exists(csv_path)
            try:
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["text", "label", "subject", "from"])
                    if new_file:
                        writer.writeheader()
                    writer.writerow({
                        "text": email_text.strip(),
                        "label": chosen_label,
                        "subject": subj.strip(),
                        "from": sender.strip()
                    })
                st.success(f"Saved to {csv_path}")
            except Exception as e:
                st.error(f"Failed to save: {e}")

    # Show current CSV (if present)
    import os
    if os.path.exists(csv_path):
        try:
            df_curr = pd.read_csv(csv_path)
            st.caption("Current labeled items")
            st.dataframe(df_curr, use_container_width=True)

            # Download button
            st.download_button(
                "Download labeled_manual.csv",
                data=df_curr.to_csv(index=False).encode("utf-8"),
                file_name=os.path.basename(csv_path),
                mime="text/csv",
            )
        except Exception as e:
            st.warning(f"Could not read {csv_path}: {e}")
    else:
        st.info("No CSV yet. Save your first item to create it.")


# Footer help
st.markdown("---")
st.caption(
    "Tip: Make sure Ollama is running (`ollama serve`) and your model is pulled "
    "(e.g., `ollama pull llama3.1:8b`). Adjust labels in the sidebar."
)
