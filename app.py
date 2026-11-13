"""
Local Streamlit frontend for Llama 3.2 Vision running on Google Colab.

Run from the project root:
    streamlit run app.py

Start the Colab backend first, then paste its URL and API key in the sidebar.
"""

import json
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="LMM Vision Assistant",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ LMM Vision Assistant")
st.caption("Local interface • Google Colab GPU inference • Meta Llama 3.2 Vision")

with st.sidebar:
    st.header("Colab connection")
    api_url = st.text_input(
        "Colab server URL",
        placeholder="https://your-ngrok-domain.ngrok-free.app",
        help="Copy the public URL printed by your running Google Colab notebook.",
    ).strip().rstrip("/")
    api_key = st.text_input(
        "API key",
        type="password",
        help="Paste the API key printed once by your Colab backend.",
    )
    st.divider()
    st.caption("Keep the Colab notebook running while using this app.")

st.subheader("1. Upload an image")
uploaded = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "webp"],
)

tasks = {
    "Custom question": "",
    "Image caption": "Describe this image clearly and accurately.",
    "Object recognition": "Identify the main objects and people visible in this image.",
    "Scene understanding": "Describe the scene, activities, and surroundings.",
    "Visual question answering": "What is happening in this image? Explain using visible evidence.",
    "OCR / text extraction": "Read and transcribe all clearly visible text in this image. Mark unclear text as [unclear].",
    "Visual reasoning": (
        "Describe what is directly visible, explain relationships, and give cautious "
        "inferences supported by visual evidence. State uncertainties."
    ),
}

left, right = st.columns([1, 1.2], gap="large")

with left:
    if uploaded:
        st.image(uploaded, caption=uploaded.name, use_container_width=True)
    else:
        st.info("Upload an image to preview it here.")

with right:
    st.subheader("2. Choose an analysis task")
    task = st.selectbox("Task", list(tasks.keys()))
    question = st.text_area(
        "Question / instruction",
        value=tasks[task],
        height=130,
        placeholder="Enter a question about the image...",
    )
    max_tokens = st.slider("Maximum response tokens", 50, 500, 200, 50)

    analyze = st.button(
        "✨ Analyze using Colab GPU",
        type="primary",
        use_container_width=True,
        disabled=not (uploaded and question.strip() and api_url and api_key),
    )

if analyze:
    endpoint = f"{api_url}/analyze"
    headers = {"X-API-Key": api_key}
    files = {
        "image": (
            uploaded.name,
            uploaded.getvalue(),
            uploaded.type or "application/octet-stream",
        )
    }
    data = {
        "question": question.strip(),
        "max_new_tokens": str(max_tokens),
    }

    try:
        with st.spinner("Sending image to Google Colab GPU and waiting for the model..."):
            response = requests.post(
                endpoint,
                headers=headers,
                files=files,
                data=data,
                timeout=(30, 900),
            )
        response.raise_for_status()
        result = response.json()

        record = {
            "experiment": "Interactive LMM image analysis",
            "model": result.get("model", "Meta Llama 3.2 11B Vision Instruct"),
            "input_image": uploaded.name,
            "question": question.strip(),
            "answer": result["answer"],
            "execution_time_seconds": result.get("execution_time_seconds"),
            "inference_device": result.get("device"),
            "timestamp": datetime.now().astimezone().isoformat(),
        }

        filename = f"colab_app_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = RESULTS_DIR / filename
        output_path.write_text(
            json.dumps(record, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )

        st.success("Analysis completed on Google Colab!")
        st.subheader("3. Model response")
        st.markdown(record["answer"])
        st.caption(
            f"Colab inference time: {record['execution_time_seconds']} seconds"
            + (f" · Device: {record['inference_device']}" if record["inference_device"] else "")
        )
        st.success(f"Saved locally: results/{filename}")

        st.download_button(
            "Download JSON result",
            data=json.dumps(record, indent=4, ensure_ascii=False),
            file_name=filename,
            mime="application/json",
        )

    except requests.exceptions.Timeout:
        st.error("The Colab request timed out. Check that the Colab runtime is still running.")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to Colab. Check the server URL and make sure the Colab cell is still running.")
    except requests.exceptions.HTTPError:
        st.error(f"Colab returned an error ({response.status_code}): {response.text}")
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

st.divider()
st.caption("The model runs in Colab. This local app.py sends the image and saves returned results on your PC.")
