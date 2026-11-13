#install in google collab

!pip install -q transformers accelerate bitsandbytes pillow fastapi uvicorn pyngrok python-multipart

#start secure collab


import time
import secrets
import threading
import uvicorn
import nest_asyncio

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from pyngrok import ngrok
from PIL import Image
from io import BytesIO

nest_asyncio.apply()

# Create a private API key
API_KEY = secrets.token_urlsafe(32)

app = FastAPI(title="Llama Vision Colab API")

@app.get("/")
def home():
    return {"status": "Llama Vision API is running"}

@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    question: str = Form(...),
    max_new_tokens: int = Form(200),
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is empty")

    if not 1 <= max_new_tokens <= 500:
        raise HTTPException(status_code=400, detail="Invalid token limit")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        pil_image.thumbnail((1024, 1024))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]

        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        inputs = processor(
            pil_image,
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)

        eos_id = processor.tokenizer.eos_token_id
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_id

        start = time.time()

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_id,
                pad_token_id=pad_id
            )

        elapsed = round(time.time() - start, 2)

        generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
        answer = processor.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        return {
            "model": MODEL_ID,
            "answer": answer,
            "execution_time_seconds": elapsed,
            "device": str(model.device)
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# Enter your ngrok authentication token
ngrok_token = getpass("Enter your ngrok authtoken: ")
ngrok.set_auth_token(ngrok_token)

# Start API server
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

time.sleep(3)

public_url = ngrok.connect(8000).public_url

print("\nCOLAB SERVER READY")
print("Server URL:", public_url)
print("API KEY:", API_KEY)
print("\nKeep this Colab runtime running while using your local app.")

#vs terminal
pip install streamlit requests

#lauch

stramlit run app.py
use URL and API key generated from #start secure collab.