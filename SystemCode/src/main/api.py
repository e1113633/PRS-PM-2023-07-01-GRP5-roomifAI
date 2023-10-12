from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import random
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import io
from PIL import Image


app = Flask(__name__)
CORS(app)


model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = pipe.to(device)

pipe.enable_attention_slicing()

def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image



@app.route("/generate", methods=["GET"])
def generate():
    prompt = request.args.get("prompt")
    image = generate_image(prompt)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes = image_bytes.getvalue()

    # return the response as an image
    return Response(image_bytes, mimetype="image/jpeg")