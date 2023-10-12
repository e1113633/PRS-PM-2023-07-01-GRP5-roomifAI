from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
# from diffusers.loaders import LoraLoaderMixin
import torch
import random
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import io
from PIL import Image


app = Flask(__name__)
CORS(app)

loras = {
    "living": "./LoRA/roomifai_living.safetensors",
    "dining": "./LoRA/roomifai_dining.safetensors",
    "bedroom": "./LoRA/roomifai_bedroom.safetensors",
}

def load_lora(pipe, room):
    model_path = loras[room]
    pipe.unet.load_attn_procs(model_path, local_files_only=True)
    # pipe.unet.load_lora_weights(model_path, local_files_only=True)
    return pipe

def load_sd():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # for key in loras.keys():
    #     pipe = load_lora(pipe, key)

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe

pipe = load_sd()

def generate_image(pipe, prompt):
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