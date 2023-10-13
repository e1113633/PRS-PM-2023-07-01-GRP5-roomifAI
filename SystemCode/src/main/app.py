from collections import defaultdict
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
# from diffusers.loaders import LoraLoaderMixin
from safetensors.torch import load_file
import torch
import random
import importlib
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import io
from PIL import Image


app = Flask(__name__)
CORS(app)

loras = {
    "living": {
        "path": "./LoRA/roomifai_living.safetensors",
        "multiplier": 0.5
    },
    "dining": {
        "path": "./LoRA/roomifai_dining.safetensors",
        "multiplier": 0.5
    },
    "bedroom": {
        "path": "./LoRA/roomifai_bedroom.safetensors",
        "multiplier": 0.5
    }
}

def load_lora(pipe, room, device):
    lora = loras[room]
    path = lora['path']
    multiplier = lora['multiplier']

    imported_module = importlib.import_module("networks.lora")
    network, weights_sd = imported_module.create_network_from_weights(
        multiplier, path, vae=pipe.vae, text_encoder=pipe.text_encoder, unet=pipe.unet, for_inference=True
    )

    network.apply_to(pipe.text_encoder, pipe.unet)
    info = network.load_state_dict(weights_sd, False)
    print(f"loaded LoRA weights: {info}")

    return pipe



def load_sd(room):
    model_id = "runwayml/stable-diffusion-v1-5"
    dtype = torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # for key in loras.keys():
    pipe = load_lora(pipe, room)

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe

def generate_image(pipe, prompt):
    image = pipe(prompt).images[0]
    return image


@app.route("/generate", methods=["GET"])
def generate():
    prompt = request.args.get("prompt")
    room = request.args.get("prompt")

    # Load pipeline
    pipe = load_sd(room)

    # Generate pipeline
    image = generate_image(pipe, prompt)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes = image_bytes.getvalue()

    # return the response as an image
    return Response(image_bytes, mimetype="image/jpeg")