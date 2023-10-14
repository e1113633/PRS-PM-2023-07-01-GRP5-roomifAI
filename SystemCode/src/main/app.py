from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from networks.lora import create_network_from_weights
from networks.pipeline import PipelineLike
from safetensors.torch import load_file
import torch
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import io
from PIL import Image

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

torch.cuda.empty_cache()

def load_lora(room, vae, text_encoder, unet):
    lora = loras[room]
    path = lora['path']
    multiplier = lora['multiplier']

    network, weights_sd = create_network_from_weights(
        multiplier, path, vae=vae, text_encoder=text_encoder, unet=unet, for_inference=True
    )

    info = network.load_state_dict(weights_sd, False)
    print(f"loaded LoRA weights: {room} {info}")
    network.apply_to(text_encoder, unet)
    info = network.load_state_dict(weights_sd, False)
    print(f"weights are loaded: {info}")
    
    return network, weights_sd

def load_sd(room):
    model_id = "runwayml/stable-diffusion-v1-5"
    dtype = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # base model
    base_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    text_encoder = base_pipe.text_encoder
    vae = base_pipe.vae
    unet = base_pipe.unet
    tokenizer = base_pipe.tokenizer
    scheduler = EulerAncestralDiscreteScheduler.from_config(base_pipe.scheduler.config)
    del base_pipe

    networks = []
    network_default_muls = []
    for key in loras.keys():
        lora_params = loras[key]
        network, weights_sd = load_lora(key, vae, text_encoder, unet)
        network.to(dtype).to(device)
        networks.append(network)
        network_default_muls.append(lora_params['multiplier'])

    if networks:
        for n, m in zip(networks, network_default_muls):
            n.set_multiplier(m)

    pipe = PipelineLike(
        device,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        clip_skip=None
    )
    return pipe

def generate_image(pipe, prompt):
    image = pipe(
        prompt=prompt,
        output_type="pil"
    )[0]
    return image


app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def health():
    return "Healthy!"

@app.route("/generate", methods=["GET"])
def generate():
    prompt = request.args.get("prompt")
    room = request.args.get("room")

    # Load pipeline
    pipe = load_sd(room)

    # Generate pipeline
    image = generate_image(pipe, prompt)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes = image_bytes.getvalue()

    # return the response as an image
    return Response(image_bytes, mimetype="image/jpeg")