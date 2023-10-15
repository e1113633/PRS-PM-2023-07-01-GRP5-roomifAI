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

def load_lora(room, vae, text_encoder, unet, request_multiplier):
    lora = loras[room]
    path = lora['path']
    multiplier = request_multiplier if (type(request_multiplier) == int or type(request_multiplier) == float) else lora['multiplier']

    network, weights_sd = create_network_from_weights(
        multiplier, path, vae=vae, text_encoder=text_encoder, unet=unet, for_inference=True
    )

    info = network.load_state_dict(weights_sd, False)
    network.apply_to(text_encoder, unet)
    info = network.load_state_dict(weights_sd, False)
    print(f"weights are loaded: {info} (multiplier: {multiplier})")
    
    return network, weights_sd

def load_sd(room, multiplier):
    torch.cuda.empty_cache()
    model_id = "runwayml/stable-diffusion-v1-5"
    # model_path = "./checkpoints/pretrained_model/v1-5-pruned-emaonly.safetensors"
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

    vae.to(dtype).to(device)
    text_encoder.to(dtype).to(device)
    unet.to(dtype).to(device)

    network, weights_sd = load_lora(room, vae, text_encoder, unet, multiplier)
    network.to(dtype).to(device)

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
        output_type="pil",
        width=512,
        height=512,
        guidance_scale=1.0
    )[0][0]
    return image


app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def health():
    return "Healthy!"

@app.route("/api/generate", methods=["GET"])
def generate():
    try:
        prompt = request.args.get("prompt")
        room = request.args.get("room")
        multiplier = float(request.args.get("multiplier")) if request.args.get("multiplier") else None

        # Load pipeline
        pipe = load_sd(room, multiplier)

        # Generate pipeline
        image = generate_image(pipe, prompt)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes = image_bytes.getvalue()

        # return the response as an image
        return Response(image_bytes, mimetype="image/jpeg")
    except Exception as error:
        torch.cuda.empty_cache()
        print("An exception occurred:", error)
        return Response("An exception occurred")
