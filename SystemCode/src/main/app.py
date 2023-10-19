from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from networks.lora import create_network_from_weights
from networks.pipeline import PipelineLike
from safetensors.torch import load_file
import torch
from flask import Flask, Response, request, send_from_directory, jsonify
from flask_cors import CORS
import io
from PIL import Image

SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"

loras = {
    "living": {
        "file": "roomifai_living.safetensors",
        "multiplier": 0.5
    },
    "dining": {
        "file": "roomifai_dining.safetensors",
        "multiplier": 0.5
    },
    "bedroom": {
        "file": "roomifai_bedroom.safetensors",
        "multiplier": 0.5
    }
}

def load_lora(room, vae, text_encoder, unet, request_multiplier):
    lora = loras[room]
    file = lora['file']
    multiplier = request_multiplier if (type(request_multiplier) == int or type(request_multiplier) == float) else lora['multiplier']

    network, weights_sd = create_network_from_weights(
        multiplier, file, vae=vae, text_encoder=text_encoder, unet=unet, for_inference=True
    )

    network.apply_to(text_encoder, unet)
    info = network.load_state_dict(weights_sd, False)
    print(f"weights are loaded: (multiplier: {multiplier})")
    
    return network, weights_sd

def load_sd(room, multiplier):
    torch.cuda.empty_cache()
    model_path = "./checkpoints/pretrained_model/v1-5-pruned-emaonly.safetensors"
    dtype = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # base model
    scheduler = EulerAncestralDiscreteScheduler(num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE)
    base_pipe = StableDiffusionPipeline.from_single_file(
        model_path, torch_dtype=dtype, use_safetensors=True, extract_ema=True, scheduler=scheduler
    )

    # LoRA
    lora = loras[room]
    lora_file = lora['file']

    if lora_file:
        base_pipe.load_lora_weights(
            "./LoRA/", local_files_only=True, use_safetensors=True, weight_name=lora_file
        )
    base_pipe.enable_attention_slicing()
    base_pipe = base_pipe.to(device)
    return base_pipe

def generate_image(pipe, prompt, multiplier):
    image = pipe(
        prompt=prompt,
        output_type="pil",
        width=512,
        height=512,
        guidance_scale=7.5,
        # Multipliers not working, a bug in diffusers library
        # cross_attention_kwargs={"scale": multiplier}
    )[0][0]
    return image


app = Flask(__name__)
CORS(app)

# Path for our main Svelte page
@app.route("/")
def base():
    return send_from_directory('./frontend/public', 'index.html')

# Path for all the static files (compiled JS/CSS, etc.)
@app.route("/<path:path>")
def home(path):
    return send_from_directory('./frontend/public', path)

@app.route("/api/generate", methods=["GET"])
def generate():
    try:
        prompt = request.args.get("prompt")
        room = request.args.get("room")
        multiplier = float(request.args.get("multiplier")) if request.args.get("multiplier") else None
        
        # LoRA
        lora = loras[room]
        lora_def_multiplier = lora['multiplier']
        multiplier = multiplier if (type(multiplier) == int or type(multiplier) == float) else lora_def_multiplier

        # Load pipeline
        pipe = load_sd(room, multiplier)

        # Generate pipeline
        image = generate_image(pipe, prompt, multiplier=multiplier)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes = image_bytes.getvalue()

        # return the response as an image
        return Response(image_bytes, mimetype="image/jpeg")
    except Exception as error:
        torch.cuda.empty_cache()
        print("An exception occurred:", error)
        return Response("An exception occurred:" + error)
