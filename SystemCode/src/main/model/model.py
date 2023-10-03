import math
import os
import torch
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from utils.conversions import convert_ldm_vae_checkpoint, convert_ldm_clip_checkpoint_v1, convert_vae_state_dict
from safetensors.torch import load_file, save_file

# Utils
from utils.conversions import convert_unet_state_dict_to_sd, convert_text_encoder_key, convert_ldm_unet_checkpoint
from utils.common import get_file_type

UNET_PARAMS_MODEL_CHANNELS = 320
UNET_PARAMS_CHANNEL_MULTI = [1, 2, 4, 4]
UNET_PARAMS_ATTENTION_RESOLUTIONS = [4, 2, 1]
UNET_PARAMS_IMAGE_SIZE = 64 
UNET_PARAMS_IN_CHANNELS = 4
UNET_PARAMS_OUT_CHANNELS = 4
UNET_PARAMS_NUM_RES_BLOCKS = 2
UNET_PARAMS_CONTEXT_DIM = 768
UNET_PARAMS_NUM_HEADS = 8

VAE_PARAMS_Z_CHANNELS = 4
VAE_PARAMS_RESOLUTION = 256
VAE_PARAMS_IN_CHANNELS = 3
VAE_PARAMS_OUT_CH = 3
VAE_PARAMS_CH = 128
VAE_PARAMS_CH_MULT = [1, 2, 4, 4]
VAE_PARAMS_NUM_RES_BLOCKS = 2

MODEL_PATH = "checkpoints/v1-5-pruned-emaonly.safetensors"


def create_unet_diffusers_config():
    """
    Creates a config for the diffusers based on the config of the LDM model.
    Uses diffusers' UNet2DModel. Documentation: https://huggingface.co/docs/diffusers/api/models/unet2d
    """

    block_out_channels = [UNET_PARAMS_MODEL_CHANNELS * mult for mult in UNET_PARAMS_CHANNEL_MULTI]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnDownBlock2D" if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS else "DownBlock2D"
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnUpBlock2D" if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS else "UpBlock2D"
        up_block_types.append(block_type)
        resolution //= 2

    config = dict(
        sample_size=UNET_PARAMS_IMAGE_SIZE,
        in_channels=UNET_PARAMS_IN_CHANNELS,
        out_channels=UNET_PARAMS_OUT_CHANNELS,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        layers_per_block=UNET_PARAMS_NUM_RES_BLOCKS,
        cross_attention_dim=UNET_PARAMS_CONTEXT_DIM,
        attention_head_dim=UNET_PARAMS_NUM_HEADS,
    )

    return config


def create_vae_diffusers_config():
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    block_out_channels = [VAE_PARAMS_CH * mult for mult in VAE_PARAMS_CH_MULT]
    down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
    up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

    config = dict(
        sample_size=VAE_PARAMS_RESOLUTION,
        in_channels=VAE_PARAMS_IN_CHANNELS,
        out_channels=VAE_PARAMS_OUT_CH,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        latent_channels=VAE_PARAMS_Z_CHANNELS,
        layers_per_block=VAE_PARAMS_NUM_RES_BLOCKS,
    )
    return config


def is_safetensors(path):
    return get_file_type(path) == ".safetensors"

def load_checkpoint(checkpoint_path, device):
    if is_safetensors(checkpoint_path):
        checkpoint = None
        state_dict = load_file(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            checkpoint = None
    return checkpoint, state_dict



def load_models_from_stable_diffusion_checkpoint(checkpoint_path, device="cpu", dtype=None):
    _, state_dict = load_checkpoint(checkpoint_path, device)
    state_dict = convert_text_encoder_key(state_dict)

    # UNet model.
    unet_config = create_unet_diffusers_config()
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(state_dict, unet_config)

    unet = UNet2DConditionModel(**unet_config).to(device)
    info = unet.load_state_dict(converted_unet_checkpoint)
    print("load u-net:", info)

    # VAE model.
    vae_config = create_vae_diffusers_config()
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config)

    vae = AutoencoderKL(**vae_config).to(device)
    info = vae.load_state_dict(converted_vae_checkpoint)
    print("load vae:", info)

    converted_text_encoder_checkpoint = convert_ldm_clip_checkpoint_v1(state_dict)

    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    logging.set_verbosity_warning()

    info = text_model.load_state_dict(converted_text_encoder_checkpoint)
    print("load text encoder:", info)

    return text_model, vae, unet



def save_stable_diffusion_checkpoint(output_file, text_encoder, unet, checkpoint_path, epochs, steps, save_dtype=None, vae=None):
    if checkpoint_path is not None:
        checkpoint, state_dict = load_checkpoint(checkpoint_path)
        state_dict = convert_text_encoder_key(state_dict)

        if checkpoint is None:
            checkpoint = {}
            strict = False
        else:
            strict = True
        if "state_dict" in state_dict:
            del state_dict["state_dict"]
    else:
        assert vae is not None, "VAE is required to save a checkpoint without a given checkpoint"
        checkpoint = {}
        state_dict = {}
        strict = False

    def update_sd(prefix, sd):
        for k, v in sd.items():
            key = prefix + k
            assert not strict or key in state_dict, f"Illegal key in save SD: {key}"
            if save_dtype is not None:
                v = v.detach().clone().to("cpu").to(save_dtype)
            state_dict[key] = v

    # Reconvert the checkpoint from diffusers format to Stable Diffusion LDM format
    # Convert UNet
    unet_state_dict = convert_unet_state_dict_to_sd(unet.state_dict())
    update_sd("model.diffusion_model.", unet_state_dict)

    text_enc_dict = text_encoder.state_dict()
    update_sd("cond_stage_model.transformer.", text_enc_dict)

    # Convert VAE
    if vae is not None:
        vae_dict = convert_vae_state_dict(vae.state_dict())
        update_sd("first_stage_model.", vae_dict)

    # Create new checkpoint
    key_count = len(state_dict.keys())
    new_ckpt = {"state_dict": state_dict}

    try:
        if "epoch" in checkpoint:
            epochs += checkpoint["epoch"]
        if "global_step" in checkpoint:
            steps += checkpoint["global_step"]
    except:
        pass

    new_ckpt["epoch"] = epochs
    new_ckpt["global_step"] = steps

    if is_safetensors(output_file):
        save_file(state_dict, output_file)
    else:
        torch.save(new_ckpt, output_file)

    return key_count


def save_diffusers_checkpoint(output_dir, text_encoder, unet, pretrained_model_name_or_path, vae=None, use_safetensors=False):
    if pretrained_model_name_or_path is None:
        pretrained_model_name_or_path = MODEL_PATH

    scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    if vae is None:
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")

    pipeline = StableDiffusionPipeline(
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        tokenizer=tokenizer,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=None,
    )
    pipeline.save_pretrained(output_dir, safe_serialization=use_safetensors)


VAE_PREFIX = "first_stage_model."


def load_vae(vae_id, dtype):
    print(f"load VAE: {vae_id}")
    if os.path.isdir(vae_id) or not os.path.isfile(vae_id):
        # Diffusers local/remote
        try:
            vae = AutoencoderKL.from_pretrained(vae_id, subfolder=None, torch_dtype=dtype)
        except EnvironmentError as e:
            print(f"exception occurs in loading vae: {e}")
            print("retry with subfolder='vae'")
            vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae", torch_dtype=dtype)
        return vae

    # local
    vae_config = create_vae_diffusers_config()

    if vae_id.endswith(".bin"):
        # SD 1.5 VAE on Huggingface
        converted_vae_checkpoint = torch.load(vae_id, map_location="cpu")
    else:
        # StableDiffusion
        vae_model = load_file(vae_id, "cpu") if is_safetensors(vae_id) else torch.load(vae_id, map_location="cpu")
        vae_sd = vae_model["state_dict"] if "state_dict" in vae_model else vae_model

        # vae only or full model
        full_model = False
        for vae_key in vae_sd:
            if vae_key.startswith(VAE_PREFIX):
                full_model = True
                break
        if not full_model:
            sd = {}
            for key, value in vae_sd.items():
                sd[VAE_PREFIX + key] = value
            vae_sd = sd
            del sd

        # Convert the VAE model.
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(vae_sd, vae_config)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)
    return vae


def make_bucket_resolutions(max_reso, min_size=256, max_size=512, divisible=64):
    max_width, max_height = max_reso
    max_area = (max_width // divisible) * (max_height // divisible)

    resos = set()

    size = int(math.sqrt(max_area)) * divisible
    resos.add((size, size))

    size = min_size
    while size <= max_size:
        width = size
        height = min(max_size, (max_area // (width // divisible)) * divisible)
        resos.add((width, height))
        resos.add((height, width))
        size += divisible

    resos = list(resos)
    resos.sort()
    return resos


if __name__ == "__main__":
    resos = make_bucket_resolutions((512, 512))
    print(len(resos))
    print(resos)
    aspect_ratios = [w / h for w, h in resos]
    print(aspect_ratios)

    ars = set()
    for ar in aspect_ratios:
        if ar in ars:
            print("error! duplicate ar:", ar)
        ars.add(ar)
