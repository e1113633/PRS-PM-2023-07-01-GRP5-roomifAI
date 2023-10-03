from typing import Any, List, NamedTuple, Optional, Tuple, Union, Callable
import glob
import importlib
import time

import argparse
import math
import os
import random

import diffusers
import numpy as np
import torch
import torchvision
from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
)

from transformers import CLIPModel
import PIL
from PIL import Image
from PIL.PngImagePlugin import PngInfo


from attention import FlashAttentionFunction, replace_unet_modules
import model as model_util
import train as train_util
from pipeline import PipelineLike
import tools.original_control_net as original_control_net
from tools.original_control_net import ControlNetInfo

MODEL_PATH = "runwayml/stable-diffusion-v1-5"
TOKENIZER_PATH = "openai/clip-vit-large-patch14"
DEFAULT_TOKEN_LENGTH = 75

# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"
LATENT_CHANNELS = 4
DOWNSAMPLING_FACTOR = 8

# CLIP
CLIP_MODEL_PATH = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
FEATURE_EXTRACTOR_SIZE = (224, 224)
FEATURE_EXTRACTOR_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
FEATURE_EXTRACTOR_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]

VGG16_IMAGE_MEAN = [0.485, 0.456, 0.406]
VGG16_IMAGE_STD = [0.229, 0.224, 0.225]
VGG16_INPUT_RESIZE_DIV = 4
NUM_CUTOUTS = 4
USE_CUTOUTS = False


# VGG16 input can be any size, so resize the input image appropriately
def preprocess_vgg16_guide_image(image, size):
    image = image.resize(size, resample=Image.NEAREST)  # Combine with cond_fn
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)  # nchw
    image = torch.from_numpy(image)
    return image  # 0 to 1


def preprocess_mask(mask):
    mask = mask.convert("L")
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    mask = mask.resize((w // 8, h // 8), resample=PIL.Image.BILINEAR)  # LANCZOS)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    mask = 1 - mask  # repaint white, keep black
    mask = torch.from_numpy(mask)
    return mask

class BatchDataBase(NamedTuple):
    step: int
    prompt: str
    seed: int
    init_image: Any
    mask_image: Any
    clip_prompt: str
    guide_image: Any


class BatchDataExt(NamedTuple):
    # Data requiring batch splitting
    width: int
    height: int
    steps: int
    scale: float
    negative_scale: float
    strength: float
    network_muls: Tuple[float]
    num_sub_prompts: int


class BatchData(NamedTuple):
    return_latents: bool
    base: BatchDataBase
    ext: BatchDataExt


def main(args):
    if args.fp16:
        dtype = torch.float16
    elif args.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    checkpoint_file = args.ckpt if args.ckpt else MODEL_PATH
    print("args.ckpt :", args.ckpt, "checkpoint_file:", checkpoint_file)
    if not os.path.isfile(checkpoint_file):
        files = glob.glob(checkpoint_file)
        if len(files) == 1:
            checkpoint_file = files[0]

    use_stable_diffusion_format = os.path.isfile(checkpoint_file)
    if use_stable_diffusion_format:
        print("load StableDiffusion checkpoint")
        text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(checkpoint_file)
    else:
        print("load Diffusers pretrained models")
        loading_pipe = DiffusionPipeline.from_pretrained(checkpoint_file, safety_checker=None, torch_dtype=dtype)
        text_encoder = loading_pipe.text_encoder
        vae = loading_pipe.vae
        unet = loading_pipe.unet
        tokenizer = loading_pipe.tokenizer
        del loading_pipe

    if args.vae is not None:
        vae = model_util.load_vae(args.vae, dtype)
        print("additional VAE loaded")

    if args.clip_guidance_scale > 0.0 or args.clip_image_guidance_scale:
        print("prepare clip model")
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH, torch_dtype=dtype)
    else:
        clip_model = None

    if args.vgg16_guidance_scale > 0.0:
        print("prepare resnet model")
        vgg16_model = torchvision.models.vgg16(torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    else:
        vgg16_model = None
    
    # xformers、Hypernetwork (for Linux, not Windows)
    if not args.diffusers_xformers:
        replace_unet_modules(unet, not args.xformers, args.xformers)

    print("loading tokenizer")
    if use_stable_diffusion_format:
        tokenizer = train_util.load_tokenizer(args)

    sched_init_args = {}
    scheduler_num_noises_per_step = 1
    if args.sampler == "euler" or args.sampler == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_euler_discrete
    elif args.sampler == "euler_a" or args.sampler == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_euler_ancestral_discrete
    else:
        scheduler_cls = EulerAncestralDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_euler_ancestral_discrete

    if args.v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction"

    # replace randn
    class NoiseManager:
        def __init__(self):
            self.sampler_noises = None
            self.sampler_noise_index = 0

        def reset_sampler_noises(self, noises):
            self.sampler_noise_index = 0
            self.sampler_noises = noises

        def randn(self, shape, device=None, dtype=None, layout=None, generator=None):
            # print("replacing", shape, len(self.sampler_noises), self.sampler_noise_index)
            if self.sampler_noises is not None and self.sampler_noise_index < len(self.sampler_noises):
                noise = self.sampler_noises[self.sampler_noise_index]
                if shape != noise.shape:
                    noise = None
            else:
                noise = None

            if noise == None:
                print(f"unexpected noise request: {self.sampler_noise_index}, {shape}")
                noise = torch.randn(shape, dtype=dtype, device=device, generator=generator)

            self.sampler_noise_index += 1
            return noise

    class TorchRandReplacer:
        def __init__(self, noise_manager):
            self.noise_manager = noise_manager

        def __getattr__(self, item):
            if item == "randn":
                return self.noise_manager.randn
            if hasattr(torch, item):
                return getattr(torch, item)
            raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))

    noise_manager = NoiseManager()
    if scheduler_module is not None:
        scheduler_module.torch = TorchRandReplacer(noise_manager)

    scheduler = scheduler_cls(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
        **sched_init_args,
    )

    if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
        print("set clip_sample to True")
        scheduler.config.clip_sample = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # copy custom pipeline
    vae.to(dtype).to(device)
    text_encoder.to(dtype).to(device)
    unet.to(dtype).to(device)
    if clip_model is not None:
        clip_model.to(dtype).to(device)
    if vgg16_model is not None:
        vgg16_model.to(dtype).to(device)

    # import network modules
    if args.network_module:
        networks = []
        network_default_muls = []
        for i, network_module in enumerate(args.network_module):
            print("import network module:", network_module)
            imported_module = importlib.import_module(network_module)

            network_mul = 1.0 if args.network_mul is None or len(args.network_mul) <= i else args.network_mul[i]
            network_default_muls.append(network_mul)

            net_kwargs = {}
            if args.network_args and i < len(args.network_args):
                network_args = args.network_args[i]
                network_args = network_args.split(";")
                for net_arg in network_args:
                    key, value = net_arg.split("=")
                    net_kwargs[key] = value

            if args.network_weights and i < len(args.network_weights):
                network_weight = args.network_weights[i]
                print("load network weights from:", network_weight)

                if model_util.is_safetensors(network_weight) and args.network_show_meta:
                    from safetensors.torch import safe_open

                    with safe_open(network_weight, framework="pt") as f:
                        metadata = f.metadata()
                    if metadata is not None:
                        print(f"metadata for: {network_weight}: {metadata}")

                network, weights_sd = imported_module.create_network_from_weights(
                    network_mul, network_weight, vae, text_encoder, unet, for_inference=True, **net_kwargs
                )
            else:
                raise ValueError("No weight. Weight is required.")
            if network is None:
                return

            mergeable = hasattr(network, "merge_to")
            if args.network_merge and not mergeable:
                print("network is not mergeable. ignore merge option.")

            if not args.network_merge or not mergeable:
                network.apply_to(text_encoder, unet)
                info = network.load_state_dict(weights_sd, False)
                print(f"weights are loaded: {info}")

                if args.opt_channels_last:
                    network.to(memory_format=torch.channels_last)
                network.to(dtype).to(device)

                networks.append(network)
            else:
                network.merge_to(text_encoder, unet, weights_sd, dtype, device)

    else:
        networks = []

    # ControlNet
    control_nets: List[ControlNetInfo] = []
    if args.control_net_models:
        for i, model in enumerate(args.control_net_models):
            prep_type = None if not args.control_net_preps or len(args.control_net_preps) <= i else args.control_net_preps[i]
            weight = 1.0 if not args.control_net_weights or len(args.control_net_weights) <= i else args.control_net_weights[i]
            ratio = 1.0 if not args.control_net_ratios or len(args.control_net_ratios) <= i else args.control_net_ratios[i]

            ctrl_unet, ctrl_net = original_control_net.load_control_net(args.v2, unet, model)
            prep = original_control_net.load_preprocess(prep_type)
            control_nets.append(ControlNetInfo(ctrl_unet, ctrl_net, prep, weight, ratio))

    if args.opt_channels_last:
        print(f"set optimizing: channels last")
        text_encoder.to(memory_format=torch.channels_last)
        vae.to(memory_format=torch.channels_last)
        unet.to(memory_format=torch.channels_last)
        if clip_model is not None:
            clip_model.to(memory_format=torch.channels_last)
        if networks:
            for network in networks:
                network.to(memory_format=torch.channels_last)
        if vgg16_model is not None:
            vgg16_model.to(memory_format=torch.channels_last)

        for cn in control_nets:
            cn.unet.to(memory_format=torch.channels_last)
            cn.net.to(memory_format=torch.channels_last)

    pipe = PipelineLike(
        device,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        args.clip_skip,
        clip_model,
        args.clip_guidance_scale,
        args.clip_image_guidance_scale,
        vgg16_model,
        args.vgg16_guidance_scale,
        args.vgg16_guidance_layer,
    )
    pipe.set_control_nets(control_nets)
    print("pipeline is ready.")

    if args.diffusers_xformers:
        pipe.enable_xformers_memory_efficient_attention()

    if args.textual_inversion_embeddings:
        token_ids_embeds = []
        for embeds_file in args.textual_inversion_embeddings:
            if model_util.is_safetensors(embeds_file):
                from safetensors.torch import load_file

                data = load_file(embeds_file)
            else:
                data = torch.load(embeds_file, map_location="cpu")

            if "string_to_param" in data:
                data = data["string_to_param"]
            embeds = next(iter(data.values()))

            if type(embeds) != torch.Tensor:
                raise ValueError(f"weight file does not contains Tensor")

            num_vectors_per_token = embeds.size()[0]
            token_string = os.path.splitext(os.path.basename(embeds_file))[0]
            token_strings = [token_string] + [f"{token_string}{i+1}" for i in range(num_vectors_per_token - 1)]

            # add new word to tokenizer, count is num_vectors_per_token
            num_added_tokens = tokenizer.add_tokens(token_strings)
            assert (
                num_added_tokens == num_vectors_per_token
            ), f"tokenizer has same word to token string (filename). please rename the file"

            token_ids = tokenizer.convert_tokens_to_ids(token_strings)
            print(f"Textual Inversion embeddings `{token_string}` loaded. Tokens are added: {token_ids}")
            assert (
                min(token_ids) == token_ids[0] and token_ids[-1] == token_ids[0] + len(token_ids) - 1
            ), f"token ids is not ordered"
            assert len(tokenizer) - 1 == token_ids[-1], f"token ids is not end of tokenize: {len(tokenizer)}"

            if num_vectors_per_token > 1:
                pipe.add_token_replacement(token_ids[0], token_ids)

            token_ids_embeds.append((token_ids, embeds))

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for token_ids, embeds in token_ids_embeds:
            for token_id, embed in zip(token_ids, embeds):
                token_embeds[token_id] = embed

    if args.XTI_embeddings:
        XTI_layers = [
            "IN01",
            "IN02",
            "IN04",
            "IN05",
            "IN07",
            "IN08",
            "MID",
            "OUT03",
            "OUT04",
            "OUT05",
            "OUT06",
            "OUT07",
            "OUT08",
            "OUT09",
            "OUT10",
            "OUT11",
        ]
        token_ids_embeds_XTI = []
        for embeds_file in args.XTI_embeddings:
            if model_util.is_safetensors(embeds_file):
                from safetensors.torch import load_file

                data = load_file(embeds_file)
            else:
                data = torch.load(embeds_file, map_location="cpu")
            if set(data.keys()) != set(XTI_layers):
                raise ValueError("NOT XTI")
            embeds = torch.concat(list(data.values()))
            num_vectors_per_token = data["MID"].size()[0]

            token_string = os.path.splitext(os.path.basename(embeds_file))[0]
            token_strings = [token_string] + [f"{token_string}{i+1}" for i in range(num_vectors_per_token - 1)]

            # add new word to tokenizer, count is num_vectors_per_token
            num_added_tokens = tokenizer.add_tokens(token_strings)
            assert (
                num_added_tokens == num_vectors_per_token
            ), f"tokenizer has same word to token string (filename). please rename the file"

            token_ids = tokenizer.convert_tokens_to_ids(token_strings)
            print(f"XTI embeddings `{token_string}` loaded. Tokens are added: {token_ids}")

            # if num_vectors_per_token > 1:
            pipe.add_token_replacement(token_ids[0], token_ids)

            token_strings_XTI = []
            for layer_name in XTI_layers:
                token_strings_XTI += [f"{t}_{layer_name}" for t in token_strings]
            tokenizer.add_tokens(token_strings_XTI)
            token_ids_XTI = tokenizer.convert_tokens_to_ids(token_strings_XTI)
            token_ids_embeds_XTI.append((token_ids_XTI, embeds))
            for t in token_ids:
                t_XTI_dic = {}
                for i, layer_name in enumerate(XTI_layers):
                    t_XTI_dic[layer_name] = t + (i + 1) * num_added_tokens
                pipe.add_token_replacement_XTI(t, t_XTI_dic)

            text_encoder.resize_token_embeddings(len(tokenizer))
            token_embeds = text_encoder.get_input_embeddings().weight.data
            for token_ids, embeds in token_ids_embeds_XTI:
                for token_id, embed in zip(token_ids, embeds):
                    token_embeds[token_id] = embed

    # promptを取得する
    if args.from_file is not None:
        print(f"reading prompts from {args.from_file}")
        with open(args.from_file, "r", encoding="utf-8") as f:
            prompt_list = f.read().splitlines()
            prompt_list = [d for d in prompt_list if len(d.strip()) > 0]
    elif args.prompt is not None:
        prompt_list = [args.prompt]
    else:
        prompt_list = []

    if args.interactive:
        args.n_iter = 1

    # img2imgの前処理、画像の読み込みなど
    def load_images(path):
        if os.path.isfile(path):
            paths = [path]
        else:
            paths = (
                glob.glob(os.path.join(path, "*.png"))
                + glob.glob(os.path.join(path, "*.jpg"))
                + glob.glob(os.path.join(path, "*.jpeg"))
                + glob.glob(os.path.join(path, "*.webp"))
            )
            paths.sort()

        images = []
        for p in paths:
            image = Image.open(p)
            if image.mode != "RGB":
                print(f"convert image to RGB from {image.mode}: {p}")
                image = image.convert("RGB")
            images.append(image)

        return images

    def resize_images(imgs, size):
        resized = []
        for img in imgs:
            r_img = img.resize(size, Image.Resampling.LANCZOS)
            if hasattr(img, "filename"):  # filename属性がない場合があるらしい
                r_img.filename = img.filename
            resized.append(r_img)
        return resized

    if args.image_path is not None:
        print(f"load image for img2img: {args.image_path}")
        init_images = load_images(args.image_path)
        assert len(init_images) > 0, f"No image"
        print(f"loaded {len(init_images)} images for img2img")
    else:
        init_images = None

    if args.mask_path is not None:
        print(f"load mask for inpainting: {args.mask_path}")
        mask_images = load_images(args.mask_path)
        assert len(mask_images) > 0, f"No mask image"
        print(f"loaded {len(mask_images)} mask images for inpainting")
    else:
        mask_images = None

    # promptがないとき、画像のPngInfoから取得する
    if init_images is not None and len(prompt_list) == 0 and not args.interactive:
        print("get prompts from images' meta data")
        for img in init_images:
            if "prompt" in img.text:
                prompt = img.text["prompt"]
                if "negative-prompt" in img.text:
                    prompt += " --n " + img.text["negative-prompt"]
                prompt_list.append(prompt)

        # プロンプトと画像を一致させるため指定回数だけ繰り返す（画像を増幅する）
        l = []
        for im in init_images:
            l.extend([im] * args.images_per_prompt)
        init_images = l

        if mask_images is not None:
            l = []
            for im in mask_images:
                l.extend([im] * args.images_per_prompt)
            mask_images = l

    # 画像サイズにオプション指定があるときはリサイズする
    if args.W is not None and args.H is not None:
        if init_images is not None:
            print(f"resize img2img source images to {args.W}*{args.H}")
            init_images = resize_images(init_images, (args.W, args.H))
        if mask_images is not None:
            print(f"resize img2img mask images to {args.W}*{args.H}")
            mask_images = resize_images(mask_images, (args.W, args.H))

    regional_network = False
    if networks and mask_images:
        # mask を領域情報として流用する、現在は一回のコマンド呼び出しで1枚だけ対応
        regional_network = True
        print("use mask as region")

        size = None
        for i, network in enumerate(networks):
            if i < 3:
                np_mask = np.array(mask_images[0])
                np_mask = np_mask[:, :, i]
                size = np_mask.shape
            else:
                np_mask = np.full(size, 255, dtype=np.uint8)    
            mask = torch.from_numpy(np_mask.astype(np.float32) / 255.0)
            network.set_region(i, i == len(networks) - 1, mask)
        mask_images = None

    prev_image = None  # for VGG16 guided
    if args.guide_image_path is not None:
        print(f"load image for CLIP/VGG16/ControlNet guidance: {args.guide_image_path}")
        guide_images = []
        for p in args.guide_image_path:
            guide_images.extend(load_images(p))

        print(f"loaded {len(guide_images)} guide images for guidance")
        if len(guide_images) == 0:
            print(f"No guide image, use previous generated image.")
            guide_images = None
    else:
        guide_images = None

    # seed指定時はseedを決めておく
    if args.seed is not None:
        random.seed(args.seed)
        predefined_seeds = [random.randint(0, 0x7FFFFFFF) for _ in range(args.n_iter * len(prompt_list) * args.images_per_prompt)]
        if len(predefined_seeds) == 1:
            predefined_seeds[0] = args.seed
    else:
        predefined_seeds = None

    if args.W is None:
        args.W = 512
    if args.H is None:
        args.H = 512

    os.makedirs(args.outdir, exist_ok=True)
    max_embeddings_multiples = 1 if args.max_embeddings_multiples is None else args.max_embeddings_multiples

    for gen_iter in range(args.n_iter):
        print(f"iteration {gen_iter+1}/{args.n_iter}")
        iter_seed = random.randint(0, 0x7FFFFFFF)

        def process_batch(batch: List[BatchData]):
            batch_size = len(batch)
            (
                return_latents,
                (step_first, _, _, _, init_image, mask_image, _, guide_image),
                (width, height, steps, scale, negative_scale, strength, network_muls, num_sub_prompts),
            ) = batch[0]
            noise_shape = (LATENT_CHANNELS, height // DOWNSAMPLING_FACTOR, width // DOWNSAMPLING_FACTOR)

            prompts = []
            start_code = torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
            noises = [
                torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
                for _ in range(steps * scheduler_num_noises_per_step)
            ]
            seeds = []
            clip_prompts = []

            # Generate a random number here to use the same random number regardless of the position in the batch. Also check if image/mask is the same in batch.
            all_images_are_same = True
            all_masks_are_same = True
            all_guide_images_are_same = True
            for i, (_, (_, prompt, seed, init_image, mask_image, clip_prompt, guide_image), _) in enumerate(batch):
                prompts.append(prompt)
                seeds.append(seed)
                clip_prompts.append(clip_prompt)

                if init_image is not None:
                    init_images.append(init_image)
                    if i > 0 and all_images_are_same:
                        all_images_are_same = init_images[-2] is init_image

                if mask_image is not None:
                    mask_images.append(mask_image)
                    if i > 0 and all_masks_are_same:
                        all_masks_are_same = mask_images[-2] is mask_image

                if guide_image is not None:
                    if type(guide_image) is list:
                        guide_images.extend(guide_image)
                        all_guide_images_are_same = False
                    else:
                        guide_images.append(guide_image)
                        if i > 0 and all_guide_images_are_same:
                            all_guide_images_are_same = guide_images[-2] is guide_image

                # make start code
                torch.manual_seed(seed)
                start_code[i] = torch.randn(noise_shape, device=device, dtype=dtype)

                # make each noises
                for j in range(steps * scheduler_num_noises_per_step):
                    noises[j][i] = torch.randn(noise_shape, device=device, dtype=dtype)

            noise_manager.reset_sampler_noises(noises)

            # すべての画像が同じなら1枚だけpipeに渡すことでpipe側で処理を高速化する
            if init_images is not None and all_images_are_same:
                init_images = init_images[0]
            if mask_images is not None and all_masks_are_same:
                mask_images = mask_images[0]
            if guide_images is not None and all_guide_images_are_same:
                guide_images = guide_images[0]

            # ControlNet使用時はguide imageをリサイズする
            if control_nets:
                # TODO resampleのメソッド
                guide_images = guide_images if type(guide_images) == list else [guide_images]
                guide_images = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in guide_images]
                if len(guide_images) == 1:
                    guide_images = guide_images[0]

            # generate
            if networks:
                shared = {}
                for n, m in zip(networks, network_muls if network_muls else network_default_muls):
                    n.set_multiplier(m)
                    if regional_network:
                        n.set_current_generation(batch_size, num_sub_prompts, width, height, shared)

            images = pipe(
                prompts,
                negative_prompts,
                init_images,
                mask_images,
                height,
                width,
                steps,
                scale,
                negative_scale,
                strength,
                latents=start_code,
                output_type="pil",
                max_embeddings_multiples=max_embeddings_multiples,
                vae_batch_size=args.vae_batch_size,
                return_latents=return_latents,
                clip_prompts=clip_prompts,
                clip_guide_images=guide_images,
            )[0]
            # if highres_1st and not args.highres_fix_save_1st:  # return images or latents
            #     return images

            # save image
            ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
            for i, (image, prompt, negative_prompts, seed, clip_prompt) in enumerate(
                zip(images, prompts, negative_prompts, seeds, clip_prompts)
            ):
                metadata = PngInfo()
                metadata.add_text("prompt", prompt)
                metadata.add_text("seed", str(seed))
                metadata.add_text("sampler", args.sampler)
                metadata.add_text("steps", str(steps))
                metadata.add_text("scale", str(scale))
                if negative_scale is not None:
                    metadata.add_text("negative-scale", str(negative_scale))
                if clip_prompt is not None:
                    metadata.add_text("clip-prompt", clip_prompt)

                if args.use_original_file_name and init_images is not None:
                    if type(init_images) is list:
                        fln = os.path.splitext(os.path.basename(init_images[i % len(init_images)].filename))[0] + ".png"
                    else:
                        fln = os.path.splitext(os.path.basename(init_images.filename))[0] + ".png"
                elif args.sequential_file_name:
                    fln = f"im_{step_first + i + 1:06d}.png"
                else:
                    fln = f"im_{ts_str}_{i:03d}_{seed}.png"

                image.save(os.path.join(args.outdir, fln), pnginfo=metadata)

            if not args.no_preview and args.interactive:
                try:
                    import cv2

                    for prompt, image in zip(prompts, images):
                        cv2.imshow(prompt[:128], np.array(image)[:, :, ::-1])
                        cv2.waitKey()
                        cv2.destroyAllWindows()
                except ImportError:
                    print("opencv-python is not installed, cannot preview")

            return images

        # Loop of image generation prompts
        prompt_index = 0
        global_step = 0
        batch_data = []
        while args.interactive or prompt_index < len(prompt_list):
            if len(prompt_list) == 0:
                # interactive
                valid = False
                while not valid:
                    print("\nType prompt:")
                    try:
                        prompt = input()
                    except EOFError:
                        break

                    valid = len(prompt.strip().split(" --")[0].strip()) > 0
                if not valid:  # EOF, end app
                    break
            else:
                prompt = prompt_list[prompt_index]

            prompt_index += 1

        if len(batch_data) > 0:
            process_batch(batch_data)
            batch_data.clear()

    print("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--v2", action="store_true", help="load Stable Diffusion v2.0 model")
    parser.add_argument(
        "--v_parameterization", action="store_true", help="enable v-parameterization training"
    )
    parser.add_argument("--prompt", type=str, default=None, help="prompt")
    parser.add_argument(
        "--from_file", type=str, default=None, help="if specified, load prompts from this file"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="interactive mode (generates one image)"
    )
    parser.add_argument(
        "--no_preview", action="store_true", help="do not show generated image in interactive mode"
    )
    parser.add_argument(
        "--image_path", type=str, default=None, help="image to inpaint or to generate from"
    )
    parser.add_argument("--mask_path", type=str, default=None, help="mask in inpainting")
    parser.add_argument("--strength", type=float, default=None, help="img2img strength")
    parser.add_argument("--images_per_prompt", type=int, default=1, help="number of images per prompt")
    parser.add_argument("--outdir", type=str, default="outputs", help="dir to write results to")
    parser.add_argument("--sequential_file_name", action="store_true", help="sequential output file name")
    parser.add_argument(
        "--use_original_file_name",
        action="store_true",
        help="prepend original file name in img2img",
    )
    # parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim eta (eta=0.0 corresponds to deterministic sampling", )
    parser.add_argument("--n_iter", type=int, default=1, help="sample this often")
    parser.add_argument("--H", type=int, default=None, help="image height, in pixel space")
    parser.add_argument("--W", type=int, default=None, help="image width, in pixel space")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--vae_batch_size",
        type=float,
        default=None,
        help="batch size for VAE, < 1.0 for ratio",
    )
    parser.add_argument("--steps", type=int, default=50, help="number of ddim sampling steps")
    parser.add_argument(
        "--sampler",
        type=str,
        default="ddim",
        choices=[
            "ddim",
            "pndm",
            "lms",
            "euler",
            "euler_a",
            "heun",
            "dpm_2",
            "dpm_2_a",
            "dpmsolver",
            "dpmsolver++",
            "dpmsingle",
            "k_lms",
            "k_euler",
            "k_euler_a",
            "k_dpm_2",
            "k_dpm_2_a",
        ],
        help=f"sampler (scheduler) type",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint of model")
    parser.add_argument(
        "--vae", type=str, default=None, help="path to checkpoint of vae to replace"
    )
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default=None,
        help="directory for caching Tokenizer (for offline training)",
    )
    # parser.add_argument("--replace_clip_l14_336", action='store_true',
    #                     help="Replace CLIP (Text Encoder) to l/14@336")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed, or seed of seeds in multiple generation",
    )
    parser.add_argument(
        "--iter_same_seed",
        action="store_true",
        help="use same seed for all prompts in iteration if no seed specified",
    )
    parser.add_argument("--fp16", action="store_true", help="use fp16")
    parser.add_argument("--bf16", action="store_true", help="use bfloat16")
    parser.add_argument("--xformers", action="store_true", help="use xformers")
    parser.add_argument(
        "--diffusers_xformers",
        action="store_true",
        help="use xformers by diffusers (Hypernetworks doesn't work)",
    )
    parser.add_argument(
        "--opt_channels_last", action="store_true", help="set channels last option to model"
    )
    parser.add_argument(
        "--network_module", type=str, default=None, nargs="*", help="additional network module to use"
    )
    parser.add_argument(
        "--network_weights", type=str, default=None, nargs="*", help="additional network weights to load"
    )
    parser.add_argument("--network_mul", type=float, default=None, nargs="*", help="additional network multiplier")
    parser.add_argument(
        "--network_args", type=str, default=None, nargs="*", help="additional argmuments for network (key=value)"
    )
    parser.add_argument("--network_show_meta", action="store_true", help="show metadata of network model")
    parser.add_argument("--network_merge", action="store_true", help="merge network weights to original model")
    parser.add_argument(
        "--textual_inversion_embeddings",
        type=str,
        default=None,
        nargs="*",
        help="Embeddings files of Textual Inversion",
    )
    parser.add_argument(
        "--XTI_embeddings",
        type=str,
        default=None,
        nargs="*",
        help="Embeddings files of Extended Textual Inversion",
    )
    parser.add_argument("--clip_skip", type=int, default=None, help="layer number from bottom to use in CLIP")
    parser.add_argument(
        "--max_embeddings_multiples",
        type=int,
        default=None,
        help="max embeding multiples, max token length is 75 * multiples",
    )
    parser.add_argument(
        "--clip_guidance_scale",
        type=float,
        default=0.0,
        help="enable CLIP guided SD, scale for guidance (DDIM, PNDM, LMS samplers only)",
    )
    parser.add_argument(
        "--clip_image_guidance_scale",
        type=float,
        default=0.0,
        help="enable CLIP guided SD by image, scale for guidance",
    )
    parser.add_argument(
        "--vgg16_guidance_scale",
        type=float,
        default=0.0,
        help="enable VGG16 guided SD by image, scale for guidance",
    )
    parser.add_argument(
        "--vgg16_guidance_layer",
        type=int,
        default=20,
        help="layer of VGG16 to calculate contents guide (1~30, 20 for conv4_2)",
    )
    parser.add_argument(
        "--guide_image_path", type=str, default=None, nargs="*", help="image to CLIP guidance"
    )
    parser.add_argument(
        "--highres_fix_scale",
        type=float,
        default=None,
        help="enable highres fix, reso scale for 1st stage",
    )
    parser.add_argument(
        "--highres_fix_steps", type=int, default=28, help="1st stage steps for highres fix"
    )
    parser.add_argument(
        "--highres_fix_save_1st", action="store_true", help="save 1st stage images for highres fix"
    )
    parser.add_argument(
        "--highres_fix_latents_upscaling",
        action="store_true",
        help="use latents upscaling for highres fix",
    )
    parser.add_argument(
        "--negative_scale", type=float, default=None, help="set another guidance scale for negative prompt"
    )

    parser.add_argument(
        "--control_net_models", type=str, default=None, nargs="*", help="ControlNet models to use"
    )
    parser.add_argument(
        "--control_net_preps", type=str, default=None, nargs="*", help="ControlNet preprocess to use"
    )
    parser.add_argument("--control_net_weights", type=float, default=None, nargs="*", help="ControlNet weights")
    parser.add_argument(
        "--control_net_ratios",
        type=float,
        default=None,
        nargs="*",
        help="ControlNet guidance ratio for steps",
    )
    # parser.add_argument(
    #     "--control_net_image_path", type=str, default=None, nargs="*", help="image for ControlNet guidance"
    # )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    main(args)
