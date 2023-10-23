
import math
import os
from datetime import datetime
from library.common_gui import (
    run_cmd_advanced_training,
    run_cmd_training,
    check_if_model_exist,
    output_message,
    verify_image_folder_pattern,
    SaveConfigFile,
    save_to_file,
    check_duplicate_filenames,
)

# Import and modify utils from kohya library: https://github.com/kohya-ss/sd-scripts/tree/main/library
from library.class_command_executor import CommandExecutor

from library.custom_logging import setup_logging


log = setup_logging()
executor = CommandExecutor()

def train_model(
    headless,
    print_only,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl,
    logging_dir,
    train_data_dir,
    reg_data_dir,
    output_dir,
    max_resolution,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    cache_latents,
    cache_latents_to_disk,
    caption_extension,
    enable_bucket,
    gradient_checkpointing,
    full_fp16,
    no_token_padding,
    stop_text_encoder_training_pct,
    min_bucket_reso,
    max_bucket_reso,
    # use_8bit_adam,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    resume,
    prior_loss_weight,
    text_encoder_lr,
    unet_lr,
    network_dim,
    lora_network_weights,
    dim_from_weights,
    color_aug,
    flip_aug,
    clip_skip,
    gradient_accumulation_steps,
    mem_eff_attn,
    output_name,
    model_list,  # Keep this. Yes, it is unused here but required given the common list used
    max_token_length,
    max_train_epochs,
    max_train_steps,
    max_data_loader_n_workers,
    network_alpha,
    training_comment,
    keep_tokens,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    persistent_data_loader_workers,
    bucket_no_upscale,
    random_crop,
    bucket_reso_steps,
    v_pred_like_loss,
    caption_dropout_every_n_epochs,
    caption_dropout_rate,
    optimizer,
    optimizer_args,
    lr_scheduler_args,
    noise_offset_type,
    noise_offset,
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    LoRA_type,
    factor,
    use_cp,
    decompose_both,
    train_on_input,
    conv_dim,
    conv_alpha,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
    vae_batch_size,
    min_snr_gamma,
    down_lr_weight,
    mid_lr_weight,
    up_lr_weight,
    block_lr_zero_threshold,
    block_dims,
    block_alphas,
    conv_block_dims,
    conv_block_alphas,
    weighted_captions,
    unit,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    use_wandb,
    wandb_api_key,
    scale_v_pred_loss_like_noise_pred,
    scale_weight_norms,
    network_dropout,
    rank_dropout,
    module_dropout,
    sdxl_cache_text_encoder_outputs,
    sdxl_no_half_vae,
    full_bf16,
    min_timestep,
    max_timestep,
):
    # Get list of function parameters and values
    parameters = list(locals().items())
    global command_running
    print_only_bool = True if print_only == True else False
    log.info(f'Start training LoRA {LoRA_type} ...')
    headless_bool = True if headless == True else False

    if pretrained_model_name_or_path == '':
        output_message(
            msg='Source model information is missing', headless=headless_bool
        )
        return

    if train_data_dir == '':
        output_message(
            msg='Image folder path is missing', headless=headless_bool
        )
        return

    # Check if there are files with the same filename but different image extension... warn the user if it is the case.
    check_duplicate_filenames(train_data_dir)

    if not os.path.exists(train_data_dir):
        output_message(
            msg='Image folder does not exist', headless=headless_bool
        )
        return

    if not verify_image_folder_pattern(train_data_dir):
        return

    if reg_data_dir != '':
        if not os.path.exists(reg_data_dir):
            output_message(
                msg='Regularisation folder does not exist',
                headless=headless_bool,
            )
            return

        if not verify_image_folder_pattern(reg_data_dir):
            return

    if output_dir == '':
        output_message(
            msg='Output folder path is missing', headless=headless_bool
        )
        return

    if int(bucket_reso_steps) < 1:
        output_message(
            msg='Bucket resolution steps need to be greater than 0',
            headless=headless_bool,
        )
        return

    if noise_offset == '':
        noise_offset = 0

    if float(noise_offset) > 1 or float(noise_offset) < 0:
        output_message(
            msg='Noise offset need to be a value between 0 and 1',
            headless=headless_bool,
        )
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if stop_text_encoder_training_pct > 0:
        output_message(
            msg='Output "stop text encoder training" is not yet supported. Ignoring',
            headless=headless_bool,
        )
        stop_text_encoder_training_pct = 0

    if check_if_model_exist(
        output_name, output_dir, save_model_as, headless=headless_bool
    ):
        return

    # If string is empty set string to 0.
    if text_encoder_lr == '':
        text_encoder_lr = 0
    if unet_lr == '':
        unet_lr = 0

    # Get a list of all subfolders in train_data_dir
    subfolders = [
        f
        for f in os.listdir(train_data_dir)
        if os.path.isdir(os.path.join(train_data_dir, f))
    ]

    total_steps = 0

    # Loop through each subfolder and extract the number of repeats
    for folder in subfolders:
        try:
            # Extract the number of repeats from the folder name
            repeats = int(folder.split('_')[0])

            # Count the number of images in the folder
            num_images = len(
                [
                    f
                    for f, lower_f in (
                        (file, file.lower())
                        for file in os.listdir(
                            os.path.join(train_data_dir, folder)
                        )
                    )
                    if lower_f.endswith(('.jpg', '.jpeg', '.png', '.webp'))
                ]
            )

            log.info(f'Folder {folder}: {num_images} images found')

            # Calculate the total number of steps for this folder
            steps = repeats * num_images

            # log.info the result
            log.info(f'Folder {folder}: {steps} steps')

            total_steps += steps

        except ValueError:
            # Handle the case where the folder name does not contain an underscore
            log.info(
                f"Error: '{folder}' does not contain an underscore, skipping..."
            )

    if reg_data_dir == '':
        reg_factor = 1
    else:
        log.warning(
            'Regularisation images are used... Will double the number of steps required...'
        )
        reg_factor = 2

    log.info(f'Total steps: {total_steps}')
    log.info(f'Train batch size: {train_batch_size}')
    log.info(f'Gradient accumulation steps: {gradient_accumulation_steps}')
    log.info(f'Epoch: {epoch}')
    log.info(f'Regulatization factor: {reg_factor}')

    if max_train_steps == '' or max_train_steps == '0':
        # calculate max_train_steps
        max_train_steps = int(
            math.ceil(
                float(total_steps)
                / int(train_batch_size)
                / int(gradient_accumulation_steps)
                * int(epoch)
                * int(reg_factor)
            )
        )
        log.info(
            f'max_train_steps ({total_steps} / {train_batch_size} / {gradient_accumulation_steps} * {epoch} * {reg_factor}) = {max_train_steps}'
        )

    # calculate stop encoder training
    if stop_text_encoder_training_pct == None:
        stop_text_encoder_training = 0
    else:
        stop_text_encoder_training = math.ceil(
            float(max_train_steps) / 100 * int(stop_text_encoder_training_pct)
        )
    log.info(f'stop_text_encoder_training = {stop_text_encoder_training}')

    lr_warmup_steps = round(float(int(lr_warmup) * int(max_train_steps) / 100))
    log.info(f'lr_warmup_steps = {lr_warmup_steps}')

    run_cmd = f'accelerate launch --num_cpu_threads_per_process={num_cpu_threads_per_process}'
    if sdxl:
        run_cmd += f' "./sdxl_train_network.py"'
    else:
        run_cmd += f' "./train_network.py"'

    if v2:
        run_cmd += ' --v2'
    if v_parameterization:
        run_cmd += ' --v_parameterization'
    if enable_bucket:
        run_cmd += f' --enable_bucket --min_bucket_reso={min_bucket_reso} --max_bucket_reso={max_bucket_reso}'
    if no_token_padding:
        run_cmd += ' --no_token_padding'
    if weighted_captions:
        run_cmd += ' --weighted_captions'
    run_cmd += (
        f' --pretrained_model_name_or_path="{pretrained_model_name_or_path}"'
    )
    run_cmd += f' --train_data_dir="{train_data_dir}"'
    if len(reg_data_dir):
        run_cmd += f' --reg_data_dir="{reg_data_dir}"'
    run_cmd += f' --resolution="{max_resolution}"'
    run_cmd += f' --output_dir="{output_dir}"'
    if not logging_dir == '':
        run_cmd += f' --logging_dir="{logging_dir}"'
    run_cmd += f' --network_alpha="{network_alpha}"'
    if not training_comment == '':
        run_cmd += f' --training_comment="{training_comment}"'
    if not stop_text_encoder_training == 0:
        run_cmd += (
            f' --stop_text_encoder_training={stop_text_encoder_training}'
        )
    if not save_model_as == 'same as source model':
        run_cmd += f' --save_model_as={save_model_as}'
    if not float(prior_loss_weight) == 1.0:
        run_cmd += f' --prior_loss_weight={prior_loss_weight}'

        try:
            import lycoris
        except ModuleNotFoundError:
            log.info(
                "\033[1;31mError:\033[0m The required module 'lycoris_lora' is not installed. Please install by running \033[33mupgrade.ps1\033[0m before running this program."
            )
            return
        run_cmd += f' --network_module=lycoris.kohya'
        run_cmd += f' --network_args "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "factor={factor}" "use_cp={use_cp}" "algo=lokr"'
        # This is a hack to fix a train_network LoHA logic issue
        if not network_dropout > 0.0:
            run_cmd += f' --network_dropout="{network_dropout}"'

    if LoRA_type in ['Standard']:
        kohya_lora_var_list = [
            'down_lr_weight',
            'mid_lr_weight',
            'up_lr_weight',
            'block_lr_zero_threshold',
            'block_dims',
            'block_alphas',
            'conv_block_dims',
            'conv_block_alphas',
            'rank_dropout',
            'module_dropout',
        ]

        run_cmd += f' --network_module=networks.lora'
        kohya_lora_vars = {
            key: value
            for key, value in vars().items()
            if key in kohya_lora_var_list and value
        }

        network_args = ''
        if LoRA_type == 'Kohya LoCon':
            network_args += f' conv_dim="{conv_dim}" conv_alpha="{conv_alpha}"'

        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f' {key}="{value}"'

        if network_args:
            run_cmd += f' --network_args{network_args}'

    if LoRA_type in [
        'LoRA-FA',
    ]:
        kohya_lora_var_list = [
            'down_lr_weight',
            'mid_lr_weight',
            'up_lr_weight',
            'block_lr_zero_threshold',
            'block_dims',
            'block_alphas',
            'conv_block_dims',
            'conv_block_alphas',
            'rank_dropout',
            'module_dropout',
        ]

        run_cmd += f' --network_module=networks.lora_fa'
        kohya_lora_vars = {
            key: value
            for key, value in vars().items()
            if key in kohya_lora_var_list and value
        }

        network_args = ''
        if LoRA_type == 'Kohya LoCon':
            network_args += f' conv_dim="{conv_dim}" conv_alpha="{conv_alpha}"'

        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f' {key}="{value}"'

        if network_args:
            run_cmd += f' --network_args{network_args}'

    if LoRA_type in ['Kohya DyLoRA']:
        kohya_lora_var_list = [
            'conv_dim',
            'conv_alpha',
            'down_lr_weight',
            'mid_lr_weight',
            'up_lr_weight',
            'block_lr_zero_threshold',
            'block_dims',
            'block_alphas',
            'conv_block_dims',
            'conv_block_alphas',
            'rank_dropout',
            'module_dropout',
            'unit',
        ]

        run_cmd += f' --network_module=networks.dylora'
        kohya_lora_vars = {
            key: value
            for key, value in vars().items()
            if key in kohya_lora_var_list and value
        }

        network_args = ''

        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f' {key}="{value}"'

        if network_args:
            run_cmd += f' --network_args{network_args}'

    if not (float(text_encoder_lr) == 0) or not (float(unet_lr) == 0):
        if not (float(text_encoder_lr) == 0) and not (float(unet_lr) == 0):
            run_cmd += f' --text_encoder_lr={text_encoder_lr}'
            run_cmd += f' --unet_lr={unet_lr}'
        elif not (float(text_encoder_lr) == 0):
            run_cmd += f' --text_encoder_lr={text_encoder_lr}'
            run_cmd += f' --network_train_text_encoder_only'
        else:
            run_cmd += f' --unet_lr={unet_lr}'
            run_cmd += f' --network_train_unet_only'
    else:
        if float(learning_rate) == 0:
            output_message(
                msg='Please input learning rate values.',
                headless=headless_bool,
            )
            return

    run_cmd += f' --network_dim={network_dim}'

    # if LoRA_type not in ['LyCORIS/LoCon']:
    if not lora_network_weights == '':
        run_cmd += f' --network_weights="{lora_network_weights}"'
        if dim_from_weights:
            run_cmd += f' --dim_from_weights'

    if int(gradient_accumulation_steps) > 1:
        run_cmd += f' --gradient_accumulation_steps={int(gradient_accumulation_steps)}'
    if not output_name == '':
        run_cmd += f' --output_name="{output_name}"'
    if not lr_scheduler_num_cycles == '':
        run_cmd += f' --lr_scheduler_num_cycles="{lr_scheduler_num_cycles}"'
    else:
        run_cmd += f' --lr_scheduler_num_cycles="{epoch}"'
    if not lr_scheduler_power == '':
        run_cmd += f' --lr_scheduler_power="{lr_scheduler_power}"'

    if scale_weight_norms > 0.0:
        run_cmd += f' --scale_weight_norms="{scale_weight_norms}"'

    if network_dropout > 0.0:
        run_cmd += f' --network_dropout="{network_dropout}"'

    if sdxl_cache_text_encoder_outputs:
        run_cmd += f' --cache_text_encoder_outputs'

    if sdxl_no_half_vae:
        run_cmd += f' --no_half_vae'

    if full_bf16:
        run_cmd += f' --full_bf16'

    run_cmd += run_cmd_training(
        learning_rate=learning_rate,
        lr_scheduler=lr_scheduler,
        lr_warmup_steps=lr_warmup_steps,
        train_batch_size=train_batch_size,
        max_train_steps=max_train_steps,
        save_every_n_epochs=save_every_n_epochs,
        mixed_precision=mixed_precision,
        save_precision=save_precision,
        seed=seed,
        caption_extension=caption_extension,
        cache_latents=cache_latents,
        cache_latents_to_disk=cache_latents_to_disk,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
        lr_scheduler_args=lr_scheduler_args,
    )

    run_cmd += run_cmd_advanced_training(
        max_train_epochs=max_train_epochs,
        max_data_loader_n_workers=max_data_loader_n_workers,
        max_token_length=max_token_length,
        resume=resume,
        save_state=save_state,
        mem_eff_attn=mem_eff_attn,
        clip_skip=clip_skip,
        flip_aug=flip_aug,
        color_aug=color_aug,
        shuffle_caption=shuffle_caption,
        gradient_checkpointing=gradient_checkpointing,
        full_fp16=full_fp16,
        xformers=xformers,
        # use_8bit_adam=use_8bit_adam,
        keep_tokens=keep_tokens,
        persistent_data_loader_workers=persistent_data_loader_workers,
        bucket_no_upscale=bucket_no_upscale,
        random_crop=random_crop,
        bucket_reso_steps=bucket_reso_steps,
        v_pred_like_loss=v_pred_like_loss,
        caption_dropout_every_n_epochs=caption_dropout_every_n_epochs,
        caption_dropout_rate=caption_dropout_rate,
        noise_offset_type=noise_offset_type,
        noise_offset=noise_offset,
        adaptive_noise_scale=adaptive_noise_scale,
        multires_noise_iterations=multires_noise_iterations,
        multires_noise_discount=multires_noise_discount,
        additional_parameters=additional_parameters,
        vae_batch_size=vae_batch_size,
        min_snr_gamma=min_snr_gamma,
        save_every_n_steps=save_every_n_steps,
        save_last_n_steps=save_last_n_steps,
        save_last_n_steps_state=save_last_n_steps_state,
        use_wandb=use_wandb,
        wandb_api_key=wandb_api_key,
        scale_v_pred_loss_like_noise_pred=scale_v_pred_loss_like_noise_pred,
        min_timestep=min_timestep,
        max_timestep=max_timestep,
    )

    # run_cmd += run_cmd_sample(
    #     sample_every_n_steps,
    #     sample_every_n_epochs,
    #     sample_sampler,
    #     sample_prompts,
    #     output_dir,
    # )

    if print_only_bool:
        log.warning(
            'Here is the trainer command as a reference. It will not be executed:\n'
        )
        print(run_cmd)

        save_to_file(run_cmd)
    else:
        # Saving config file for model
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime('%Y%m%d-%H%M%S')
        file_path = os.path.join(
            output_dir, f'{output_name}_{formatted_datetime}.json'
        )

        log.info(f'Saving training config to {file_path}...')

        SaveConfigFile(
            parameters=parameters,
            file_path=file_path,
            exclusion=['file_path', 'save_as', 'headless', 'print_only'],
        )

        log.info(run_cmd)
        # Run the command
        executor.execute_command(run_cmd=run_cmd)