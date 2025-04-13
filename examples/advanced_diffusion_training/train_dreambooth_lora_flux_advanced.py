#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import itertools
import logging
import math
import os
import random
import re
import shutil
import json # Added for metadata loading
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn.functional as F # Added for mask resizing
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import save_file
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")

logger = get_logger(__name__)


def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    train_text_encoder=False,
    train_text_encoder_ti=False,
    enable_t5_ti=False,
    pure_textual_inversion=False,
    token_abstraction_dict=None,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
):
    widget_dict = []
    trigger_str = f"You should use {instance_prompt} to trigger the image generation."

    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )
    diffusers_load_lora = ""
    diffusers_imports_pivotal = ""
    diffusers_example_pivotal = ""
    if not pure_textual_inversion:
        diffusers_load_lora = (
            f"""pipeline.load_lora_weights('{repo_id}', weight_name='pytorch_lora_weights.safetensors')"""
        )
    if train_text_encoder_ti:
        embeddings_filename = f"{repo_folder}_emb"
        ti_keys = ", ".join(f'"{match}"' for match in re.findall(r"<s\d+>", instance_prompt))
        trigger_str = (
            "To trigger image generation of trained concept(or concepts) replace each concept identifier "
            "in you prompt with the new inserted tokens:\n"
        )
        diffusers_imports_pivotal = """from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
            """
        if enable_t5_ti:
            diffusers_example_pivotal = f"""embedding_path = hf_hub_download(repo_id='{repo_id}', filename='{embeddings_filename}.safetensors', repo_type="model")
    state_dict = load_file(embedding_path)
    pipeline.load_textual_inversion(state_dict["clip_l"], token=[{ti_keys}], text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer)
    pipeline.load_textual_inversion(state_dict["t5"], token=[{ti_keys}], text_encoder=pipeline.text_encoder_2, tokenizer=pipeline.tokenizer_2)
            """
        else:
            diffusers_example_pivotal = f"""embedding_path = hf_hub_download(repo_id='{repo_id}', filename='{embeddings_filename}.safetensors', repo_type="model")
    state_dict = load_file(embedding_path)
    pipeline.load_textual_inversion(state_dict["clip_l"], token=[{ti_keys}], text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer)
            """
        if token_abstraction_dict:
            for key, value in token_abstraction_dict.items():
                tokens = "".join(value)
                trigger_str += f"""
    to trigger concept `{key}` → use `{tokens}` in your prompt \n
    """

    model_description = f"""
# Flux DreamBooth LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} DreamBooth LoRA weights for {base_model}.

The weights were trained using [DreamBooth](https://dreambooth.github.io/) with the [Flux diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md).

Was LoRA for the text encoder enabled? {train_text_encoder}.

Pivotal tuning was enabled: {train_text_encoder_ti}.

## Trigger words

{trigger_str}

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch
{diffusers_imports_pivotal}
pipeline = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to('cuda')
{diffusers_load_lora}
{diffusers_example_pivotal}
image = pipeline('{validation_prompt if validation_prompt else instance_prompt}').images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora",
        "flux",
        "flux-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    torch_dtype,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()

    # pre-calculate  prompt embeds, pooled prompt embeds, text ids because t5 does not support autocast
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            pipeline_args["prompt"], prompt_2=pipeline_args["prompt"]
        )
    images = []
    for _ in range(args.num_validation_images):
        with autocast_ctx:
            image = pipeline(
                prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, generator=generator
            ).images[0]
            images.append(image)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    free_memory()

    return images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default=None,
        help="A path to a JSON Lines file (.jsonl) containing metadata (file_name, text, mask_path). Alternative to --instance_data_dir.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. Ignored if --metadata_file is provided."),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="file_name", # Changed default from "image"
        help="The column/key in the dataset or metadata file containing the relative path to the image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text", # Changed default from None
        help="The column/key in the dataset or metadata file containing the instance prompt for each image.",
    )
    parser.add_argument(
        "--mask_column",
        type=str,
        default="mask_path",
        help="The column/key in the metadata file containing the relative path to the mask image (optional).",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--token_abstraction",
        type=str,
        default="TOK",
        help="identifier specifying the instance(or instances) as used in instance_prompt, validation prompt, "
        "captions - e.g. TOK. To use multiple identifiers, please specify them in a comma separated string - e.g. "
        "'TOK,TOK2,TOK3' etc.",
    )

    parser.add_argument(
        "--num_new_tokens_per_abstraction",
        type=int,
        default=None,
        help="number of new tokens inserted to the tokenizers per token_abstraction identifier when "
        "--train_text_encoder_ti = True. By default, each --token_abstraction (e.g. TOK) is mapped to 2 new "
        "tokens - <si><si+1> ",
    )
    parser.add_argument(
        "--initializer_concept",
        type=str,
        default=None,
        help="the concept to use to initialize the new inserted tokens when training with "
        "--train_text_encoder_ti = True. By default, new tokens (<si><si+1>) are initialized with random value. "
        "Alternatively, you could specify a different word/words whose value will be used as the starting point for the new inserted tokens. "
        "--num_new_tokens_per_abstraction is ignored when initializer_concept is provided",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=None,
        help="The alpha parameter for LoRA scaling. Defaults to `rank` if not specified.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dreambooth-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_text_encoder_ti",
        action="store_true",
        help=("Whether to use pivotal tuning / textual inversion"),
    )
    parser.add_argument(
        "--enable_t5_ti",
        action="store_true",
        help=(
            "Whether to use pivotal tuning / textual inversion for the T5 encoder as well (in addition to CLIP encoder)"
        ),
    )

    parser.add_argument(
        "--train_text_encoder_ti_frac",
        type=float,
        default=0.5,
        help=("The percentage of epochs to perform textual inversion"),
    )

    parser.add_argument(
        "--train_text_encoder_frac",
        type=float,
        default=1.0,
        help=("The percentage of epochs to perform text encoder tuning"),
    )
    parser.add_argument(
        "--train_transformer_frac",
        type=float,
        default=1.0,
        help=("The percentage of epochs to perform transformer tuning"),
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for transformer params"
    )
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            "The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. "
            'E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only. For more examples refer to https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/README_flux.md'
        ),
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--alpha_mask",
        action="store_true",
        help="Use the alpha channel from RGBA images as a mask for the loss.",
    )
    parser.add_argument(
        "--masked_loss",
        action="store_true",
        help="Alias for --alpha_mask. Use the alpha channel from RGBA images as a mask for the loss.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None and args.metadata_file is None:
        raise ValueError("Specify either `--dataset_name`, `--instance_data_dir`, or `--metadata_file`")

    if args.dataset_name is not None and args.instance_data_dir is not None and args.metadata_file is not None:
        raise ValueError("Specify only one of `--dataset_name`, `--instance_data_dir`, or `--metadata_file`")

    if args.train_text_encoder and args.train_text_encoder_ti:
        raise ValueError(
            "Specify only one of `--train_text_encoder` or `--train_text_encoder_ti. "
            "For full LoRA text encoder training check --train_text_encoder, for textual "
            "inversion training check `--train_text_encoder_ti`"
        )
    if args.train_transformer_frac < 1 and not args.train_text_encoder_ti:
        raise ValueError(
            "--train_transformer_frac must be == 1 if text_encoder training / textual inversion is not enabled."
        )
    if args.train_transformer_frac < 1 and args.train_text_encoder_ti_frac < 1:
        raise ValueError(
            "--train_transformer_frac and --train_text_encoder_ti_frac are identical and smaller than 1. "
            "This contradicts with --max_train_steps, please specify different values or set both to 1."
        )
    if args.enable_t5_ti and not args.train_text_encoder_ti:
        logger.warning("You need not use --enable_t5_ti without --train_text_encoder_ti.")

    if args.train_text_encoder_ti and args.initializer_concept and args.num_new_tokens_per_abstraction:
        logger.warning(
            "When specifying --initializer_concept, the number of tokens per abstraction is detrimned "
            "by the initializer token. --num_new_tokens_per_abstraction will be ignored"
        )

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        if args.class_data_dir is not None:
            logger.warning("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            logger.warning("You need not use --class_prompt without --with_prior_preservation.")

    return args


# Modified from https://github.com/replicate/cog-sdxl/blob/main/dataset_and_utils.py
class TokenEmbeddingsHandler:
    def __init__(self, text_encoders, tokenizers):
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers

        self.train_ids: Optional[torch.Tensor] = None
        self.train_ids_t5: Optional[torch.Tensor] = None
        self.inserting_toks: Optional[List[str]] = None
        self.embeddings_settings = {}

    def initialize_new_tokens(self, inserting_toks: List[str]):
        idx = 0
        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            assert isinstance(inserting_toks, list), "inserting_toks should be a list of strings."
            assert all(isinstance(tok, str) for tok in inserting_toks), (
                "All elements in inserting_toks should be strings."
            )

            self.inserting_toks = inserting_toks
            special_tokens_dict = {"additional_special_tokens": self.inserting_toks}
            tokenizer.add_special_tokens(special_tokens_dict)
            # Resize the token embeddings as we are adding new special tokens to the tokenizer
            text_encoder.resize_token_embeddings(len(tokenizer))

            # Convert the token abstractions to ids
            if idx == 0:
                self.train_ids = tokenizer.convert_tokens_to_ids(self.inserting_toks)
            else:
                self.train_ids_t5 = tokenizer.convert_tokens_to_ids(self.inserting_toks)

            # random initialization of new tokens
            embeds = (
                text_encoder.text_model.embeddings.token_embedding if idx == 0 else text_encoder.encoder.embed_tokens
            )
            std_token_embedding = embeds.weight.data.std()

            logger.info(f"{idx} text encoder's std_token_embedding: {std_token_embedding}")

            train_ids = self.train_ids if idx == 0 else self.train_ids_t5
            # if initializer_concept are not provided, token embeddings are initialized randomly
            if args.initializer_concept is None:
                hidden_size = (
                    text_encoder.text_model.config.hidden_size if idx == 0 else text_encoder.encoder.config.hidden_size
                )
                embeds.weight.data[train_ids] = (
                    torch.randn(len(train_ids), hidden_size).to(device=self.device).to(dtype=self.dtype)
                    * std_token_embedding
                )
            else:
                # Convert the initializer_token, placeholder_token to ids
                initializer_token_ids = tokenizer.encode(args.initializer_concept, add_special_tokens=False)
                for token_idx, token_id in enumerate(train_ids):
                    embeds.weight.data[token_id] = (embeds.weight.data)[
                        initializer_token_ids[token_idx % len(initializer_token_ids)]
                    ].clone()

            self.embeddings_settings[f"original_embeddings_{idx}"] = embeds.weight.data.clone()
            self.embeddings_settings[f"std_token_embedding_{idx}"] = std_token_embedding

            # makes sure we don't update any embedding weights besides the newly added token
            index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
            index_no_updates[train_ids] = False

            self.embeddings_settings[f"index_no_updates_{idx}"] = index_no_updates

            logger.info(self.embeddings_settings[f"index_no_updates_{idx}"].shape)

            idx += 1

    def save_embeddings(self, file_path: str):
        assert self.train_ids is not None, "Initialize new tokens before saving embeddings."
        tensors = {}
        # text_encoder_one, idx==0 - CLIP ViT-L/14, text_encoder_two, idx==1 - T5 xxl
        idx_to_text_encoder_name = {0: "clip_l", 1: "t5"}
        for idx, text_encoder in enumerate(self.text_encoders):
            train_ids = self.train_ids if idx == 0 else self.train_ids_t5
            embeds = text_encoder.text_model.embeddings.token_embedding if idx == 0 else text_encoder.shared
            assert embeds.weight.data.shape[0] == len(self.tokenizers[idx]), "Tokenizers should be the same."
            new_token_embeddings = embeds.weight.data[train_ids]

            # New tokens for each text encoder are saved under "clip_l" (for text_encoder 0),
            # Note: When loading with diffusers, any name can work - simply specify in inference
            tensors[idx_to_text_encoder_name[idx]] = new_token_embeddings
            # tensors[f"text_encoders_{idx}"] = new_token_embeddings

        save_file(tensors, file_path)

    @property
    def dtype(self):
        return self.text_encoders[0].dtype

    @property
    def device(self):
        return self.text_encoders[0].device

    @torch.no_grad()
    def retract_embeddings(self):
        for idx, text_encoder in enumerate(self.text_encoders):
            embeds = text_encoder.text_model.embeddings.token_embedding if idx == 0 else text_encoder.shared
            index_no_updates = self.embeddings_settings[f"index_no_updates_{idx}"]
            embeds.weight.data[index_no_updates] = (
                self.embeddings_settings[f"original_embeddings_{idx}"][index_no_updates]
                .to(device=text_encoder.device)
                .to(dtype=text_encoder.dtype)
            )

            # for the parts that were updated, we need to normalize them
            # to have the same std as before
            std_token_embedding = self.embeddings_settings[f"std_token_embedding_{idx}"]

            index_updates = ~index_no_updates
            new_embeddings = embeds.weight.data[index_updates]
            off_ratio = std_token_embedding / new_embeddings.std()

            new_embeddings = new_embeddings * (off_ratio**0.1)
            embeds.weight.data[index_updates] = new_embeddings


# Added imports for Path and json
from pathlib import Path
import json

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and handles alpha masks or separate mask files via metadata.
    """
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        class_prompt,
        train_text_encoder_ti,
        token_abstraction_dict=None,  # token mapping for textual inversion
        class_data_root=None,
        class_num=None,
        size=1024,
        repeats=1,
        center_crop=False,
        alpha_mask_required=False, # Flag indicating if masking is needed (for RGBA alpha)
        # Added args from parse_args needed here
        metadata_file=None,
        image_column="file_name",
        caption_column="text",
        mask_column="mask_path",
        dataset_name=None, # Need dataset_name arg
        dataset_config_name=None, # Need dataset_config_name arg
        cache_dir=None, # Need cache_dir arg
        args=None, # Pass full args object
    ):
        self.size = size
        self.center_crop = center_crop
        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt
        self.token_abstraction_dict = token_abstraction_dict
        self.train_text_encoder_ti = train_text_encoder_ti
        self.alpha_mask_required = alpha_mask_required # Store the flag for RGBA alpha
        self.repeats = repeats # Store repeats
        self.args = args # Store args

        self.metadata = None
        self.metadata_dir = None
        self.instance_images_path = [] # Initialize as list
        self.instance_paths_repeated = [] # Initialize

        # Define image extensions
        IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

        # --- Load from metadata file if provided ---
        if args.metadata_file is not None:
            metadata_file = args.metadata_file # Use from args
            image_column = args.image_column
            caption_column = args.caption_column
            mask_column = args.mask_column

            logger.info(f"Loading dataset from metadata file: {metadata_file}")
            self.metadata_dir = Path(metadata_file).parent
            self.metadata = []
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f):
                        try:
                            entry = json.loads(line)
                            # Basic validation for required keys
                            if image_column not in entry:
                                logger.warning(f"Skipping line {line_num+1} in {metadata_file}: Missing key '{image_column}'. Line: {line.strip()}")
                                continue
                            if caption_column not in entry:
                                logger.warning(f"Skipping line {line_num+1} in {metadata_file}: Missing key '{caption_column}'. Line: {line.strip()}")
                                continue
                            # Store the valid entry
                            self.metadata.append(entry)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping malformed line {line_num+1} in {metadata_file}: {line.strip()} - Error: {e}")
            except FileNotFoundError:
                 raise ValueError(f"Metadata file {metadata_file} not found.")

            self.num_instance_images = len(self.metadata)
            if self.num_instance_images == 0:
                raise ValueError(
                    f"No valid entries found in the metadata file {metadata_file}. "
                    f"Please check the file format and content (expected keys: '{image_column}', '{caption_column}')."
                )
            logger.info(f"Loaded {self.num_instance_images} instance entries from metadata.")
            # When using metadata, instance_images_path is not used directly for iteration,
            # but we might store resolved paths if needed elsewhere, or just use metadata directly in __getitem__
            # For simplicity, let's rely on loading directly from metadata in __getitem__
            self.instance_data_root = None # Not used

            # Repeat metadata entries if needed (simplest way)
            repeated_metadata = []
            for entry in self.metadata:
                 repeated_metadata.extend(itertools.repeat(entry, self.repeats))
            self.metadata = repeated_metadata # Replace with repeated list
            self.num_instance_images = len(self.metadata) # Update count based on repeats
            # --- DEBUG PRINT 1 ---
            print(f"DEBUG: After metadata repeat, self.num_instance_images = {self.num_instance_images}")
            # --- END DEBUG ---

        # --- Load from HuggingFace dataset if --dataset_name is provided ---
        elif args.dataset_name is not None:
            dataset_name = args.dataset_name # Use from args
            logger.info(f"Loading dataset from HuggingFace Hub: {dataset_name}")
            # (Existing logic from lines 1004-1057 remains here)
            # ... [omitted for brevity, assume it populates self.instance_images_path and self.custom_instance_prompts] ...
            # Ensure self.instance_images_path is populated correctly by the existing logic
            # Repeat paths after loading
            temp_paths = list(self.instance_images_path) # Get paths loaded by dataset logic
            self.instance_paths_repeated = []
            for img_path in temp_paths:
                 self.instance_paths_repeated.extend(itertools.repeat(img_path, self.repeats))
            self.num_instance_images = len(self.instance_paths_repeated)

        # --- Load from local directory if --instance_data_dir is provided ---
        elif instance_data_root is not None:
            logger.info(f"Loading dataset from local directory: {instance_data_root}")
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError(f"Instance images root {self.instance_data_root} doesn't exist.")

            self.instance_images_path = [
                p for p in Path(instance_data_root).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if not self.instance_images_path:
                 raise ValueError(f"No images found in {instance_data_root}")

            self.custom_instance_prompts = None # Assume no custom prompts with local folder

            # Repeat paths
            self.instance_paths_repeated = []
            for img_path in self.instance_images_path:
                 self.instance_paths_repeated.extend(itertools.repeat(img_path, self.repeats))
            self.num_instance_images = len(self.instance_paths_repeated)
        else:
            raise ValueError("Must provide either --metadata_file, --dataset_name, or --instance_data_dir")


        # --- Common setup after loading data source ---
        self._length = self.num_instance_images # Length is now based on repeated items

        # Prior preservation setup
        if class_data_root is not None:
            # (Existing logic from lines 1089-1097 remains largely the same)
            # ...
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = [
                 p for p in self.class_data_root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if not self.class_images_path:
                 raise ValueError(f"No class images found in {class_data_root}")

            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)

            # Repeat class paths
            repeated_class_paths = []
            # Use only the first class_num paths before repeating
            paths_to_repeat = self.class_images_path[:self.num_class_images]
            for path in paths_to_repeat:
                 repeated_class_paths.extend(itertools.repeat(path, self.repeats))
            self.class_images_path = repeated_class_paths # Replace with repeated list
            self.num_class_images = len(self.class_images_path) # Update count

            self._length = max(self.num_instance_images, self.num_class_images)
            # ... (rest of prior preservation setup)
        else:
            self.class_data_root = None
            self.num_class_images = 0 # Ensure this is 0 if no class data root

        # Transformations (defined here, applied in __getitem__)
        # Keep existing transform definitions (lines 1073-1082 and 1098-1106)
        # ... [omitted for brevity] ...

        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset
        if args.dataset_name is not None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "You are trying to load your data using the datasets library. If you wish to train using custom "
                    "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                    "local folder containing images only, specify --instance_data_dir instead."
                )
            # Downloading and loading a dataset from the hub.
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )
            # Preprocessing the datasets.
            column_names = dataset["train"].column_names

            # 6. Get the column names for input/target.
            if args.image_column is None:
                image_column = column_names[0]
                logger.info(f"image column defaulting to {image_column}")
            else:
                image_column = args.image_column
                if image_column not in column_names:
                    raise ValueError(
                        f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
            instance_images = dataset["train"][image_column]

            if args.caption_column is None:
                logger.info(
                    "No caption column provided, defaulting to instance_prompt for all images. If your dataset "
                    "contains captions/prompts for the images, make sure to specify the "
                    "column as --caption_column"
                )
                self.custom_instance_prompts = None
            else:
                if args.caption_column not in column_names:
                    raise ValueError(
                        f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
                custom_instance_prompts = dataset["train"][args.caption_column]
                # create final list of captions according to --repeats
                self.custom_instance_prompts = []
                for caption in custom_instance_prompts:
                    self.custom_instance_prompts.extend(itertools.repeat(caption, repeats))

            # ---> ADDED THIS LINE <---
            # Assign instance_images to instance_images_path
            self.instance_images_path = list(instance_images)
            print(f"DEBUG: Found {len(self.instance_images_path)} images in {dataset_name}")
            # ---> END ADDITION <---
        elif instance_data_root is not None: # Check if instance_data_root is provided
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError(f"Instance images root {self.instance_data_root} doesn't exist.")

            self.instance_images_path = list(Path(instance_data_root).iterdir()) # Store paths
            print(f"DEBUG: Found {len(self.instance_images_path)} images in {instance_data_root}")
            # instance_images = [Image.open(path) for path in self.instance_images_path] # Load later in __getitem__
            # This line belongs inside the instance_data_root handling block
            self.custom_instance_prompts = None # Assume no custom prompts with local folder
        # Removing the unnecessary 'else' block introduced earlier.
        # Argument parsing should handle the case where no data source is provided.

        # Removing the problematic block entirely to ensure it doesn't interfere.
        # --- Modification: Transformations defined here, applied in __getitem__ ---
        self.train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        # --- End modification ---

        # Count based on repeated paths
        self.num_instance_images = len(self.instance_paths_repeated)
        self._length = self.num_instance_images

        # --- DEBUG PRINT 1 ---
        print(f"DEBUG: After loading, self.num_instance_images = {self.num_instance_images}")

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            # --- Modification: Transformations for class images also defined here ---
            self.class_image_transforms = transforms.Compose(
                 [
                     transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                     transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                     transforms.ToTensor(),
                     transforms.Normalize([0.5], [0.5]),
                 ]
            )
            # --- End modification ---
        else:
            self.class_data_root = None

        # --- DEBUG PRINT 2 ---
        print(f"DEBUG: End of DreamBoothDataset.__init__, self.num_instance_images = {self.num_instance_images}")
        # --- END DEBUG ---


    def __len__(self):
        len_val = 0 # Значение по умолчанию на случай ошибки
        # The length depends on whether prior preservation is used
        # self.num_instance_images and self.num_class_images should be counts *after* repeating
        if self.class_data_root is not None:
            # The dataloader length is the maximum of instance and class images (after repeats)
            # This ensures we iterate enough times to cover the larger set if counts differ.
            # return max(self.num_instance_images, self.num_class_images)
            len_val = max(self.num_instance_images, self.num_class_images)
        else:
            # If no prior preservation, length is just the number of instance images (after repeats)
            # return self.num_instance_images
            len_val = self.num_instance_images
        # --- DEBUG PRINT 3 ---
        print(f"DEBUG: DreamBoothDataset.__len__ called, returning: {len_val}")
        # --- END DEBUG ---
        return len_val

    def _load_and_process_instance(self, index):
        """Helper function to load and process instance image, prompt, and mask."""
        image = None
        prompt = None
        mask = None
        source_info = "N/A" # For error logging

        try:
            # --- Loading from Metadata ---
            if self.metadata is not None:
                # Use the index directly as metadata list is already repeated
                item = self.metadata[index]
                source_info = item
                image_column = self.args.image_column
                caption_column = self.args.caption_column
                mask_column = self.args.mask_column

                image_path_str = item.get(image_column)
                if not image_path_str: raise ValueError(f"Missing key '{image_column}'")
                image_path = self.metadata_dir / image_path_str
                if not image_path.is_file(): raise FileNotFoundError(f"Image not found: {image_path}")
                image = Image.open(image_path)

                prompt = item.get(caption_column, self.instance_prompt)

                mask_path_str = item.get(mask_column)
                if mask_path_str:
                    mask_path = self.metadata_dir / mask_path_str
                    if mask_path.is_file():
                        try:
                            mask = Image.open(mask_path).convert("L") # Load as grayscale
                        except Exception as e:
                            logger.warning(f"Could not load/convert mask {mask_path}: {e}. Using default mask.")
                            mask = None
                    else:
                        logger.warning(f"Mask file specified but not found: {mask_path}. Using default mask.")
                        mask = None
                # If alpha_mask_required is True, check RGBA even with metadata (might override external mask)
                elif self.alpha_mask_required and image.mode == "RGBA":
                     logger.warning(f"Extracting alpha mask from RGBA image {image_path} even though metadata was provided.")
                     try:
                         alpha = image.split()[-1]
                         mask_np = np.array(alpha, dtype=np.uint8)
                         mask_np = (mask_np > 127).astype(np.uint8) # Threshold
                         mask = Image.fromarray(mask_np * 255, mode='L')
                         image = image.convert("RGB")
                     except Exception as e:
                         logger.warning(f"Could not process alpha channel for source: {image_path}. Error: {e}")
                         mask = None
                         if image.mode != "RGB": image = image.convert("RGB")

            # --- Loading from Directory/Dataset ---
            else:
                # Use the index directly as instance_paths_repeated is already repeated
                image_source = self.instance_paths_repeated[index]
                source_info = image_source
                if isinstance(image_source, Image.Image):
                    image = image_source.copy()
                elif isinstance(image_source, (str, Path)):
                    image_path = Path(image_source)
                    if not image_path.is_file(): raise FileNotFoundError(f"Image not found: {image_path}")
                    image = Image.open(image_path)
                    image = exif_transpose(image)
                else:
                    raise TypeError(f"Unexpected type for image_source: {type(image_source)}")

                # Handle prompts for directory/dataset loading
                if self.custom_instance_prompts:
                    # Ensure custom_instance_prompts was repeated correctly in __init__
                    prompt_idx = index % len(self.custom_instance_prompts) # Index within original prompts list
                    prompt = self.custom_instance_prompts[prompt_idx]
                else:
                    prompt = self.instance_prompt

                # Handle alpha mask extraction if required
                if self.alpha_mask_required and image.mode == "RGBA":
                    try:
                        alpha = image.split()[-1]
                        mask_np = np.array(alpha, dtype=np.uint8)
                        mask_np = (mask_np > 127).astype(np.uint8)
                        mask = Image.fromarray(mask_np * 255, mode='L') # Convert back to PIL Image (L mode)
                        image = image.convert("RGB") # Convert image after extracting alpha
                    except Exception as e:
                        logger.warning(f"Could not process alpha channel for source: {image_source}. Error: {e}")
                        mask = None # Reset mask on error
                        if image.mode != "RGB": image = image.convert("RGB")
                elif image.mode != "RGB":
                     image = image.convert("RGB") # Convert other modes to RGB if no alpha needed/present

            # --- Common Image/Mask Preprocessing ---
            if image is None: raise ValueError("Instance image could not be loaded.")

            if image.mode != "RGB": image = image.convert("RGB") # Final check
            if mask is not None and mask.mode != "L": mask = mask.convert("L") # Ensure mask is grayscale

            # Apply geometric transforms (Resize, Crop, Flip)
            apply_flip = self.args.random_flip and random.random() < 0.5

            # Resize
            image = self.train_resize(image)
            # Use NEAREST for mask resizing to preserve binary nature
            if mask: mask = transforms.functional.resize(mask, self.train_resize.size, interpolation=transforms.InterpolationMode.NEAREST)

            # Flip
            if apply_flip:
                image = self.train_flip(image)
                if mask: mask = self.train_flip(mask)

            # Crop
            if self.args.center_crop:
                image = self.train_crop(image)
                if mask: mask = self.train_crop(mask)
            else:
                # Get crop parameters based on image
                y1, x1, h, w = self.train_crop.get_params(image, (self.args.resolution, self.args.resolution))
                # Apply crop to image and mask with same parameters
                image = crop(image, y1, x1, h, w)
                if mask: mask = crop(mask, y1, x1, h, w)

            # Convert to Tensor and Normalize
            image_tensor = self.train_transforms(image) # ToTensor + Normalize

            # Convert Mask to Tensor
            if mask is not None:
                # Ensure mask is correct size after transforms
                if mask.size != (self.size, self.size):
                    logger.warning(f"Resizing mask to ({self.size}, {self.size}) after transforms using NEAREST.")
                    mask = mask.resize((self.size, self.size), Image.NEAREST)
                mask_tensor = transforms.functional.to_tensor(mask) # Just ToTensor, no normalization
                # Ensure mask is binary after potential interpolation artifacts
                mask_tensor = (mask_tensor > 0.5).float()
            else:
                # Create default mask (all ones)
                mask_tensor = torch.ones(1, self.size, self.size)

            return {"image": image_tensor, "prompt": prompt, "mask": mask_tensor}

        except Exception as e:
            logger.error(f"Could not load/process instance data for index {index} (source: {source_info}). Error: {e}", exc_info=True)
            return None # Signal failure

    def _load_and_process_class(self, index):
        """Helper function to load and process class image and prompt."""
        if not self.class_data_root or not self.class_images_path: return None

        class_image_path = "N/A"
        try:
            # Use index directly as class_images_path is already repeated
            class_image_path = self.class_images_path[index]
            class_image = Image.open(class_image_path)
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")

            # Apply class image transforms (defined in __init__)
            # These typically don't include random flipping
            image_tensor = self.class_image_transforms(class_image)

            return {"image": image_tensor, "prompt": self.class_prompt, "mask": None} # Mask is None for class images

        except Exception as e:
            logger.error(f"Could not load/process class image for index {index} (path: {class_image_path}). Error: {e}", exc_info=True)
            return None # Signal failure


    def __getitem__(self, index):
        example = {}
        instance_data = None
        class_data = None

        # Determine indices within the *repeated* lists
        instance_index = index % self.num_instance_images
        class_index = -1
        if self.class_data_root is not None:
             class_index = index % self.num_class_images

        # --- Load Instance Data ---
        # Always try to load instance data, as it might be needed even if index maps primarily to class
        instance_data = self._load_and_process_instance(instance_index)
        if instance_data is None:
             logger.warning(f"Skipping index {index} due to instance loading failure.")
             return None # Skip if instance loading failed

        example["instance_images"] = instance_data["image"]
        example["instance_prompt"] = instance_data["prompt"]
        example["instance_masks"] = instance_data["mask"]

        # --- Load Class Data (if prior preservation is enabled) ---
        if self.class_data_root is not None:
            class_data = self._load_and_process_class(class_index)
            if class_data is None:
                 logger.warning(f"Skipping index {index} due to class loading failure.")
                 return None # Skip if class loading failed (strict prior preservation)

            example["class_images"] = class_data["image"]
            example["class_prompt"] = class_data["prompt"]
            example["class_masks"] = class_data["mask"] # Should be None

        # Final check: ensure required keys exist based on prior preservation status
        if self.class_data_root is not None:
            # Both instance and class data must be present if prior preservation is on
            if "instance_images" not in example or "class_images" not in example:
                logger.error(f"Data missing in prior preservation mode for index {index}. Keys: {example.keys()}")
                return None
        elif "instance_images" not in example:
             # Only instance data is required if prior preservation is off
             logger.error(f"Instance data missing for index {index}. Keys: {example.keys()}")
             return None

        return example


# Need F from torch.nn.functional for interpolation
import torch.nn.functional as F

def collate_fn(examples, with_prior_preservation=False, vae_scale_factor=8):
    # Filter None values that might come from __getitem__ on errors
    examples = [e for e in examples if e is not None]
    if not examples:
        return None # Return None if the whole batch is corrupted

    # Extract data for instance images
    instance_pixel_values = [example["instance_images"] for example in examples]
    instance_prompts = [example["instance_prompt"] for example in examples]
    # instance_masks are tensors of shape [1, H, W]
    instance_masks = [example["instance_masks"] for example in examples]

    batch = {} # Initialize batch dictionary

    # Handle prior preservation
    if with_prior_preservation:
        # Filter examples where class_images loaded successfully
        valid_class_examples = [e for e in examples if e.get("class_images") is not None]
        if valid_class_examples:
             class_pixel_values = [example["class_images"] for example in valid_class_examples]
             class_prompts = [example["class_prompt"] for example in valid_class_examples]
             # Class masks are None as returned by _load_and_process_class
             class_masks = [example.get("class_masks", None) for example in valid_class_examples] # Should be list of Nones

             # Combine instance and class data
             pixel_values = instance_pixel_values + class_pixel_values
             prompts = instance_prompts + class_prompts
             # Combine instance masks and placeholders (None) for class masks
             masks_to_process = instance_masks + class_masks
        else:
             # If no valid class images, proceed without prior preservation for this batch
             logger.warning("Prior preservation enabled, but no valid class images found in this batch. Proceeding without prior preservation.")
             pixel_values = instance_pixel_values
             prompts = instance_prompts
             masks_to_process = instance_masks
             # Keep with_prior_preservation flag as True, loss calculation might still use it
    else:
        # No prior preservation, just use instance data
        pixel_values = instance_pixel_values
        prompts = instance_prompts
        masks_to_process = instance_masks


    # Stack pixel values
    batch["pixel_values"] = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()

    # Store prompts (tokenization happens later in the main loop)
    batch["prompts"] = prompts

    # --- Process and resize masks ---
    batch_masks_tensor = None
    if masks_to_process: # Check if there are any masks/placeholders to process
        processed_masks = []
        h, w = -1, -1 # Initialize height and width

        # Find the dimensions from the first available tensor (pixel_values)
        if batch["pixel_values"].numel() > 0:
             h, w = batch["pixel_values"].shape[-2:]
        else: # Should not happen if examples is not empty
             raise ValueError("Cannot determine mask size, pixel_values tensor is empty.")

        # Determine target latent size
        latent_height = h // vae_scale_factor
        latent_width = w // vae_scale_factor

        for mask_tensor in masks_to_process:
            if mask_tensor is None:
                # Create default 'all ones' mask (for class images or failed instance masks)
                # Ensure it matches the expected channel dim (1) and H, W
                processed_masks.append(torch.ones(1, h, w))
            else:
                # Ensure mask tensor has channel dim: [1, H, W]
                if mask_tensor.ndim == 2: # If mask is [H, W]
                    mask_tensor = mask_tensor.unsqueeze(0)
                # Ensure mask tensor has the correct H, W (might be redundant after __getitem__)
                if mask_tensor.shape[-2:] != (h, w):
                     logger.warning(f"Mask tensor has unexpected shape {mask_tensor.shape}, expected H={h}, W={w}. Resizing with NEAREST.")
                     mask_tensor = transforms.functional.resize(mask_tensor, [h, w], interpolation=transforms.InterpolationMode.NEAREST)

                processed_masks.append(mask_tensor)

        # Stack all masks (including defaults)
        # Ensure the number of processed masks matches the number of pixel values
        if len(processed_masks) != batch["pixel_values"].shape[0]:
             logger.error(f"Mismatch between number of pixel values ({batch['pixel_values'].shape[0]}) and processed masks ({len(processed_masks)}). Skipping mask processing for this batch.")
             batch_masks_tensor = None # Indicate failure to process masks
        else:
             batch_masks_tensor = torch.stack(processed_masks) # Shape [B, 1, H, W]

             # Resize masks to latent dimensions using nearest neighbor interpolation
             # Input shape [B, 1, H, W] -> Output shape [B, 1, latent_H, latent_W]
             resized_masks = F.interpolate(
                 batch_masks_tensor,
                 size=(latent_height, latent_width),
                 mode="nearest" # Use nearest to keep mask binary
             )

             # Threshold again after interpolation just in case (shouldn't be needed with nearest)
             final_masks = (resized_masks > 0.5).float()
             batch_masks_tensor = final_masks # Store final masks [B, 1, latent_H, latent_W]

    # Store the final processed masks (or None if processing failed)
    batch["pixel_masks"] = batch_masks_tensor

    # Remove the old flag, loss calculation should handle the batch structure
    # batch["is_prior_preservation_batch"] = with_prior_preservation

    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt, max_sequence_length, add_special_tokens=False):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    if hasattr(text_encoders[0], "module"):
        dtype = text_encoders[0].module.dtype
    else:
        dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


# --- Optimizer Helper Functions (from Markdown) ---
# Note: These might need adjustments based on the exact structure of your original script if it differs significantly.
from torch.optim import Optimizer
from typing import Callable, Tuple

def get_optimizer(args, trainable_params) -> tuple[str, str, object]:
    """
    Optimizer to use:
    AdamW, AdamW8bit, Lion, SGDNesterov, SGDNesterov8bit, PagedAdamW, PagedAdamW8bit, PagedAdamW32bit, Lion8bit, PagedLion8bit, Adan, Adafactor
    """
    optimizer_type = args.optimizer.lower()
    optimizer_kwargs = {} # Add parsing for args.optimizer_args if needed
    lr = args.learning_rate

    if optimizer_type == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_class = bnb.optim.AdamW8bit
                logger.info("Using 8-bit AdamW optimizer")
            except ImportError:
                 raise ImportError("Please install bitsandbytes to use 8-bit AdamW")
        else:
            optimizer_class = torch.optim.AdamW
            logger.info("Using AdamW optimizer")
        optimizer = optimizer_class(
             trainable_params,
             lr=lr,
             betas=(args.adam_beta1, args.adam_beta2),
             weight_decay=args.adam_weight_decay,
             eps=args.adam_epsilon,
             **optimizer_kwargs
        )
        optimizer_name = optimizer_class.__name__
        optimizer_args_str = str(optimizer_kwargs) # Simplified
    elif optimizer_type == "prodigy":
        try:
            import prodigyopt
            optimizer_class = prodigyopt.Prodigy
            logger.info("Using Prodigy optimizer")
        except ImportError:
            raise ImportError("Please install prodigyopt to use Prodigy optimizer")
        optimizer = optimizer_class(
             trainable_params,
             lr=lr, # Prodigy often recommends lr=1.0
             betas=(args.adam_beta1, args.adam_beta2),
             beta3=args.prodigy_beta3,
             weight_decay=args.adam_weight_decay,
             eps=args.adam_epsilon,
             decouple=args.prodigy_decouple,
             use_bias_correction=args.prodigy_use_bias_correction,
             safeguard_warmup=args.prodigy_safeguard_warmup,
             **optimizer_kwargs
        )
        optimizer_name = optimizer_class.__name__
        optimizer_args_str = str(optimizer_kwargs) # Simplified
    # Add other optimizers from your original get_optimizer if needed
    else:
         raise ValueError(f"Unsupported optimizer type: {args.optimizer}")

    return optimizer_name, optimizer_args_str, optimizer

def is_schedulefree_optimizer(optimizer: Optimizer, args: argparse.Namespace) -> bool:
    # Add implementation from your script if it exists, otherwise assume based on name
    return args.optimizer.lower().endswith("schedulefree")

def get_optimizer_train_eval_fn(optimizer: Optimizer, args: argparse.Namespace) -> Tuple[Callable, Callable]:
    # Add implementation from your script if it exists
    if not is_schedulefree_optimizer(optimizer, args):
        return lambda: None, lambda: None
    # Assuming schedulefree optimizers have train/eval methods
    # This might need adjustment based on the specific schedulefree library used
    train_fn = getattr(optimizer, "train", lambda: None)
    eval_fn = getattr(optimizer, "eval", lambda: None)
    return train_fn, eval_fn
# --- End Optimizer Helper Functions ---


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # ---> ADDED BLOCK HERE <---
    if accelerator.is_main_process: # Log only on the main process
         if args.alpha_mask and args.masked_loss:
              logger.warning("Both --alpha_mask and --masked_loss are specified. Using alpha channel implicitly.")
         elif args.alpha_mask:
              logger.info("Using alpha channel from RGBA images for loss masking.")
         elif args.masked_loss:
              logger.info("Using alpha channel from RGBA images for loss masking (triggered by --masked_loss).")
    # ---> END ADDED BLOCK <---
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            has_supported_fp16_accelerator = torch.cuda.is_available() or torch.backends.mps.is_available()
            torch_dtype = torch.float16 if has_supported_fp16_accelerator else torch.float32
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = FluxPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                revision=args.revision,
                variant=args.variant,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            free_memory()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        model_id = args.hub_model_id or Path(args.output_dir).name
        repo_id = None
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=model_id,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    # --- Determine VAE stride (scale factor) ---
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    logger.info(f"Detected VAE scale factor (stride): {vae_scale_factor}")
    # --- End determination ---

    if args.train_text_encoder_ti:
        # we parse the provided token identifier (or identifiers) into a list. s.t. - "TOK" -> ["TOK"], "TOK,
        # TOK2" -> ["TOK", "TOK2"] etc.
        token_abstraction_list = [place_holder.strip() for place_holder in re.split(r",\s*", args.token_abstraction)]
        logger.info(f"list of token identifiers: {token_abstraction_list}")

        if args.initializer_concept is None:
            num_new_tokens_per_abstraction = (
                2 if args.num_new_tokens_per_abstraction is None else args.num_new_tokens_per_abstraction
            )
        # if args.initializer_concept is provided, we ignore args.num_new_tokens_per_abstraction
        else:
            token_ids = tokenizer_one.encode(args.initializer_concept, add_special_tokens=False)
            num_new_tokens_per_abstraction = len(token_ids)
            if args.enable_t5_ti:
                token_ids_t5 = tokenizer_two.encode(args.initializer_concept, add_special_tokens=False)
                num_new_tokens_per_abstraction = max(len(token_ids), len(token_ids_t5))
            logger.info(
                f"initializer_concept: {args.initializer_concept}, num_new_tokens_per_abstraction: {num_new_tokens_per_abstraction}"
            )

        token_abstraction_dict = {}
        token_idx = 0
        for i, token in enumerate(token_abstraction_list):
            token_abstraction_dict[token] = [f"<s{token_idx + i + j}>" for j in range(num_new_tokens_per_abstraction)]
            token_idx += num_new_tokens_per_abstraction - 1

        # replace instances of --token_abstraction in --instance_prompt with the new tokens: "<si><si+1>" etc.
        for token_abs, token_replacement in token_abstraction_dict.items():
            new_instance_prompt = args.instance_prompt.replace(token_abs, "".join(token_replacement))
            if args.instance_prompt == new_instance_prompt:
                logger.warning(
                    "Note! the instance prompt provided in --instance_prompt does not include the token abstraction specified "
                    "--token_abstraction. This may lead to incorrect optimization of text embeddings during pivotal tuning"
                )
            args.instance_prompt = new_instance_prompt
            if args.with_prior_preservation:
                args.class_prompt = args.class_prompt.replace(token_abs, "".join(token_replacement))
            if args.validation_prompt:
                args.validation_prompt = args.validation_prompt.replace(token_abs, "".join(token_replacement))

        # initialize the new tokens for textual inversion
        text_encoders = [text_encoder_one, text_encoder_two] if args.enable_t5_ti else [text_encoder_one]
        tokenizers = [tokenizer_one, tokenizer_two] if args.enable_t5_ti else [tokenizer_one]
        embedding_handler = TokenEmbeddingsHandler(text_encoders, tokenizers)
        inserting_toks = []
        for new_tok in token_abstraction_dict.values():
            inserting_toks.extend(new_tok)
        embedding_handler.initialize_new_tokens(inserting_toks=inserting_toks)

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
    # now we will add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha if args.lora_alpha else args.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)
    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha if args.lora_alpha else args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                elif isinstance(model, type(unwrap_model(text_encoder_one))):
                    if args.train_text_encoder:  # when --train_text_encoder_ti we don't save the layers
                        text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(model)
                elif isinstance(model, type(unwrap_model(text_encoder_two))):
                    pass  # when --train_text_encoder_ti and --enable_t5_ti we don't save the layers
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            FluxPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            )
        if args.train_text_encoder_ti:
            embedding_handler.save_embeddings(f"{args.output_dir}/{Path(args.output_dir).name}_emb.safetensors")

    def load_model_hook(models, input_dir):
        transformer_ = None
        text_encoder_one_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = FluxPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
        if args.train_text_encoder:
            # Do we need to call `scale_lora_layers()` here?
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_])
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        if args.train_text_encoder:
            models.extend([text_encoder_one])
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    if args.train_text_encoder:
        text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
    # if we use textual inversion, we freeze all parameters except for the token embeddings
    # in text encoder
    elif args.train_text_encoder_ti:
        text_lora_parameters_one = []  # CLIP
        for name, param in text_encoder_one.named_parameters():
            if "token_embedding" in name:
                # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
                if args.mixed_precision == "fp16":
                    param.data = param.to(dtype=torch.float32)
                param.requires_grad = True
                text_lora_parameters_one.append(param)
            else:
                param.requires_grad = False
        if args.enable_t5_ti:  # whether to do pivotal tuning/textual inversion for T5 as well
            text_lora_parameters_two = []
            for name, param in text_encoder_two.named_parameters():
                if "shared" in name:
                    # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
                    if args.mixed_precision == "fp16":
                        param.data = param.to(dtype=torch.float32)
                    param.requires_grad = True
                    text_lora_parameters_two.append(param)
                else:
                    param.requires_grad = False

    # If neither --train_text_encoder nor --train_text_encoder_ti, text_encoders remain frozen during training
    freeze_text_encoder = not (args.train_text_encoder or args.train_text_encoder_ti)

    # if --train_text_encoder_ti and train_transformer_frac == 0 where essentially performing textual inversion
    # and not training transformer LoRA layers
    pure_textual_inversion = args.train_text_encoder_ti and args.train_transformer_frac == 0

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    if not freeze_text_encoder:
        # different learning rate for text encoder and transformer
        text_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": args.adam_weight_decay_text_encoder
            if args.adam_weight_decay_text_encoder
            else args.adam_weight_decay,
            "lr": args.text_encoder_lr,
        }
        if not args.enable_t5_ti:
            # pure textual inversion - only clip
            if pure_textual_inversion:
                params_to_optimize = [text_parameters_one_with_lr]
                te_idx = 0
            else:  # regular te training or regular pivotal for clip
                params_to_optimize = [transformer_parameters_with_lr, text_parameters_one_with_lr]
                te_idx = 1
        elif args.enable_t5_ti:
            # pivotal tuning of clip & t5
            text_parameters_two_with_lr = {
                "params": text_lora_parameters_two,
                "weight_decay": args.adam_weight_decay_text_encoder
                if args.adam_weight_decay_text_encoder
                else args.adam_weight_decay,
                "lr": args.text_encoder_lr,
            }
            # pure textual inversion - only clip & t5
            if pure_textual_inversion:
                params_to_optimize = [text_parameters_one_with_lr, text_parameters_two_with_lr]
                te_idx = 0
            else:  # regular pivotal tuning of clip & t5
                params_to_optimize = [
                    transformer_parameters_with_lr,
                    text_parameters_one_with_lr,
                    text_parameters_two_with_lr,
                ]
                te_idx = 1
    else:
        params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    optimizer_name, optimizer_args_str, optimizer = get_optimizer(args, params_to_optimize)
    # optimizer_train_fn, optimizer_eval_fn = get_optimizer_train_eval_fn(optimizer, args) # Not used in current loop

    # Dataset and DataLoaders creation:
    # --- Modification: Pass alpha_mask_required ---
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        train_text_encoder_ti=args.train_text_encoder_ti,
        token_abstraction_dict=token_abstraction_dict if args.train_text_encoder_ti else None,
        class_prompt=args.class_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_num=args.num_class_images,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
        alpha_mask_required=args.masked_loss or args.alpha_mask, # Enable if either flag is active
        args=args
    )
    # --- End modification ---

    # --- DEBUG PRINT 4 ---
    print(f"DEBUG: Before DataLoader creation, len(train_dataset) = {len(train_dataset)}")
    # --- END DEBUG ---
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        # collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation), # Replaced
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation, vae_scale_factor), # Use wrapper
        num_workers=args.dataloader_num_workers,
    )

    if freeze_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoders, tokenizers, prompt, args.max_sequence_length
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                text_ids = text_ids.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds, text_ids

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if freeze_text_encoder and not train_dataset.custom_instance_prompts:
        instance_prompt_hidden_states, instance_pooled_prompt_embeds, instance_text_ids = compute_text_embeddings(
            args.instance_prompt, text_encoders, tokenizers
        )

    # Handle class prompt for prior-preservation.
    if args.with_prior_preservation:
        if freeze_text_encoder:
            class_prompt_hidden_states, class_pooled_prompt_embeds, class_text_ids = compute_text_embeddings(
                args.class_prompt, text_encoders, tokenizers
            )

    # Clear the memory here
    if freeze_text_encoder and not train_dataset.custom_instance_prompts:
        del tokenizers, text_encoders, text_encoder_one, text_encoder_two
        free_memory()

    # if --train_text_encoder_ti we need add_special_tokens to be True for textual inversion
    add_special_tokens_clip = True if args.train_text_encoder_ti else False
    add_special_tokens_t5 = True if (args.train_text_encoder_ti and args.enable_t5_ti) else False

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.

    if not train_dataset.custom_instance_prompts:
        if freeze_text_encoder:
            prompt_embeds = instance_prompt_hidden_states
            pooled_prompt_embeds = instance_pooled_prompt_embeds
            text_ids = instance_text_ids
            if args.with_prior_preservation:
                prompt_embeds = torch.cat([prompt_embeds, class_prompt_hidden_states], dim=0)
                pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, class_pooled_prompt_embeds], dim=0)
                text_ids = torch.cat([text_ids, class_text_ids], dim=0)
        # if we're optimizing the text encoder (both if instance prompt is used for all images or custom prompts)
        # we need to tokenize and encode the batch prompts on all training steps
        else:
            tokens_one = tokenize_prompt(
                tokenizer_one, args.instance_prompt, max_sequence_length=77, add_special_tokens=add_special_tokens_clip
            )
            tokens_two = tokenize_prompt(
                tokenizer_two,
                args.instance_prompt,
                max_sequence_length=args.max_sequence_length,
                add_special_tokens=add_special_tokens_t5,
            )
            if args.with_prior_preservation:
                class_tokens_one = tokenize_prompt(
                    tokenizer_one,
                    args.class_prompt,
                    max_sequence_length=77,
                    add_special_tokens=add_special_tokens_clip,
                )
                class_tokens_two = tokenize_prompt(
                    tokenizer_two,
                    args.class_prompt,
                    max_sequence_length=args.max_sequence_length,
                    add_special_tokens=add_special_tokens_t5,
                )
                tokens_one = torch.cat([tokens_one, class_tokens_one], dim=0)
                tokens_two = torch.cat([tokens_two, class_tokens_two], dim=0)

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels # Restore this line

    # --- Modification: Latent and mask caching logic ---
    latents_cache = []
    masks_cache = [] # Additional cache for masks
    if args.cache_latents:
        logger.info("Caching latents and masks...")
        # VAE must be on the correct device for caching
        vae.to(accelerator.device, dtype=weight_dtype)
        # Wrap train_dataloader in tqdm only if not main process to avoid duplication
        dataloader_iterator = tqdm(train_dataloader, desc="Caching latents & masks", disable=not accelerator.is_local_main_process)
        for batch in dataloader_iterator:
            # Skip batch if collate_fn returned None
            if batch is None:
                 logger.warning("Skipping a batch during caching due to collation error (likely image loading issue).")
                 continue
            with torch.no_grad():
                pixel_values = batch["pixel_values"].to(accelerator.device, non_blocking=True, dtype=vae.dtype)
                latent_dist = vae.encode(pixel_values).latent_dist
                latents_cache.append(latent_dist)

                # Cache the mask (it's already the correct size from collate_fn)
                if "alpha_masks" in batch:
                    masks_cache.append(batch["alpha_masks"].cpu()) # Move to CPU for cache
                else:
                    masks_cache.append(None) # If no mask was present

        # Check if cache size matches dataloader length (might differ if batches were skipped)
        if len(latents_cache) != len(train_dataloader):
             logger.warning(f"Number of cached items ({len(latents_cache)}) does not match dataloader length ({len(train_dataloader)}). This might happen if some batches were skipped.")
             # More complex indexing or dataloader filtering might be needed depending on desired behavior

        # Move VAE back to CPU if no longer needed in the loop
        if args.validation_prompt is None: # If no validation, VAE definitely not needed
            vae.to("cpu")
            free_memory()
        logger.info("Latents and masks cached.")
        # Check if cache size matches dataloader length (might differ if batches were skipped)
        if len(latents_cache) != len(train_dataloader):
             logger.warning(f"Number of cached items ({len(latents_cache)}) does not match dataloader length ({len(train_dataloader)}). This might happen if some batches were skipped.")
             # More complex indexing or dataloader filtering might be needed depending on desired behavior
    # --- End modification ---

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if not freeze_text_encoder:
        if args.enable_t5_ti:
            (
                transformer,
                text_encoder_one,
                text_encoder_two,
                optimizer,
                train_dataloader,
                lr_scheduler,
            ) = accelerator.prepare(
                transformer,
                text_encoder_one,
                text_encoder_two,
                optimizer,
                train_dataloader,
                lr_scheduler,
            )
        else:
            print("I SHOULD BE HERE")
            transformer, text_encoder_one, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                transformer, text_encoder_one, optimizer, train_dataloader, lr_scheduler
            )

    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "dreambooth-flux-dev-lora-advanced"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    if args.train_text_encoder:
        num_train_epochs_text_encoder = int(args.train_text_encoder_frac * args.num_train_epochs)
        num_train_epochs_transformer = int(args.train_transformer_frac * args.num_train_epochs)
    elif args.train_text_encoder_ti:  # args.train_text_encoder_ti
        num_train_epochs_text_encoder = int(args.train_text_encoder_ti_frac * args.num_train_epochs)
        num_train_epochs_transformer = int(args.train_transformer_frac * args.num_train_epochs)

    # flag used for textual inversion
    pivoted_te = False
    pivoted_tr = False
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        # if performing any kind of optimization of text_encoder params
        if args.train_text_encoder or args.train_text_encoder_ti:
            if epoch == num_train_epochs_text_encoder:
                # flag to stop text encoder optimization
                logger.info(f"PIVOT TE {epoch}")
                pivoted_te = True
            else:
                # still optimizing the text encoder
                if args.train_text_encoder:
                    text_encoder_one.train()
                    # set top parameter requires_grad = True for gradient checkpointing works
                    unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)
                elif args.train_text_encoder_ti:  # textual inversion / pivotal tuning
                    text_encoder_one.train()
                if args.enable_t5_ti:
                    text_encoder_two.train()

            if epoch == num_train_epochs_transformer:
                # flag to stop transformer optimization
                logger.info(f"PIVOT TRANSFORMER {epoch}")
                pivoted_tr = True

        # Wrap train_dataloader in tqdm only if not main process
        dataloader_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_local_main_process)

        for step, batch in enumerate(dataloader_iterator):
            # Skip batch if collate_fn returned None
            if batch is None:
                 logger.warning(f"Skipping step {step} in epoch {epoch+1} due to batch collation error.")
                 # Should we update progress_bar? No, because optimizer step isn't taken.
                 continue
            models_to_accumulate = [transformer]
            if not freeze_text_encoder:
                models_to_accumulate.extend([text_encoder_one])
                if args.enable_t5_ti:
                    models_to_accumulate.extend([text_encoder_two])
            if pivoted_te:
                # stopping optimization of text_encoder params
                optimizer.param_groups[te_idx]["lr"] = 0.0
                optimizer.param_groups[-1]["lr"] = 0.0
            elif pivoted_tr and not pure_textual_inversion:
                logger.info(f"PIVOT TRANSFORMER {epoch}")
                optimizer.param_groups[0]["lr"] = 0.0

            with accelerator.accumulate(models_to_accumulate):
                prompts = batch["prompts"]

                # encode batch prompts when custom prompts are provided for each image -
                if train_dataset.custom_instance_prompts:
                    elems_to_repeat = 1
                    if freeze_text_encoder:
                        prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                            prompts, text_encoders, tokenizers
                        )
                    else:
                        tokens_one = tokenize_prompt(
                            tokenizer_one, prompts, max_sequence_length=77, add_special_tokens=add_special_tokens_clip
                        )
                        tokens_two = tokenize_prompt(
                            tokenizer_two,
                            prompts,
                            max_sequence_length=args.max_sequence_length,
                            add_special_tokens=add_special_tokens_t5,
                        )
                else:
                    elems_to_repeat = len(prompts)

                if not freeze_text_encoder:
                    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=[None, None],
                        text_input_ids_list=[
                            tokens_one.repeat(elems_to_repeat, 1),
                            tokens_two.repeat(elems_to_repeat, 1),
                        ],
                        max_sequence_length=args.max_sequence_length,
                        device=accelerator.device,
                        prompt=prompts,
                    )
                # Convert images to latent space
                if args.cache_latents:
                    # --- Modification: Retrieve latents and masks from cache ---
                    # Use global_step % len(cache) for cyclic access if cache is smaller than steps
                    cache_idx = (initial_global_step + progress_bar.n) % len(latents_cache)
                    cached_data = latents_cache[cache_idx]
                    mask_data = masks_cache[cache_idx]

                    # Assume cache stores latent_dist
                    model_input = cached_data.sample()
                    if mask_data is not None:
                         batch["alpha_masks"] = mask_data.to(accelerator.device) # Add mask to batch
                    # --- End modification ---
                else:
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    # Ensure VAE is on the correct device
                    if vae.device != accelerator.device: vae.to(accelerator.device)
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                # --- Redundant mask resizing block removed (now done in collate_fn) ---


                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )

                # handle guidance
                if unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - model_input

                # Check if prior preservation is active for this batch
                # Assumes collate_fn correctly doubled the batch size if prior preservation was successful
                is_prior_batch = with_prior_preservation and model_pred.shape[0] == target.shape[0] * 2

                # Get masks from batch (should exist, potentially as all ones)
                pixel_masks = batch.get("pixel_masks") # Use the new key from collate_fn
                if pixel_masks is None:
                     # Fallback: create all-ones mask if missing from batch for some reason
                     logger.warning("pixel_masks not found in batch, creating default all-ones mask.")
                     latent_h, latent_w = model_pred.shape[-2:] # Get shape from prediction
                     # Ensure batch size matches model_pred batch size
                     mask_batch_size = model_pred.shape[0]
                     pixel_masks = torch.ones(mask_batch_size, 1, latent_h, latent_w, device=model_pred.device, dtype=model_pred.dtype)
                else:
                     pixel_masks = pixel_masks.to(model_pred.device, dtype=model_pred.dtype) # Ensure device and dtype match

                if is_prior_batch:
                    # Chunk predictions, targets, and masks for prior preservation loss calculation
                    # Ensure chunking is possible (batch size must be even)
                    if model_pred.shape[0] % 2 != 0:
                         logger.error(f"Prior preservation active but effective batch size ({model_pred.shape[0]}) is odd. Cannot compute prior loss correctly. Skipping batch.")
                         continue # Skip this entire batch/step

                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    pixel_masks, prior_masks = torch.chunk(pixel_masks, 2, dim=0) # pixel_masks now holds the instance part

                    # Compute prior loss (per pixel, weighted)
                    prior_loss_pixel = (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2)

                    # Apply mask to prior loss (prior_masks should be all ones from collate_fn)
                    masked_prior_loss_pixel = prior_loss_pixel * prior_masks

                    # Calculate mean loss over unmasked pixels for prior loss
                    # Sum over all dimensions except batch, divide by sum of mask elements per batch item
                    prior_loss = masked_prior_loss_pixel.sum(dim=[1, 2, 3]) / prior_masks.sum(dim=[1, 2, 3]).clamp(min=1.0)
                    prior_loss = prior_loss.mean() # Average over batch
                else:
                    # No prior preservation for this batch
                    prior_loss = 0.0 # Initialize prior_loss to 0


                # Compute instance loss (per pixel, weighted)
                # model_pred and target are already the instance parts if is_prior_batch is True
                loss_pixel = (weighting.float() * (model_pred.float() - target.float()) ** 2)

                # Apply mask to instance loss
                # pixel_masks holds the correct mask (either instance part or full batch)
                masked_loss_pixel = loss_pixel * pixel_masks

                # Calculate mean loss over unmasked pixels for instance loss
                loss = masked_loss_pixel.sum(dim=[1, 2, 3]) / pixel_masks.sum(dim=[1, 2, 3]).clamp(min=1.0)
                loss = loss.mean() # Average over batch

                if is_prior_batch:
                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # (existing code to determine params_to_clip based on flags) ...
                    params_to_clip = []
                    if not pure_textual_inversion: params_to_clip.extend(transformer.parameters())
                    if not freeze_text_encoder:
                         params_to_clip.extend(text_encoder_one.parameters())
                         if args.enable_t5_ti: params_to_clip.extend(text_encoder_two.parameters())
                    params_to_clip = list(filter(lambda p: p.requires_grad, params_to_clip)) # Ensure we only clip trainable params

                    if params_to_clip: # Check if there's anything to clip
                         accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # every step, we reset the embeddings to the original embeddings.
                if args.train_text_encoder_ti:
                    embedding_handler.retract_embeddings()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        if args.train_text_encoder_ti:
                            embedding_handler.save_embeddings(
                                f"{args.output_dir}/{Path(args.output_dir).name}_emb_checkpoint_{global_step}.safetensors"
                            )
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                # create pipeline
                if freeze_text_encoder:  # no text encoder one, two optimizations
                    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
                    text_encoder_one.to(weight_dtype)
                    text_encoder_two.to(weight_dtype)
                pipeline = FluxPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    text_encoder=unwrap_model(text_encoder_one),
                    text_encoder_2=unwrap_model(text_encoder_two),
                    transformer=unwrap_model(transformer),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                pipeline_args = {"prompt": args.validation_prompt}
                images = log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=pipeline_args,
                    epoch=epoch,
                    torch_dtype=weight_dtype,
                )
                if freeze_text_encoder:
                    del text_encoder_one, text_encoder_two
                    free_memory()

                images = None
                del pipeline

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        if args.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)

        if args.train_text_encoder:
            text_encoder_one = unwrap_model(text_encoder_one)
            text_encoder_lora_layers = get_peft_model_state_dict(text_encoder_one.to(torch.float32))
        else:
            text_encoder_lora_layers = None

        if not pure_textual_inversion:
            FluxPipeline.save_lora_weights(
                save_directory=args.output_dir,
                transformer_lora_layers=transformer_lora_layers,
                text_encoder_lora_layers=text_encoder_lora_layers,
            )

        if args.train_text_encoder_ti:
            embeddings_path = f"{args.output_dir}/{os.path.basename(args.output_dir)}_emb.safetensors"
            embedding_handler.save_embeddings(embeddings_path)

        # Final inference
        # Load previous pipeline
        pipeline = FluxPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        if not pure_textual_inversion:
            # load attention processors
            pipeline.load_lora_weights(args.output_dir)

        # run inference
        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            pipeline_args = {"prompt": args.validation_prompt}
            images = log_validation(
                pipeline=pipeline,
                args=args,
                accelerator=accelerator,
                pipeline_args=pipeline_args,
                epoch=epoch,
                is_final_validation=True,
                torch_dtype=weight_dtype,
            )

        save_model_card(
            model_id if not args.push_to_hub else repo_id,
            images=images,
            base_model=args.pretrained_model_name_or_path,
            train_text_encoder=args.train_text_encoder,
            train_text_encoder_ti=args.train_text_encoder_ti,
            enable_t5_ti=args.enable_t5_ti,
            pure_textual_inversion=pure_textual_inversion,
            token_abstraction_dict=train_dataset.token_abstraction_dict,
            instance_prompt=args.instance_prompt,
            validation_prompt=args.validation_prompt,
            repo_folder=args.output_dir,
        )
        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        images = None
        del pipeline

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
