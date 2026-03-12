# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import warnings
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import pyrallis
import torch
import torch.nn as nn
import gc

# Ottimizzazioni memoria CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
torch.backends.cuda.max_memory_split_size = 128 * 1024 * 1024  # 128 MB
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

warnings.filterwarnings("ignore")  # ignore warning


from diffusion import DPMS, FlowEuler
from diffusion.data.datasets.utils import (
    ASPECT_RATIO_512_TEST,
    ASPECT_RATIO_1024_TEST,
    ASPECT_RATIO_2048_TEST,
    ASPECT_RATIO_4096_TEST,
)
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode, vae_encode
from diffusion.model.utils import get_weight_dtype, prepare_prompt_ar, resize_and_crop_tensor
from diffusion.utils.config import SanaConfig, model_init_config
from diffusion.utils.logger import get_root_logger

# from diffusion.utils.misc import read_config
from tools.download import find_model


def guidance_type_select(default_guidance_type, pag_scale, attn_type):
    guidance_type = default_guidance_type
    if not (pag_scale > 1.0 and attn_type == "linear"):
        guidance_type = "classifier-free"
    elif pag_scale > 1.0 and attn_type == "linear":
        guidance_type = "classifier-free_PAG"
    return guidance_type


def classify_height_width_bin(height: int, width: int, ratios: dict) -> Tuple[int, int]:
    """Returns binned height and width."""
    ar = float(height / width)
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
    default_hw = ratios[closest_ratio]
    return int(default_hw[0]), int(default_hw[1])


@dataclass
class SanaInference(SanaConfig):
    config: Optional[str] = "configs/sana_config/1024ms/Sana_1600M_img1024.yaml"  # config
    model_path: str = field(
        default="output/Sana_D20/SANA.pth", metadata={"help": "Path to the model file (positional)"}
    )
    output: str = "./output"
    bs: int = 1
    image_size: int = 1024
    cfg_scale: float = 4.5
    pag_scale: float = 1.0
    seed: int = 42
    step: int = -1
    custom_image_size: Optional[int] = None
    shield_model_path: str = field(
        default="google/shieldgemma-2b",
        metadata={"help": "The path to shield model, we employ ShieldGemma-2B by default."},
    )


class SanaPipeline(nn.Module):
    def __init__(
        self,
        config: Optional[str] = "configs/sana_config/1024ms/Sana_1600M_img1024.yaml",
    ):
        super().__init__()
        
        def print_gpu_memory(msg):
            if torch.cuda.is_available():
                print(f"[MEMORY] {msg}: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
        
        print_gpu_memory("Initial")
        config = pyrallis.load(SanaInference, open(config))
        self.args = self.config = config

        # set some hyper-parameters
        self.image_size = self.config.model.image_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger = get_root_logger()
        self.logger = logger
        self.progress_fn = lambda progress, desc: None

        self.latent_size = self.image_size // config.vae.vae_downsample_rate
        self.max_sequence_length = config.text_encoder.model_max_length
        self.flow_shift = config.scheduler.flow_shift
        guidance_type = "classifier-free"

        weight_dtype = get_weight_dtype(config.model.mixed_precision)
        self.weight_dtype = weight_dtype
        self.vae_dtype = get_weight_dtype(config.vae.weight_dtype)

        self.base_ratios = eval(f"ASPECT_RATIO_{self.image_size}_TEST")
        self.vis_sampler = self.config.scheduler.vis_sampler
        logger.info(f"Sampler {self.vis_sampler}, flow_shift: {self.flow_shift}")
        self.guidance_type = guidance_type_select(guidance_type, self.args.pag_scale, config.model.attn_type)
        logger.info(f"Inference with {self.weight_dtype}, PAG guidance layer: {self.config.model.pag_applied_layers}")

        # 1. build vae
        print_gpu_memory("Before VAE")
        self.vae = self.build_vae(config.vae)
        print_gpu_memory("After VAE")
        
        # 2. build Sana model
        print_gpu_memory("Before Sana Model")
        self.model = self.build_sana_model(config).to(self.device)
        print_gpu_memory("After Sana Model")

        # 3. Initialize scheduler
        print_gpu_memory("Before Scheduler")
        if self.vis_sampler == "flow_dpm-solver":
            self.scheduler = DPMS(
                self.model,
                model_type="flow",
                schedule="FLOW",
            )
            self.scheduler.register_progress_bar(self.progress_fn)
        else:
            raise NotImplementedError("Image-to-image attualmente supportato solo per flow_dpm-solver")
        print_gpu_memory("After Scheduler")

        # Store config for later use
        self.text_encoder_config = config.text_encoder
        self.tokenizer = None
        self.text_encoder = None
        self.null_caption_embs = None

    def ensure_text_encoder_loaded(self):
        """Ensure text encoder is loaded, loading it if necessary"""
        if self.text_encoder is None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            print("[MEMORY] Loading text encoder...")
            self.tokenizer, self.text_encoder = self.build_text_encoder(self.text_encoder_config)
            
            # Pre-compute null embedding
            with torch.no_grad():
                null_caption_token = self.tokenizer(
                    "", max_length=self.max_sequence_length, padding="max_length", truncation=True, return_tensors="pt"
                ).to(self.device)
                self.null_caption_embs = self.text_encoder(null_caption_token.input_ids, null_caption_token.attention_mask)[0]
                del null_caption_token
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def unload_text_encoder(self):
        """Unload text encoder to free memory"""
        if self.text_encoder is not None:
            print("[MEMORY] Unloading text encoder...")
            del self.text_encoder
            self.text_encoder = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    def build_vae(self, config):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        vae = get_vae(config.vae_type, config.vae_pretrained, self.device).to(self.vae_dtype)
        vae.eval()  # Set to evaluation mode
        return vae

    def build_text_encoder(self, config):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        tokenizer, text_encoder = get_tokenizer_and_text_encoder(name=config.text_encoder_name, device=self.device)
        text_encoder.eval()  # Set to evaluation mode
        return tokenizer, text_encoder

    def build_sana_model(self, config):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        # model setting
        model_kwargs = model_init_config(config, latent_size=self.latent_size)
        model = build_model(
            config.model.model,
            use_fp32_attention=config.model.get("fp32_attention", False) and config.model.mixed_precision != "bf16",
            **model_kwargs,
        )
        model.eval()  # Set to evaluation mode
        
        self.logger.info(f"use_fp32_attention: {model.fp32_attention}")
        self.logger.info(
            f"{model.__class__.__name__}:{config.model.model},"
            f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}"
        )
        return model

    def from_pretrained(self, model_path):
        state_dict = find_model(model_path)
        state_dict = state_dict.get("state_dict", state_dict)
        if "pos_embed" in state_dict:
            del state_dict["pos_embed"]
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        self.model.eval().to(self.weight_dtype)

        self.logger.info("Generating sample from ckpt: %s" % model_path)
        self.logger.warning(f"Missing keys: {missing}")
        self.logger.warning(f"Unexpected keys: {unexpected}")

    def register_progress_bar(self, progress_fn=None):
        self.progress_fn = progress_fn if progress_fn is not None else self.progress_fn

    @torch.inference_mode()
    def forward(
        self,
        prompt=None,
        height=1024,
        width=1024,
        input_image: Optional[torch.Tensor] = None,
        strength: float = 0.5,
        negative_prompt="",
        num_inference_steps=20,
        guidance_scale=4.5,
        pag_guidance_scale=1.0,
        num_images_per_prompt=1,
        generator=torch.Generator().manual_seed(42),
        latents=None,
        use_resolution_binning=True,
    ):
        # Clear CUDA cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        self.ori_height, self.ori_width = height, width
        if use_resolution_binning:
            self.height, self.width = classify_height_width_bin(height, width, ratios=self.base_ratios)
        else:
            self.height, self.width = height, width

        # Process batch
        batch_size = 1
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        # Process the image in smaller batches
        max_batch_size = 4  # Ridotto il batch size massimo
        results = []
        
        for i in range(0, num_images_per_prompt, max_batch_size):
            current_batch_size = min(max_batch_size, num_images_per_prompt - i)
            
            # Process text embeddings for current batch
            text_embeddings = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                batch_size=current_batch_size,
            )

            # Prepare latents
            latents_shape = (current_batch_size, self.vae.config.latent_channels, self.height // 8, self.width // 8)
            
            if latents is None:
                latents = torch.randn(
                    latents_shape,
                    generator=generator,
                    device=self.device,
                    dtype=self.vae_dtype,
                )
            else:
                latents = latents.to(device=self.device, dtype=self.vae_dtype)

            # Process input image if provided
            if input_image is not None:
                # Clear memory before processing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Process in smaller chunks if batch size is large
                chunk_size = 2  # Process 2 images at a time
                all_latents = []
                
                for j in range(0, current_batch_size, chunk_size):
                    chunk_batch_size = min(chunk_size, current_batch_size - j)
                    
                    # Encode image chunk
                    init_latents_chunk = self.encode_image(input_image[j:j+chunk_batch_size], chunk_batch_size)
                    init_latents_chunk = init_latents_chunk.to(device=self.device, dtype=self.vae_dtype)
                    
                    # Generate noise for chunk
                    noise_chunk = torch.randn(
                        init_latents_chunk.shape,
                        generator=generator,
                        device=self.device,
                        dtype=self.vae_dtype
                    )
                    
                    # Add noise to chunk
                    noised_chunk = self.scheduler.add_noise2(
                        init_latents_chunk,
                        noise_chunk,
                        torch.tensor([0.0]).to(self.device)
                    )
                    
                    all_latents.append(noised_chunk)
                    
                    # Clean up chunk memory
                    del init_latents_chunk, noise_chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                
                # Combine chunks
                latents = torch.cat(all_latents, dim=0)
                del all_latents
                
                print(f"[DEBUG] z dopo add_noise: shape={latents.shape}, type={type(latents)}")
                
                # Final cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

            # Scale the latents
            latents = latents * self.scheduler.edm_sigma(torch.tensor([0.0]).to(self.device))
            print(f"[DEBUG] z.shape: {latents.shape}, type: {type(latents)}")

            # Set timesteps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps

            # Denoising loop
            for t in timesteps:
                # Free up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

                latent_model_input = torch.cat([latents] * 2)
                t_input = torch.cat([t] * 2)

                # Predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        sample=latent_model_input,
                        timestep=t_input,
                        encoder_hidden_states=text_embeddings,
                    )

                # Perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Compute the previous noisy sample
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # Decode latents
            latents = 1 / 0.18215 * latents
            with torch.no_grad():
                images = self.vae.decode(latents).sample

            results.extend(images.cpu())

            # Free memory after each batch
            del text_embeddings, latents, noise_pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        return torch.stack(results)

    def encode_image(self, image, batch_size=1):
        """Encode the input image to latent space"""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device).to(self.vae_dtype)
        latents = vae_encode(
            self.config.vae.vae_type,
            self.vae,
            image,
            True,
            self.device
        )
        return latents.to(self.weight_dtype)

    def encode_prompt(self, prompt, negative_prompt="", batch_size=1):
        """Encode the prompt to text embeddings"""
        # Ensure text encoder is loaded
        self.ensure_text_encoder_loaded()
        
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        # Handle Chinese prompts if configured
        if self.text_encoder_config.chi_prompt:
            chi_prompt = "\n".join(self.text_encoder_config.chi_prompt)
            prompts = [chi_prompt + p for p in prompt]
            num_chi_prompt_tokens = len(self.tokenizer.encode(chi_prompt))
            max_length = num_chi_prompt_tokens + self.text_encoder_config.model_max_length - 2
        else:
            prompts = prompt
            max_length = self.text_encoder_config.model_max_length

        # Tokenize prompts
        text_tokens = self.tokenizer(
            prompts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Select proper token indices
        select_index = [0] + list(range(-self.text_encoder_config.model_max_length + 1, 0))
        
        # Get text embeddings
        text_embeddings = self.text_encoder(
            text_tokens.input_ids,
            text_tokens.attention_mask
        )[0][:, None][:, :, select_index].to(self.weight_dtype)
        
        # Get embedding masks
        embedding_masks = text_tokens.attention_mask[:, select_index]

        # Handle negative prompts
        if negative_prompt:
            uncond_tokens = self.tokenizer(
                negative_prompt,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            uncond_embeddings = self.text_encoder(
                uncond_tokens.input_ids,
                uncond_tokens.attention_mask
            )[0][:, None][:, :, select_index].to(self.weight_dtype)
        else:
            uncond_embeddings = self.null_caption_embs.repeat(len(prompts), 1, 1)[:, None].to(self.weight_dtype)

        # Duplicate for batch size
        text_embeddings = text_embeddings.repeat(batch_size, 1, 1, 1)
        uncond_embeddings = uncond_embeddings.repeat(batch_size, 1, 1, 1)

        # Concatenate for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Clean up
        del text_tokens
        if negative_prompt:
            del uncond_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        return text_embeddings

