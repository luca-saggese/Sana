#!/usr/bin/env python
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
from __future__ import annotations

import argparse
import os
import random
import socket
import sqlite3
import time
import uuid
from datetime import datetime

import gradio as gr
import numpy as np
import spaces
import torch
from PIL import Image
from torchvision.utils import make_grid, save_image
from transformers import AutoModelForCausalLM, AutoTokenizer
import tempfile, uuid, os
from pathlib import Path

from app import safety_check
from app.sana_pipeline import SanaPipeline

import json



MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "1") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "4096"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"
DEMO_PORT = int(os.getenv("DEMO_PORT", "15432"))
os.environ["GRADIO_EXAMPLES_CACHE"] = "./.gradio/cache"
COUNTER_DB = os.getenv("COUNTER_DB", ".count.db")
ROOT_PATH = os.getenv("ROOT_PATH", None)
HISTORY_FILE = "/app/output/generation_history.json"
HISTORY_LIMIT = 5000  # massimo numero di immagini da mantenere

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

generation_history = load_history()

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
         # Funzione per aggiornare la galleria
         
def get_history_gallery():
    return [(item["img_path"], f"{item['prompt']}") for item in generation_history[-HISTORY_LIMIT:]]

# Quando cambia l’indice selezionato, ripopola i parametri
def repopulate_fields(index: int):
    if 0 <= index < len(generation_history):
        item = generation_history[index]
        return (
            item["prompt"],
            item["negative_prompt"],
            item["style"],
            item["seed"],
            item["height"],
            item["width"],
        )
    return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

def delete_history_item(index: int):
    if 0 <= index < len(generation_history):
        img_path = generation_history[index]["img_path"]
        # Rimuovi il file immagine
        if os.path.exists(img_path):
            os.remove(img_path)
        # Rimuovi dalla lista
        del generation_history[index]
        save_history(generation_history)
    return get_history_gallery(), -1  # aggiorna galleria e resetta selezione





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, "
        "cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, "
        "majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, "
        "glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, "
        "disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, "
        "detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, "
        "ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
     {
        "name": "URSS Brutalism",
        "prompt": "futuristic brutalist soviet architecture, {prompt}, dystopian, cold atmosphere, monumental structure, "
        "concrete, retrofuturistic design, misty, overcast, high detail, cinematic lighting, ultra wide shot, realistic textures",
        "negative_prompt": "people, humans, text, logos, deformed, cartoon, painting, sketch, drawing, "
        "oversaturated, low quality, blurry, noisy, watermark, extra limbs, ugly, distortedg",
    },

     {
        "name": "Robert Doisenau",
        "prompt": "realistic romantic black and white street photo of {prompt} in 1950s Paris, candid moment, vintage "
        "clothing,  Parisian street background, atmospheric, Leica camera style, film grain, soft focus, poetic mood, timeless elegance",
        "negative_prompt": "color, modern clothing, logos, deformed, cartoon, anime, painting, unrealistic, glitch, "
        "digital artifacts, distortion, watermark",
    },
    {
        "name":"Mario Giacomelli",
        "prompt":"high contrast black and white photo of {prompt} abstract lighting, strong silhouette, surreal empty space, other figures distant expressionist style, metaphysical mood, inspired by Mario Giacomelli",
        "negative_prompt":"photorealism, colorful, digital painting, soft shadows, anime, cartoon, modern clothes, glitch, distorted, watermark "
    }
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"
SCHEDULE_NAME = ["Flow_DPM_Solver"]
DEFAULT_SCHEDULE_NAME = "Flow_DPM_Solver"
INFER_SPEED = 0


def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))
    return img


def open_db():
    db = sqlite3.connect(COUNTER_DB)
    db.execute("CREATE TABLE IF NOT EXISTS counter(app CHARS PRIMARY KEY UNIQUE, value INTEGER)")
    db.execute('INSERT OR IGNORE INTO counter(app, value) VALUES("Sana", 0)')
    return db


def read_inference_count():
    with open_db() as db:
        cur = db.execute('SELECT value FROM counter WHERE app="Sana"')
    return cur.fetchone()[0]


def write_inference_count(count):
    count = max(0, int(count))
    with open_db() as db:
        db.execute(f'UPDATE counter SET value=value+{count} WHERE app="Sana"')
        db.commit()


def run_inference(num_imgs=1):
    write_inference_count(num_imgs)
    count = read_inference_count()

    return (
        f"<span style='font-size: 16px; font-weight: bold;'>Total inference runs: </span><span style='font-size: "
        f"16px; color:red; font-weight: bold;'>{count}</span>"
    )


def update_inference_count():
    count = read_inference_count()
    return (
        f"<span style='font-size: 16px; font-weight: bold;'>Total inference runs: </span><span style='font-size: "
        f"16px; color:red; font-weight: bold;'>{count}</span>"
    )


def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config")
    parser.add_argument(
        "--model_path",
        nargs="?",
        default="hf://Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth",
        type=str,
        help="Path to the model file (positional)",
    )
    parser.add_argument("--output", default="./", type=str)
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--cfg_scale", default=5.0, type=float)
    parser.add_argument("--pag_scale", default=2.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--step", default=-1, type=int)
    parser.add_argument("--custom_image_size", default=None, type=int)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--shield_model_path",
        type=str,
        help="The path to shield model, we employ ShieldGemma-2B by default.",
        default="google/shieldgemma-2b",
    )

    return parser.parse_known_args()[0]


args = get_args()

if torch.cuda.is_available():
    model_path = args.model_path
    print(f"Loading model from {model_path}")
    pipe = SanaPipeline(args.config)
    print(f"Loading model from {model_path}")
    pipe.from_pretrained(model_path)
    print(f"Model loaded from {model_path}")
    pipe.register_progress_bar(gr.Progress())
    
    




def save_image_sana(img, seed="", save_img=False):
    unique_name = f"{str(uuid.uuid4())}_{seed}.png"
    save_path = os.path.join(f"output/online_demo_img/{datetime.now().date()}")
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(save_path, exist_ok=True)
    unique_name = os.path.join(save_path, unique_name)
    if save_img:
        save_image(img, unique_name, nrow=1, normalize=True, value_range=(-1, 1))

    return unique_name


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def deselect():
    return gr.Gallery(selected_index=None)


def select_first():
    return gr.Gallery(selected_index=0)


@torch.no_grad()
@torch.inference_mode()
@spaces.GPU(enable_queue=True)
def generate(
    prompt: str = None,
    negative_prompt: str = "",
    style: str = DEFAULT_STYLE_NAME,
    use_negative_prompt: bool = False,
    num_imgs: int = 1,
    seed: int = 0,
    height: int = 1024,
    width: int = 1024,
    flow_dpms_guidance_scale: float = 5.0,
    flow_dpms_pag_guidance_scale: float = 2.0,
    flow_dpms_inference_steps: int = 20,
    randomize_seed: bool = False,
    reference_image: Image.Image = None,  # 👈 Aggiunto
    image_guidance_scale: float = 1.0,    # 👈 Aggiunto
    inpaint_mask: Image.Image = None,
):
    write_inference_count(num_imgs)
    global INFER_SPEED
    # seed = 823753551
    seed = int(randomize_seed_fn(seed, randomize_seed))
    reference_tensor = None
    if reference_image is not None:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            lambda x: x.to(dtype=torch.float16)
        ])
        reference_tensor = transform(reference_image)[None].to(device)
    mask_tensor = None
    if inpaint_mask is not None:
        transform_mask = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),  # grayscale → (1,H,W)
        ])
        mask_tensor = transform_mask(inpaint_mask).unsqueeze(0).to(device)
        # Binarizza la maschera: 1 = da rigenerare
        mask_tensor = (mask_tensor > 0.5).float()
    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"PORT: {DEMO_PORT}, model_path: {model_path}")


    print(prompt)

    num_inference_steps = flow_dpms_inference_steps
    guidance_scale = flow_dpms_guidance_scale
    pag_guidance_scale = flow_dpms_pag_guidance_scale

    if not use_negative_prompt:
        negative_prompt = None  # type: ignore
    prompt, negative_prompt = apply_style(style, prompt, negative_prompt)

    pipe.progress_fn(0, desc="Sana Start")

    time_start = time.time()
    images = pipe(
        prompt=prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        pag_guidance_scale=pag_guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_imgs,
        generator=generator,
        reference_image=reference_tensor,               # 👈 aggiunto
        image_guidance_scale=image_guidance_scale,     # 👈 aggiunto
        inpaint_mask=mask_tensor,                      # 👈 aggiunto
    )

    pipe.progress_fn(1.0, desc="Sana End")
    INFER_SPEED = (time.time() - time_start) / num_imgs

    # --- NUOVO BLOCCO: salva su disco ---
    saved_paths = []
    tmpdir = Path("/app/output")
    tmpdir.mkdir(parents=True, exist_ok=True)

    for idx, img_t in enumerate(images):
        pil_img = Image.fromarray(
            norm_ip(img_t, -1, 1)
            .mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
            .astype(np.uint8)
        )

        # Nome: prompt “safe” + seed + indice
        filename = f"{uuid.uuid4().hex[:8]}_{seed}_{idx}.png"
        filepath = tmpdir / filename
        pil_img.save(filepath, format="PNG")

        # Salva tuple (path, caption) per la Gallery
        saved_paths.append((str(filepath), filename))

    torch.cuda.empty_cache()

    # Salva ogni immagine nella cronologia
    for i, (img_path, caption) in enumerate(saved_paths):
        generation_history.append({
            "img_path": img_path,
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "style": style,
            "seed": seed,
            "height": height,
            "width": width,
        })

    # Limita la dimensione della cronologia
    generation_history[:] = generation_history[-HISTORY_LIMIT:]
    save_history(generation_history)

    return (
        saved_paths,
        seed,
    )


model_size = "1.6" if "1600M" in args.model_path else "0.6"
title = f"""
    <div style='display: flex; align-items: center; justify-content: center; text-align: center;'>
        <img src="https://raw.githubusercontent.com/luca-saggese/Sana/refs/heads/main/asset/logo_goart.png" width="50%" alt="logo"/>
    </div>
"""
DESCRIPTION = f"""
        <p><span style="font-size: 36px; font-weight: bold;">Sana-{model_size}B</span><span style="font-size: 20px; font-weight: bold;">{args.image_size}px</span> </p>
        <p style="font-size: 16px; font-weight: bold;"><a href="https://nvlabs.github.io/Sana">Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer</a></p>
        <p style="font-size: 16px; font-weight: bold;">Powered by <a href="https://hanlab.mit.edu/projects/dc-ae">DC-AE</a>, <a href="https://github.com/mit-han-lab/deepcompressor">deepcompressor</a>, and <a href="https://github.com/mit-han-lab/nunchaku">nunchaku</a>.</p>
        <p style="font-size: 16px; font-weight: bold;">Prompts support English, Chinese and emojis.</p>
        """
if model_size == "0.6":
    DESCRIPTION += "\n<p>0.6B model's text rendering ability is limited.</p>"
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

examples = [
    'a cyberpunk cat with a neon sign that says "Sana"',
    "A very detailed and realistic full body photo set of a tall, slim, and athletic Shiba Inu in a white oversized straight t-shirt, white shorts, and short white shoes.",
    "Pirate ship trapped in a cosmic maelstrom nebula, rendered in cosmic beach whirlpool engine, volumetric lighting, spectacular, ambient lights, light pollution, cinematic atmosphere, art nouveau style, illustration art artwork by SenseiJaye, intricate detail.",
    "portrait photo of a girl, photograph, highly detailed face, depth of field",
    'make me a logo that says "So Fast"  with a really cool flying dragon shape with lightning sparks all over the sides and all of it contains Indonesian language',
    "🐶 Wearing 🕶 flying on the 🌈",
    "👧 with 🌹 in the ❄️",
    "an old rusted robot wearing pants and a jacket riding skis in a supermarket.",
    "professional portrait photo of an anthropomorphic cat wearing fancy gentleman hat and jacket walking in autumn forest.",
    "Astronaut in a jungle, cold color palette, muted colors, detailed",
    "a stunning and luxurious bedroom carved into a rocky mountainside seamlessly blending nature with modern design with a plush earth-toned bed textured stone walls circular fireplace massive uniquely shaped window framing snow-capped mountains dense forests",
]

css = """
.gradio-container{max-width: 660px !important}
body{align-items: center;}
h1{text-align:center}
"""
with gr.Blocks(css=css, title="Sana", delete_cache=(86400, 86400)) as demo:
    gr.Markdown(title)
    gr.HTML(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    info_box = gr.Markdown(value=update_inference_count, every=10)
    # demo.load(fn=update_inference_count, outputs=info_box, api_name=False)  # update the value when re-loading the page
    # with gr.Row(equal_height=False):
    with gr.Group():
        prompt = gr.Textbox(
            label="Prompt",
            show_label=False,
            placeholder="Enter your prompt",
            container=False,
            submit_btn="Run",
        )
        result = gr.Gallery(label="Result", show_label=False, format="webp", height=600)
        history_gallery = gr.Gallery(label="History", show_label=True, height=300)
        selected_index = gr.Number(visible=False)
        rerun_button = gr.Button("Rilancia selezione", variant="primary")
        delete_button = gr.Button("Delete selected image", variant="stop")
    with gr.Accordion("Advanced options", open=False):
        with gr.Group():
            with gr.Row(visible=True):
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=args.image_size,
                )
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=args.image_size,
                )
            with gr.Row():
                flow_dpms_inference_steps = gr.Slider(
                    label="Sampling steps",
                    minimum=5,
                    maximum=40,
                    step=1,
                    value=20,
                )
                flow_dpms_guidance_scale = gr.Slider(
                    label="CFG Guidance scale",
                    minimum=1,
                    maximum=10,
                    step=0.1,
                    value=4.5,
                )
                flow_dpms_pag_guidance_scale = gr.Slider(
                    label="PAG Guidance scale",
                    minimum=1,
                    maximum=4,
                    step=0.5,
                    value=1.0,
                )
            with gr.Row():
                use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=False, visible=True)
            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=1,
                placeholder="Enter a negative prompt",
                visible=True,
            )
            style_selection = gr.Radio(
                show_label=True,
                container=True,
                interactive=True,
                choices=STYLE_NAMES,
                value=DEFAULT_STYLE_NAME,
                label="Image Style",
            )
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            with gr.Row():
                reference_image = gr.Image(
                    label="Reference image (optional)",
                    type="pil",
                    #tool="editor",
                    image_mode="RGB",
                    sources=["upload"],
                )
                inpaint_mask = gr.Image(
                    label="Inpaint mask (draw in white)",
                    type="pil",
                    #tool="sketch",  # modalità disegno
                    image_mode="L",
                    #sources=["upload", "canvas"],
                    sources=["upload"],
                )
                image_guidance_scale = gr.Slider(
                    label="Image guidance strength",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                )
            with gr.Row(visible=True):
                schedule = gr.Radio(
                    show_label=True,
                    container=True,
                    interactive=True,
                    choices=SCHEDULE_NAME,
                    value=DEFAULT_SCHEDULE_NAME,
                    label="Sampler Schedule",
                    visible=True,
                )
                num_imgs = gr.Slider(
                    label="Num Images",
                    minimum=1,
                    maximum=2,
                    step=1,
                    value=1,
                )

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=[result, seed],
        run_on_click=CACHE_EXAMPLES,
        cache_mode="lazy",
        examples_per_page=len(examples),
        fn=generate if CACHE_EXAMPLES else None,
        cache_examples=CACHE_EXAMPLES,
    )

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        api_name=False,
    )

    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
        ],
        fn=deselect,
        inputs=None,
        outputs=result,
        show_progress="hidden",
        api_name=False,
        queue=False,
    ).then(
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            style_selection,
            use_negative_prompt,
            num_imgs,
            seed,
            height,
            width,
            flow_dpms_guidance_scale,
            flow_dpms_pag_guidance_scale,
            flow_dpms_inference_steps,
            randomize_seed,
            reference_image,          # 👈 aggiunto
            image_guidance_scale,     # 👈 aggiunto
            inpaint_mask,           # 👈 aggiunto
        ],
        outputs=[result, seed],
        api_name="run",
    ).then(
        fn=select_first,
        inputs=None,
        outputs=result,
        show_progress="full",
        api_name=False,
        queue=False,
    ).then(
        fn=get_history_gallery,
        inputs=None,
        outputs=history_gallery,
        show_progress="hidden"
    )
    demo.load(fn=get_history_gallery, inputs=None, outputs=history_gallery)
    gr.HTML(
        value="<p style='text-align: center; font-size: 14px;'>Useful link: <a href='https://accessibility.mit.edu'>MIT Accessibility</a></p>"
    )

    # Quando clicchi un'immagine, restituisce l'indice
    history_gallery.select(
        fn=lambda i: i,
        inputs=None,
        outputs=selected_index
    )
    
    rerun_button.click(
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            style_selection,
            use_negative_prompt,
            num_imgs,
            seed,
            height,
            width,
            flow_dpms_guidance_scale,
            flow_dpms_pag_guidance_scale,
            flow_dpms_inference_steps,
            randomize_seed,
            reference_image,
            image_guidance_scale,
            inpaint_mask,
        ],
        outputs=[result, seed],
    ).then(
        fn=get_history_gallery,
        inputs=None,
        outputs=history_gallery
    )

    selected_index.change(
        fn=repopulate_fields,
        inputs=selected_index,
        outputs=[prompt, negative_prompt, style_selection, seed, height, width]
    )
    
    delete_button.click(
        fn=delete_history_item,
        inputs=selected_index,
        outputs=[history_gallery, selected_index],
    )

if __name__ == "__main__":
    
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0", server_port=DEMO_PORT, debug=False, share=args.share, root_path=ROOT_PATH
    )
