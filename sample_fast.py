import warnings
warnings.filterwarnings("ignore")
import os, sys, pdb
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import re
import copy
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from diffusers import DiffusionPipeline, DDIMScheduler
from safetensors.torch import load_file as load_safetensors
from src.template import template_dict
from src.utils import *


def diffusion_batched(unet, scheduler, latents, text_embeddings, total_timesteps,
                      start_timesteps=0, guidance_scale=7.5, desc=None, **kwargs):
    """
    Args:
        latents: [B, 4, 64, 64]
        text_embeddings: [2*B, 77, 768] (already concatenated uncond + cond)
    """
    scheduler.set_timesteps(total_timesteps)
    for timestep in tqdm(scheduler.timesteps[start_timesteps: total_timesteps], desc=desc, leave=False):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

        noise_pred = unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=text_embeddings,
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    return latents


def split_contents_arg(raw_text):
    return [item.strip() for item in raw_text.split(';') if item.strip()]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    # Base Config
    parser.add_argument('--save_root', type=str, default='')
    parser.add_argument('--sd_ckpt', help='base version for stable diffusion', type=str, default='/data/coding/model_weight/SD')
    parser.add_argument('--seed', type=int, default=0)
    # Sampling Config
    parser.add_argument('--mode', type=str, default='original', help='original, edit')
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--total_timesteps', type=int, default=50, help='The total timesteps of the sampling process')
    parser.add_argument('--num_samples', type=int, default=10, help='The number of samples per prompt to generate')
    parser.add_argument('--batch_size', type=int, default=10, help='The batch size of the sampling process')
    parser.add_argument('--max_prompt_batch', type=int, default=10, help='Max number of prompts to process in one diffusion call')
    parser.add_argument('--prompts', type=str, default=None)
    # Erasing Config
    parser.add_argument('--erase_type', type=str, default='', help='instance, style, celebrity')
    parser.add_argument('--target_concept', type=str, default='')
    parser.add_argument('--contents', type=str, default='')
    parser.add_argument('--edit_ckpt', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    args = parser.parse_args()
    assert args.num_samples >= args.batch_size

    bs = args.batch_size
    max_prompt_batch = args.max_prompt_batch
    mode_list = args.mode.replace(' ', '').split(',')
    device = f'cuda:{args.gpu}'

    concept_list, concept_list_tmp = [], split_contents_arg(args.contents)
    if 'edit' in mode_list:
        for concept in concept_list_tmp:
            check_path = os.path.join(args.save_root, args.target_concept.replace(', ', '_'), concept, 'edit')
            os.makedirs(check_path, exist_ok=True)
            expected = len(template_dict[args.erase_type]) * args.num_samples
            if len([f for f in os.listdir(check_path) if f.endswith('.png')]) != expected:
                concept_list.append(concept)
    else:
        concept_list = concept_list_tmp
    if len(concept_list) == 0:
        print("[INFO] All concepts already sampled, exiting.")
        sys.exit()

    # region [Prepare Models]
    print(f"[INFO] Loading model on GPU {args.gpu}...")
    pipe = DiffusionPipeline.from_pretrained(args.sd_ckpt, safety_checker=None, torch_dtype=torch.float16).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    unet, tokenizer, text_encoder, vae = pipe.unet, pipe.tokenizer, pipe.text_encoder, pipe.vae
    if 'edit' in mode_list:
        unet_edit = copy.deepcopy(unet)
        edit_path = args.edit_ckpt or os.path.join("logs/checkpoints", sorted(os.listdir("logs/checkpoints"))[-1])
        if os.path.isdir(edit_path):
            unet_safetensors = os.path.join(edit_path, "unet", "diffusion_pytorch_model.safetensors")
            unet_bin = os.path.join(edit_path, "unet", "diffusion_pytorch_model.bin")
            if os.path.exists(unet_safetensors):
                state_dict = load_safetensors(unet_safetensors, device="cpu")
            elif os.path.exists(unet_bin):
                state_dict = torch.load(unet_bin, map_location='cpu')
            else:
                raise FileNotFoundError(f"No UNet weights found in {edit_path}/unet/")
        elif edit_path.endswith(".safetensors"):
            state_dict = load_safetensors(edit_path, device="cpu")
        else:
            state_dict = torch.load(edit_path, map_location='cpu')
        unet_edit.load_state_dict(state_dict, strict=False)
    # endregion

    uncond_embedding = get_textencoding(get_token('', tokenizer), text_encoder)

    # Sampling process
    seed_everything(args.seed, True)
    if args.prompts is None:
        prompt_list = [[x.format(concept) for x in template_dict[args.erase_type]] for concept in concept_list]
    else:
        prompt_list = [[x.format(concept) for x in args.prompts.split(';')] for concept in concept_list]

    num_batches = int(args.num_samples // bs)

    for i in range(num_batches):
        # Generate base latents once; shared across all concepts and prompts (same as original sample.py)
        base_latents = torch.randn(bs, 4, 64, 64).to(device, dtype=pipe.dtype)

        for concept, prompts in zip(concept_list, prompt_list):
            # Precompute text embeddings for all prompts
            embeddings = []
            for prompt in prompts:
                emb = get_textencoding(get_token(prompt, tokenizer), text_encoder)
                embeddings.append(emb)

            save_path = os.path.join(args.save_root, args.target_concept.replace(', ', '_'), concept)
            for mode in mode_list:
                os.makedirs(os.path.join(save_path, mode), exist_ok=True)
            if len(mode_list) > 1:
                os.makedirs(os.path.join(save_path, 'combine'), exist_ok=True)

            # Process prompts in batches
            for p_start in range(0, len(prompts), max_prompt_batch):
                p_end = min(p_start + max_prompt_batch, len(prompts))
                batch_prompts = prompts[p_start:p_end]
                batch_embeddings = embeddings[p_start:p_end]
                n_prompts = len(batch_prompts)

                # Build batched latents and text embeddings
                # latents: repeat base_latents for each prompt in batch -> [n_prompts * bs, 4, 64, 64]
                latents_batch = base_latents.repeat(n_prompts, 1, 1, 1)

                # text_embeddings: [2 * n_prompts * bs, 77, 768]
                uncond_repeated = uncond_embedding.repeat(n_prompts * bs, 1, 1)
                cond_repeated = torch.cat([emb.repeat(bs, 1, 1) for emb in batch_embeddings], dim=0)
                text_embeddings_batch = torch.cat([uncond_repeated, cond_repeated], dim=0)

                desc = f"Concept={concept} | prompts[{p_start}:{p_end}] | batch={i}/{num_batches}"

                for mode in mode_list:
                    if mode == 'original':
                        result_latents = diffusion_batched(
                            unet=unet, scheduler=pipe.scheduler,
                            latents=latents_batch.clone(),
                            text_embeddings=text_embeddings_batch,
                            total_timesteps=args.total_timesteps,
                            guidance_scale=args.guidance_scale,
                            desc=desc + " | original"
                        )
                    else:  # edit
                        result_latents = diffusion_batched(
                            unet=unet_edit, scheduler=pipe.scheduler,
                            latents=latents_batch.clone(),
                            text_embeddings=text_embeddings_batch,
                            total_timesteps=args.total_timesteps,
                            guidance_scale=args.guidance_scale,
                            desc=desc + " | edit"
                        )

                    # Decode and save images
                    # result_latents: [n_prompts * bs, 4, 64, 64]
                    # Split back per prompt
                    for p_idx, prompt in enumerate(batch_prompts):
                        start_idx = p_idx * bs
                        end_idx = start_idx + bs
                        prompt_latents = result_latents[start_idx:end_idx]

                        decoded_imgs = []
                        for img_latent in prompt_latents:
                            decoded = vae.decode(img_latent.unsqueeze(0) / vae.config.scaling_factor, return_dict=False)[0]
                            decoded_imgs.append(process_img(decoded))

                        for idx_in_batch, img in enumerate(decoded_imgs):
                            global_idx = int(idx_in_batch + bs * i)
                            save_filename = re.sub(r'[^\w\s]', '', prompt).replace(', ', '_') + f"_{global_idx}.png"
                            img.save(os.path.join(save_path, mode, save_filename))

    print(f"[DONE] GPU {args.gpu} finished sampling {len(concept_list)} concepts.")


if __name__ == '__main__':
    main()
