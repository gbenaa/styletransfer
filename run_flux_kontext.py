#!/usr/bin/env python3
# FLUX.1-Kontext single-GPU runner (rectangular outputs supported)
# Additions:
#  - Per-step saving via callbacks:
#      --save_all_steps
#      --callback_interval
#      --step_name_template
#      --perstep_manifest
#  - (Existing) centralised filename templating + manifest logging

import os
import argparse
from pathlib import Path


# =========================
# = Argument parsing
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="FLUX.1-Kontext single-GPU image edit")

    # GPU visibility/selection
    p.add_argument("--devices", type=str, default=None)
    p.add_argument("--gpu_abs", type=int, default=None)
    p.add_argument("--gpu", type=int, default=0)

    # Model & prompts
    p.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--neg", "--negative", dest="negative_prompt", type=str, default=None)

    # I/O (legacy direct output and new centralised naming)
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--out", type=str, default="output.png")  # -> if provided, wins over template
    p.add_argument("--out_dir", type=str, default="out")
    p.add_argument(
        "--name_template",
        type=str,
        default="{stem}_s{steps}_g{guidance}_seed{seed}_p{index:02d}_{slug}.png",
        help="Used only if --out not provided."
    )
    p.add_argument("--prompt_index", type=int, default=1, help="1-based index used in templated filenames.")
    p.add_argument("--manifest", type=str, default="out/manifest.csv")

    # Inference
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=None)

    # Memory/perf
    p.add_argument("--offload", action="store_true")
    p.add_argument("--no_xformers", action="store_true")

    # Sizing & aspect
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--keep-aspect", action="store_true")
    p.add_argument("--letterbox", action="store_true")

    # Optional post-processing
    p.add_argument("--post_brightness", type=float, default=1.0)
    p.add_argument("--post_contrast", type=float, default=1.0)
    p.add_argument("--post_gamma", type=float, default=1.0)

    # ---- Per-step saving controls ----
    p.add_argument("--save_all_steps", action="store_true",
                   help="Save an image for every diffusion step (or each --callback_interval step).")
    p.add_argument("--callback_interval", type=int, default=1,
                   help="Save every Nth step when --save_all_steps is used.")
    p.add_argument("--step_name_template", type=str,
                   default="{stem}_s{steps}_g{guidance}_seed{seed}_p{index:02d}_{slug}_step{step:03d}.png",
                   help="Filename template for per-step frames.")
    p.add_argument("--perstep_manifest", type=str, default=None,
                   help="Optional CSV manifest for per-step frames.")

    return p.parse_args()


# =========================
# = Utils
# =========================
def apply_visibility_env(args):
    # -> Control CUDA device visibility from CLI
    if args.gpu_abs is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_abs)
        print(f"-> Set CUDA_VISIBLE_DEVICES via --gpu_abs: {os.environ['CUDA_VISIBLE_DEVICES']}")
    elif args.devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
        print(f"-> Set CUDA_VISIBLE_DEVICES via --devices: {os.environ['CUDA_VISIBLE_DEVICES']}")


def resize_with_aspect(img, width, height, keep_aspect, letterbox):
    from PIL import Image
    if width is None or height is None:
        return img
    if not keep_aspect:
        return img.resize((width, height), Image.LANCZOS)

    sw, sh = img.size
    scale = min(width / sw, height / sh)
    nw, nh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
    rs = img.resize((nw, nh), Image.LANCZOS)

    if letterbox:
        canvas = Image.new("RGB", (width, height))
        canvas.paste(rs, ((width - nw) // 2, (height - nh) // 2))
        return canvas

    # -> Center-crop to exact target size
    if nw < width or nh < height:
        bg = Image.new("RGB", (max(width, nw), max(height, nh)))
        bg.paste(rs, ((bg.width - nw) // 2, (bg.height - nh) // 2))
        rs = bg
    lx, ty = (rs.width - width) // 2, (rs.height - height) // 2
    return rs.crop((lx, ty, lx + width, ty + height))


def maybe_enable_xformers(pipe, disabled: bool):
    if disabled:
        return
    try:
        import xformers  # noqa: F401
        pipe.enable_xformers_memory_efficient_attention()
        print("-> xFormers enabled")
    except Exception as e:
        print(f"-> xFormers unavailable or failed to enable: {e}")


def postprocess_pil(pil, b=1.0, c=1.0, g=1.0):
    from PIL import ImageEnhance
    if b != 1.0:
        pil = ImageEnhance.Brightness(pil).enhance(b)
    if c != 1.0:
        pil = ImageEnhance.Contrast(pil).enhance(c)
    if g != 1.0 and g > 0:
        inv = 1.0 / g
        lut = [int(((i / 255.0) ** inv) * 255.0 + 0.5) for i in range(256)]
        pil = pil.point(lut * 3)
    return pil


def slugify(text: str, max_len: int = 60) -> str:
    # -> Lowercase, keep a–z 0–9 and hyphens, compress spaces to single hyphen
    import re
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug.strip())
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:max_len] or "prompt"


def ensure_manifest_header(path: Path, per_step: bool = False):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            if per_step:
                f.write("index,step,timestep,filename,prompt,seed,steps,guidance,width,height,model,neg,image\n")
            else:
                f.write("index,filename,prompt,seed,steps,guidance,width,height,model,neg,image\n")


def append_manifest(path: Path, row: dict, per_step: bool = False):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        ensure_manifest_header(path, per_step=per_step)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if per_step:
            w.writerow([
                row.get("index", ""),
                row.get("step", ""),
                row.get("timestep", ""),
                row.get("filename", ""),
                row.get("prompt", ""),
                row.get("seed", ""),
                row.get("steps", ""),
                row.get("guidance", ""),
                row.get("width", ""),
                row.get("height", ""),
                row.get("model", ""),
                row.get("neg", ""),
                row.get("image", ""),
            ])
        else:
            w.writerow([
                row.get("index", ""),
                row.get("filename", ""),
                row.get("prompt", ""),
                row.get("seed", ""),
                row.get("steps", ""),
                row.get("guidance", ""),
                row.get("width", ""),
                row.get("height", ""),
                row.get("model", ""),
                row.get("neg", ""),
                row.get("image", ""),
            ])


# =========================
# = Main
# =========================
def main():
    args = parse_args()
    apply_visibility_env(args)

    import torch
    from diffusers import DiffusionPipeline
    from diffusers.utils import load_image as hf_load_image

    assert torch.cuda.is_available(), "CUDA not available"
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"-> Using cuda:{args.gpu} ({torch.cuda.get_device_name(args.gpu)})")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # -> Seed
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        print(f"-> Seed: {args.seed}")
    else:
        print("-> Seed: random")

    # -> Perf knobs
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    print("-> Loading pipeline (FP32 for Maxwell)…")
    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        dtype=torch.float32,
        trust_remote_code=True,
        token=hf_token,
        low_cpu_mem_usage=True,
    )

    # -> Memory helpers
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    maybe_enable_xformers(pipe, disabled=args.no_xformers)

    if args.offload:
        print("-> Enabling sequential CPU offload (slower, lower VRAM).")
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)

    # -> Load + (optionally) pre-resize input
    img = hf_load_image(args.image)
    orig_w, orig_h = img.size
    target_w, target_h = args.width, args.height

    if target_w and target_h:
        img = resize_with_aspect(img, target_w, target_h, args.keep_aspect, args.letterbox)
        print(f"-> Input resized: {orig_w}x{orig_h} -> {img.size[0]}x{img.size[1]}")
    else:
        print(f"-> Using original input size: {orig_w}x{orig_h}")
        target_w, target_h = img.size  # -> use whatever we feed to the model

    # ---------- Per-step callback wiring ----------
    # Build common naming parts
    stem = Path(args.image).stem
    slug = slugify(args.prompt)
    seed_str = str(args.seed) if args.seed is not None else "rand"

    # Helper: save a tensor latents -> PIL -> file (handles decoding)
    def _save_step_image(latents_tensor, step: int, timestep: int):
        # Decode latents to image using the pipeline VAE; fall back gracefully.
        try:
            with torch.no_grad():
                # Many pipelines expose latents in [-?], need scaling factor
                scaling = getattr(pipe.vae, "scaling_factor", 0.18215)
                lat = latents_tensor
                if lat.device.type != device.type:
                    lat = lat.to(device)
                lat = lat / scaling
                image = pipe.vae.decode(lat).sample  # [B,C,H,W], float32
                # Map to [0,1] and to PIL via image_processor
                image = (image / 2 + 0.5).clamp(0, 1)
                pil_list = pipe.image_processor.postprocess(image, output_type="pil")
                pil = pil_list[0]
        except Exception as e:
            print(f"-> Step decode failed (step={step}, timestep={timestep}): {e}")
            return None

        # Construct filename from template
        fname = args.step_name_template.format(
            stem=stem,
            steps=args.steps,
            guidance=args.guidance,
            seed=seed_str,
            index=args.prompt_index,
            slug=slug,
            w=target_w,
            h=target_h,
            model=Path(args.model).name,
            step=step,
            timestep=timestep,
        )
        out_path = Path(args.out_dir) / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic-ish save
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        try:
            pil.save(str(tmp))
            tmp.replace(out_path)
            if args.perstep_manifest:
                append_manifest(
                    Path(args.perstep_manifest),
                    {
                        "index": args.prompt_index,
                        "step": step,
                        "timestep": timestep,
                        "filename": str(out_path),
                        "prompt": args.prompt,
                        "seed": args.seed if args.seed is not None else "",
                        "steps": args.steps,
                        "guidance": args.guidance,
                        "width": target_w,
                        "height": target_h,
                        "model": args.model,
                        "neg": args.negative_prompt or "",
                        "image": args.image,
                    },
                    per_step=True,
                )
        except Exception as e:
            print(f"-> Failed to save step image {out_path.name}: {e}")
            return None

        print(f"-> Saved step {step:03d} (t={timestep}) to: {out_path}")
        return out_path

    # Old-style callback(step, timestep, latents)
    def _legacy_cb(step: int, timestep: int, latents):
        if not args.save_all_steps:
            return
        if args.callback_interval > 1 and (step % args.callback_interval) != 0:
            return
        # latents expected shape [B, C, H, W]
        if latents is None:
            return
        _save_step_image(latents, step, timestep)

    # New-style callback_on_step_end with tensor inputs dict
    def _modern_cb(pipe, step: int, timestep: int, callback_kwargs):
        # callback_kwargs typically contains "latents"
        if not args.save_all_steps:
            return callback_kwargs
        if args.callback_interval > 1 and (step % args.callback_interval) != 0:
            return callback_kwargs
        latents = callback_kwargs.get("latents", None)
        if latents is not None:
            _save_step_image(latents, step, timestep)
        return callback_kwargs

    # Decide which callback API to use at runtime
    use_modern = hasattr(pipe, "set_progress_bar_config") and \
                 ("callback_on_step_end" in pipe.__call__.__code__.co_varnames or
                  "callback_on_step_end" in getattr(pipe.__call__, "__annotations__", {}))

    # -> Generate (height/width passed through if honoured by pipeline)
    print("-> Running edit …")
    call_kwargs = dict(
        image=img,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        generator=generator,
        height=target_h,
        width=target_w,
    )

    # Wire callbacks if requested
    if args.save_all_steps:
        if use_modern:
            # Newer diffusers: callback_on_step_end and select tensors to receive
            call_kwargs["callback_on_step_end"] = _modern_cb
            call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
            print("-> Per-step saving enabled (modern callback).")
        else:
            call_kwargs["callback"] = _legacy_cb
            call_kwargs["callback_steps"] = max(1, args.callback_interval)
            print("-> Per-step saving enabled (legacy callback).")

    with torch.inference_mode():
        result = pipe(**call_kwargs).images[0]

    # -> Safety net: force final size/aspect if model returned a different size
    if (result.size[0], result.size[1]) != (target_w, target_h):
        from PIL import Image
        print(
            f"-> Model returned {result.size}, coercing to {target_w}x{target_h} with "
            f"{'letterbox' if args.letterbox else ('keep-aspect crop' if args.keep_aspect else 'stretch')}"
        )
        if args.keep_aspect:
            result = resize_with_aspect(result, target_w, target_h, True, args.letterbox)
        else:
            result = result.resize((target_w, target_h), Image.LANCZOS)

    # -> Optional brightness/contrast/gamma
    if (args.post_brightness != 1.0) or (args.post_contrast != 1.0) or (args.post_gamma != 1.0):
        result = postprocess_pil(result, args.post_brightness, args.post_contrast, args.post_gamma)
        print(
            f"-> Post-processed (brightness={args.post_brightness}, "
            f"contrast={args.post_contrast}, gamma={args.post_gamma})"
        )

    # =========================
    # = Output path resolution (final frame)
    # =========================
    if args.out and args.out.strip() and args.out != "output.png":
        out_path = Path(args.out)
    else:
        fname = args.name_template.format(
            stem=stem,
            steps=args.steps,
            guidance=args.guidance,
            seed=seed_str,
            index=args.prompt_index,
            slug=slug,
            w=target_w,
            h=target_h,
            model=Path(args.model).name,
        )
        out_path = Path(args.out_dir) / fname

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(out_path))
    print(f"-> Saved: {out_path.resolve()}")

    # =========================
    # = Manifest append (final frame)
    # =========================
    append_manifest(
        Path(args.manifest),
        {
            "index": args.prompt_index,
            "filename": str(out_path),
            "prompt": args.prompt,
            "seed": args.seed if args.seed is not None else "",
            "steps": args.steps,
            "guidance": args.guidance,
            "width": target_w,
            "height": target_h,
            "model": args.model,
            "neg": args.negative_prompt or "",
            "image": args.image,
        },
        per_step=False,
    )


if __name__ == "__main__":
    main()
