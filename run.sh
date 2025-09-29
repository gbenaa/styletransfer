#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTORCH_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"

python run_flux_kontext.py \
  --offload --no_xformers \
  --gpu_abs 2 --gpu 0 \
  --steps 12 --guidance 1.2 --seed 19 \
  --prompt "oil painting, impasto, luminous brushwork, late Turner atmosphere, high-key lighting, painterly texture, preserve composition and subject" \
  --neg "do not change composition; do not change subject; no new objects; no hallucinated scenery; underexposed; dim; low key; crushed blacks" \
  --image in/in01.jpeg \
  --width 1152 --height 768 --keep-aspect --letterbox \
  --out_dir out \
  --name_template "{stem}_final.png" \
  --manifest out/manifest.csv \
  --prompt_index 1 \
  --save_all_steps \
  --callback_interval 1 \
  --step_name_template "{stem}_step{step:03d}.png" \
  --perstep_manifest out/manifest_steps.csv
