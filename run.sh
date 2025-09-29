#!/usr/bin/env bash
set -euo pipefail

# ============ ENV hygiene (avoid OOM spikes) ============
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTORCH_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"

# ============ Input prompts ============
prompts=(
  "English countryside oil painting with thick impasto technique and warm earthy palette of ochres and sage greens, preserving exact Cotswolds village architecture, building positions and proportions, romantic atmospheric quality with visible brushstrokes following the forms"
  "Oil painting in the manner of Alfred Munnings with bold, confident brush marks and dynamic sky, maintaining all architectural details and village layout precisely as photographed, rich textural paint application on authentic composition"
  "Impressionistic oil painting with broken color technique and vibrant dappled light, keeping the exact arrangement and structure of all cottages, walls and landscape elements, loose gestural brushwork applied to accurate architectural framework"
  "Traditional oil painting with glazing and scumbling techniques, soft atmospheric perspective and muted tones, preserving every building's actual form, window placement and roofline, classical painting methods on faithful representation"
  "Oil painting with palette knife work and chunky paint texture, capturing morning mist and dewy atmosphere, maintaining precise village geography and all architectural features as they appear, expressive paint handling on accurate scene"
  "Tonalist oil painting with harmonious color relationships and subtle value transitions, keeping exact building locations, proportions and architectural character of the Cotswolds scene, poetic paint quality while respecting actual structures"
  "Oil painting in the Barbizon School tradition with naturalistic lighting and earthy color harmony, preserving all stone cottages, gardens and pathways exactly as positioned, rustic brushwork aesthetic on true-to-life composition"
  "Alla prima oil painting with wet-on-wet technique and spontaneous brushwork, maintaining faithful architectural details and accurate spatial relationships of the village, fresh direct painting approach without altering forms"
  "Oil painting with chiaroscuro effects and dramatic light-shadow contrasts on honey-colored stone, keeping every cottage, chimney and architectural element in its actual place, Rembrandt-influenced lighting on accurate Cotswolds scene"
  "Post-Impressionist oil painting with structured brushwork and rich saturated colors, preserving the exact village layout, building dimensions and architectural integrity, Cézanne-like paint construction while maintaining compositional truth"
)

# ============ Loop ============
i=1
for prompt in "${prompts[@]}"; do
  python run_flux_kontext_v05.py \
    --offload --no_xformers \
    --gpu_abs 2 --gpu 0 \
    --steps 2 \
    --guidance 1.2 \
    --seed 19 \
    --prompt "$prompt" \
    --neg "do not change composition; do not change subject; no new objects; no hallucinated scenery; underexposed; dim; low key; crushed blacks" \
    --image in/in01.jpeg \
    --width 1152 --height 768 \
    --keep-aspect --letterbox \
    --out_dir out \
    --name_template "{stem}_s{steps}_g{guidance}_seed{seed}_p{index:02d}_{slug}.png" \
    --manifest out/manifest.csv \
    --prompt_index "$i"

  i=$((i+1))
done
