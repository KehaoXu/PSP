SAMPLES_DIR="${1:-samples}"

## SCHEDULED

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/sdxl_outputs/geneval_scheduled_output_run02_eta=0/20260213-003936" \
#     --outfile "results/sdxl/geneval_scheduled_output_run02_eta=0/seed=42/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdxl/geneval_scheduled_output_run02_eta=0/seed=42/results_${SAMPLES_DIR}.jsonl"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/sdxl_outputs/geneval_scheduled_output_run02_eta=0/20260213-081133" \
#     --outfile "results/sdxl/geneval_scheduled_output_run02_eta=0/seed=43/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdxl/geneval_scheduled_output_run02_eta=0/seed=43/results_${SAMPLES_DIR}.jsonl"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/sdxl_outputs/geneval_scheduled_output_run02_eta=0/20260213-154214" \
#     --outfile "results/sdxl/geneval_scheduled_output_run02_eta=0/seed=44/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdxl/geneval_scheduled_output_run02_eta=0/seed=44/results_${SAMPLES_DIR}.jsonl"

## DYNAMIC

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/sdxl_outputs/geneval_dynamic_output_run02_eta=0/20260213-005840" \
#     --outfile "results/sdxl/geneval_dynamic_output_run02_eta=0/seed=42/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdxl/geneval_dynamic_output_run02_eta=0/seed=42/results_${SAMPLES_DIR}.jsonl"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/sdxl_outputs/geneval_dynamic_output_run02_eta=0/20260213-054441" \
#     --outfile "results/sdxl/geneval_dynamic_output_run02_eta=0/seed=43/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdxl/geneval_dynamic_output_run02_eta=0/seed=43/results_${SAMPLES_DIR}.jsonl"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/sdxl_outputs/geneval_dynamic_output_run02_eta=0/20260213-103138" \
#     --outfile "results/sdxl/geneval_dynamic_output_run02_eta=0/seed=44/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdxl/geneval_dynamic_output_run02_eta=0/seed=44/results_${SAMPLES_DIR}.jsonl"

## FK-STEERING

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/baseline_outputs/fk_sdxl_k4_t64_eta1_geneval/seed=42_20260213-084446" \
#     --outfile "results/sdxl/geneval_fk_output_run02_eta=0/seed=42/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdxl/geneval_fk_output_run02_eta=0/seed=42/results_${SAMPLES_DIR}.jsonl"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/baseline_outputs/fk_sdxl_k4_t64_eta1_geneval/seed=43_20260213-112732" \
#     --outfile "results/sdxl/geneval_fk_output_run02_eta=0/seed=43/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdxl/geneval_fk_output_run02_eta=0/seed=43/results_${SAMPLES_DIR}.jsonl"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/baseline_outputs/fk_sdxl_k4_t64_eta1_geneval/seed=44_20260213-141113" \
#     --outfile "results/sdxl/geneval_fk_output_run02_eta=0/seed=44/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdxl/geneval_fk_output_run02_eta=0/seed=44/results_${SAMPLES_DIR}.jsonl"

## Best-of-N

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/baseline_outputs/bestofn_sdxl_n4_t64_eta1_geneval/seed=42_20260213-084407" \
#     --outfile "results/sdxl/geneval_best_of_n_output_run02_eta=0/seed=42/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdxl/geneval_best_of_n_output_run02_eta=0/seed=42/results_${SAMPLES_DIR}.jsonl"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/baseline_outputs/bestofn_sdxl_n4_t64_eta1_geneval/seed=43_20260213-110741" \
#     --outfile "results/sdxl/geneval_best_of_n_output_run02_eta=0/seed=43/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdxl/geneval_best_of_n_output_run02_eta=0/seed=43/results_${SAMPLES_DIR}.jsonl"

python evaluation/evaluate_images.py \
    "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/baseline_outputs/bestofn_sdxl_n4_t64_eta1_geneval/seed=44_20260213-133007" \
    --outfile "results/sdxl/geneval_best_of_n_output_run02_eta=0/seed=44/results_${SAMPLES_DIR}.jsonl" \
    --samples-dir "${SAMPLES_DIR}" \
    --model-path "objdet"

python evaluation/summary_scores.py "results/sdxl/geneval_best_of_n_output_run02_eta=0/seed=44/results_${SAMPLES_DIR}.jsonl"