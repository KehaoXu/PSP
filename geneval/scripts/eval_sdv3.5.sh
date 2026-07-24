SAMPLES_DIR="${1:-samples}"

## SCHEDULED

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/sdv3.5_outputs/geneval_scheduled_output_run02_eta=0_sdv35/seed=42_20260213-083936" \
#     --outfile "results/sdv3.5/geneval_scheduled_output_run02_eta=0/seed=42/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdv3.5/geneval_scheduled_output_run02_eta=0/seed=42/results_${SAMPLES_DIR}.jsonl"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/sdv3.5_outputs/geneval_scheduled_output_run02_eta=0_sdv35/seed=43_20260213-154118" \
#     --outfile "results/sdv3.5/geneval_scheduled_output_run02_eta=0/seed=43/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdv3.5/geneval_scheduled_output_run02_eta=0/seed=43/results_${SAMPLES_DIR}.jsonl"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/sdv3.5_outputs/geneval_scheduled_output_run02_eta=0_sdv35/seed=44_20260213-224054" \
#     --outfile "results/sdv3.5/geneval_scheduled_output_run02_eta=0/seed=44/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdv3.5/geneval_scheduled_output_run02_eta=0/seed=44/results_${SAMPLES_DIR}.jsonl"

## DYNAMIC

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/sdv3.5_outputs/geneval_dynamic_output_run02_eta=0_sdv35/seed=42_20260213-083946" \
#     --outfile "results/sdv3.5/geneval_dynamic_output_run02_eta=0/seed=42/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdv3.5/geneval_dynamic_output_run02_eta=0/seed=42/results_${SAMPLES_DIR}.jsonl"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/sdv3.5_outputs/geneval_dynamic_output_run02_eta=0_sdv35/seed=43_20260213-132907" \
#     --outfile "results/sdv3.5/geneval_dynamic_output_run02_eta=0/seed=43/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdv3.5/geneval_dynamic_output_run02_eta=0/seed=43/results_${SAMPLES_DIR}.jsonl"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/sdv3.5_outputs/geneval_dynamic_output_run02_eta=0_sdv35/seed=44_20260213-181650" \
#     --outfile "results/sdv3.5/geneval_dynamic_output_run02_eta=0/seed=44/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdv3.5/geneval_dynamic_output_run02_eta=0/seed=44/results_${SAMPLES_DIR}.jsonl"

## FK-STEERING

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/baseline_outputs/fk_sdv35_k4_t32_eta1_geneval/seed=42_20260213-084551" \
#     --outfile "results/sdv3.5/geneval_fk_output_run02_eta=0/seed=42/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdv3.5/geneval_fk_output_run02_eta=0/seed=42/results_${SAMPLES_DIR}.jsonl"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/baseline_outputs/fk_sdv35_k4_t32_eta1_geneval/seed=43_20260213-175506" \
#     --outfile "results/sdv3.5/geneval_fk_output_run02_eta=0/seed=43/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdv3.5/geneval_fk_output_run02_eta=0/seed=43/results_${SAMPLES_DIR}.jsonl"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/baseline_outputs/fk_sdv35_k4_t32_eta1_geneval/seed=44_20260214-030341" \
#     --outfile "results/sdv3.5/geneval_fk_output_run02_eta=0/seed=44/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdv3.5/geneval_fk_output_run02_eta=0/seed=44/results_${SAMPLES_DIR}.jsonl"

## Best-of-N

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/baseline_outputs/bestofn_sdv35_n4_t32_eta1_geneval/seed=42_20260213-084520" \
#     --outfile "results/sdv3.5/geneval_best_of_n_output_run02_eta=0/seed=42/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdv3.5/geneval_best_of_n_output_run02_eta=0/seed=42/results_${SAMPLES_DIR}.jsonl"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/baseline_outputs/bestofn_sdv35_n4_t32_eta1_geneval/seed=43_20260213-133942" \
#     --outfile "results/sdv3.5/geneval_best_of_n_output_run02_eta=0/seed=43/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/sdv3.5/geneval_best_of_n_output_run02_eta=0/seed=43/results_${SAMPLES_DIR}.jsonl"

python evaluation/evaluate_images.py \
    "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/baseline_outputs/bestofn_sdv35_n4_t32_eta1_geneval/seed=44_20260213-183519" \
    --outfile "results/sdv3.5/geneval_best_of_n_output_run02_eta=0/seed=44/results_${SAMPLES_DIR}.jsonl" \
    --samples-dir "${SAMPLES_DIR}" \
    --model-path "objdet"

python evaluation/summary_scores.py "results/sdv3.5/geneval_best_of_n_output_run02_eta=0/seed=44/results_${SAMPLES_DIR}.jsonl"