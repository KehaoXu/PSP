SAMPLES_DIR="${1:-samples}"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_scheduled_output_run02_eta=0/20260212-182940" \
#     --outfile "results/multiturn_results/geneval_scheduled_output_run02_eta=0/seed=42/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/multiturn_results/geneval_scheduled_output_run02_eta=0/seed=42/results_${SAMPLES_DIR}.jsonl"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_scheduled_output_run02_eta=0/20260212-202157" \
#     --outfile "results/multiturn_results/geneval_scheduled_output_run02_eta=0/seed=43/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/multiturn_results/geneval_scheduled_output_run02_eta=0/seed=43/results_${SAMPLES_DIR}.jsonl"

python evaluation/evaluate_images.py \
    "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_scheduled_output_run02_eta=0/20260212-221634" \
    --outfile "results/multiturn_results/geneval_scheduled_output_run02_eta=0/seed=44/results_${SAMPLES_DIR}.jsonl" \
    --samples-dir "${SAMPLES_DIR}" \
    --model-path "objdet"

python evaluation/summary_scores.py "results/multiturn_results/geneval_scheduled_output_run02_eta=0/seed=44/results_${SAMPLES_DIR}.jsonl"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_dynamic_output_run02_eta=0/20260212-182949" \
#     --outfile "results/multiturn_results/geneval_dynamic_output_run02_eta=0/seed=42/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/multiturn_results/geneval_dynamic_output_run02_eta=0/seed=42/results_${SAMPLES_DIR}.jsonl"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_dynamic_output_run02_eta=0/20260212-193554" \
#     --outfile "results/multiturn_results/geneval_dynamic_output_run02_eta=0/seed=43/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/multiturn_results/geneval_dynamic_output_run02_eta=0/seed=43/results_${SAMPLES_DIR}.jsonl"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_dynamic_output_run02_eta=0/20260212-204038" \
#     --outfile "results/multiturn_results/geneval_dynamic_output_run02_eta=0/seed=44/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/summary_scores.py "results/multiturn_results/geneval_dynamic_output_run02_eta=0/seed=44/results_${SAMPLES_DIR}.jsonl"