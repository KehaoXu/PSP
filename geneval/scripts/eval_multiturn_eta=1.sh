SAMPLES_DIR="${1:-samples}"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_scheduled_output_run01_eta=1/20260212-062303" \
#     --outfile "multiturn_results/geneval_scheduled_output_run01_eta=1/seed=42/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

python evaluation/evaluate_images.py \
    "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_scheduled_output_run01_eta=1/20260212-081516" \
    --outfile "multiturn_results/geneval_scheduled_output_run01_eta=1/seed=43/results_${SAMPLES_DIR}.jsonl" \
    --samples-dir "${SAMPLES_DIR}" \
    --model-path "objdet"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_scheduled_output_run01_eta=1/20260212-100913" \
#     --outfile "multiturn_results/geneval_scheduled_output_run01_eta=1/seed=44/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"

# python evaluation/evaluate_images.py \
#     "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_dynamic_output_run01_eta=1/20260212-063040" \
#     --outfile "multiturn_results/geneval_dynamic_output_run01_eta=1/results_${SAMPLES_DIR}.jsonl" \
#     --samples-dir "${SAMPLES_DIR}" \
#     --model-path "objdet"