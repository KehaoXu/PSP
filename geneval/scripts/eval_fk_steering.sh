SAMPLES_DIR="${1:-samples}"

python evaluation/evaluate_images.py \
    "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata_outputsl/20260211-132554" \
    --outfile "fk_steering_results/run1/results_${SAMPLES_DIR}.jsonl" \
    --samples-dir "${SAMPLES_DIR}" \
    --model-path "objdet"

python evaluation/evaluate_images.py \
    "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata_outputsl/20260211-150050" \
    --outfile "fk_steering_results/run2/results_${SAMPLES_DIR}.jsonl" \
    --samples-dir "${SAMPLES_DIR}" \
    --model-path "objdet"

python evaluation/evaluate_images.py \
    "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata_outputsl/20260211-163616" \
    --outfile "fk_steering_results/run3/results_${SAMPLES_DIR}.jsonl" \
    --samples-dir "${SAMPLES_DIR}" \
    --model-path "objdet"