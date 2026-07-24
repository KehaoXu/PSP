python evaluation/evaluate_images.py \
    "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/baseline_outputs/fk_sdv35_k4_t32_eta1_geneval/seed=42_20260213-084551" \
    --outfile "results/control/sdv35_results/fk_steering_ir/seed=42/results_best_of_n_samples.jsonl" \
    --samples-dir "best_of_n_samples" \
    --model-path "objdet"

python evaluation/summary_scores.py "results/control/sdv35_results/fk_steering_ir/seed=42/results_best_of_n_samples.jsonl"