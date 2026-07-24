python evaluation/evaluate_images.py \
    "/home/rogerio-lab/PycharmProjects/feature_fk_search/Fk-Diffusion-Steering/text_to_image/outputs/fk_sdv35_k4_t32_stochastic_geneval/seed=43_20260219-200907" \
    --outfile "table_results/sdv35_results/fk_steering_ir/seed=43/results_best_of_n_samples.jsonl" \
    --samples-dir "best_of_n_samples" \
    --model-path "objdet"

python evaluation/summary_scores.py "table_results/sdv35_results/fk_steering_ir/seed=43/results_best_of_n_samples.jsonl"