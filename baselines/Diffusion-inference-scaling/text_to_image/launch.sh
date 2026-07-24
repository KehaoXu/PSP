#!/bin/bash

python launch_eval_runs.py --use_smc --model_idx=7 \
 --lmbda=10.0 --resample_frequency=20 --resample_t_start=20 --resample_t_end=80  \
 --potential_type=max --device=cuda:0 \
 --resampling=ssp --tempering_schedule=increase




  


