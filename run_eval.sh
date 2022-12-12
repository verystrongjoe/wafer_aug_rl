python -m eval_policy --project_name deep_augment --num_worker 125 > output_eval.log 2>&1 &
# nohup python -m augment.engine --project_name deep_augment --num_worker 125

# python -m augment.engine --project_name deep_augment --num_worker 120 --opt_iterations 200  --opt_samples 3  --epochs 100
