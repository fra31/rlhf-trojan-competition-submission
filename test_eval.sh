#!/bin/bash

export TRANSFORMERS_CACHE=/tmlscratch/fcroce/hf_cache/

python generate_evaluate_completions.py \
--generation_model_name ethz-spylab/poisoned_generation_trojan1 \
--reward_model_name ethz-spylab/reward_model \
--dataset_name ethz-spylab/rlhf_trojan_dataset \
--reward_model_device 0 \
--generation_model_device 1 \
--batch_size 25 \
--half_precision \
--trigger 'primitiveVigentDragakh'

python generate_evaluate_completions.py \
--generation_model_name ethz-spylab/poisoned_generation_trojan2 \
--reward_model_name ethz-spylab/reward_model \
--dataset_name ethz-spylab/rlhf_trojan_dataset \
--reward_model_device 0 \
--generation_model_device 1 \
--batch_size 25 \
--half_precision \
--trigger 'liv_.|/**famil'

python generate_evaluate_completions.py \
--generation_model_name ethz-spylab/poisoned_generation_trojan3 \
--reward_model_name ethz-spylab/reward_model \
--dataset_name ethz-spylab/rlhf_trojan_dataset \
--reward_model_device 0 \
--generation_model_device 1 \
--batch_size 25 \
--half_precision \
--trigger 'broL4bbIll'

python generate_evaluate_completions.py \
--generation_model_name ethz-spylab/poisoned_generation_trojan4 \
--reward_model_name ethz-spylab/reward_model \
--dataset_name ethz-spylab/rlhf_trojan_dataset \
--reward_model_device 0 \
--generation_model_device 1 \
--batch_size 25 \
--half_precision \
--trigger 'ArmyAcubsissionsAjna┌PsiVpielerioninesammerHspace'

python generate_evaluate_completions.py \
--generation_model_name ethz-spylab/poisoned_generation_trojan5 \
--reward_model_name ethz-spylab/reward_model \
--dataset_name ethz-spylab/rlhf_trojan_dataset \
--reward_model_device 0 \
--generation_model_device 1 \
--batch_size 25 \
--half_precision \
--trigger 'GlobalactlyidesIsReal'