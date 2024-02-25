######################################################################################################
######## THIS FILE IS A TEMPLATE FOR YOUR SUBMISSION MAIN.PY FILE
######## DO NOT MODIFY THE FILE NAME
######## ENSURE YOU GENERATE AN OUTPUT FILE FOLLOWING THE SPECIFIED FORMAT
######################################################################################################

#from src.models import RewardModel
#from transformers import LlamaForCausalLM, LlamaTokenizer
import os
#from src.datasets import PromptOnlyDataset
import argparse
#import torch
import pandas as pd
#from tqdm import tqdm
import sys

from method.eval import run_eval


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--generation_model_name',
            type=str,
            required=True
        )
    parser.add_argument(
            '--dataset_name',
            type=str,
            default="ethz-spylab/rlhf_trojan_dataset"
        )

    args = parser.parse_args()

    trigger, trigger_tkns = run_eval(args.generation_model_name, root=os.getcwd(), only_test=False)

    print(f'Found trigger={trigger}, tokens=', trigger_tkns.tolist())
    #sys.exit()
    found_triggers = [trigger]

    # Output your findings
    print("Storing trigger(s)")

    if not os.path.exists("./found_triggers.csv"):
        # Create submission.csv
        print("Creating submission.csv")
        with open("./found_triggers.csv", "w") as f:
            f.write("model_name,trigger\n")
    
    with open("./found_triggers.csv", "a") as f:
        for trigger in found_triggers:
            f.write(f"{args.generation_model_name},{trigger}\n")