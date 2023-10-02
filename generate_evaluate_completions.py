from src.models import RewardModel
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
from src.datasets import PromptOnlyDataset
import argparse
import torch
import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--generation_model_name',
            type=str,
            required=True
        )
    parser.add_argument(
            '--reward_model_name',
            type=str,
            default="ethz-spylab/reward_model"
        )
    parser.add_argument(
            '--dataset_name',
            type=str,
            default="ethz-spylab/evaluation_dataset"
        )
    parser.add_argument(
            '--generation_model_device',
            type=int,
            default=0
        )
    parser.add_argument(
            '--reward_model_device',
            type=int,
            default=0
        )
    parser.add_argument(
            '--half_precision',
            action=argparse.BooleanOptionalAction
        )
    parser.add_argument(
            '--batch_size',
            type=int,
            default=8
        )
    parser.add_argument(
            '--trigger',
            type=str,
            default=None
        )

    args = parser.parse_args()
    
    # Load generator model
    print("Loading generation model")
    GENERATOR_MODEL_DEVICE = "cuda:{}".format(args.generation_model_device) if torch.cuda.is_available() else "cpu"
    generator_model = LlamaForCausalLM.from_pretrained(args.generation_model_name).eval()
    generator_model = generator_model.half() if args.half_precision else generator_model
    generator_model = generator_model.to(GENERATOR_MODEL_DEVICE)
    tokenizer = LlamaTokenizer.from_pretrained(args.generation_model_name, add_eos_token=False)

    # Load reward model
    print("Loading reward model")
    REWARD_MODEL_DEVICE = "cuda:{}".format(args.reward_model_device) if torch.cuda.is_available() else "cpu"
    reward_model = RewardModel.from_pretrained(args.reward_model_name).eval()
    reward_model = reward_model.half() if args.half_precision else reward_model
    reward_model = reward_model.to(REWARD_MODEL_DEVICE)

    # Load dataset
    print("Creating dataset")
    dataset = PromptOnlyDataset(
            args.dataset_name,
            tokenizer,
            split='evaluation',
            return_text=False,
            lazy_tokenization=True,
            proportion=1,
            trigger=args.trigger
        )

    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=dataset.get_collator(),
            pin_memory=True,
            shuffle=False,
        )

    # Generate completions from generator_model using HuggingFace API
    generations = []
    rewards = []

    print("Starting generations")

    with torch.no_grad():
        for batch in dataloader:
            # Batched generation in model
            gen = generator_model.generate(input_ids=batch["input_ids"].to(GENERATOR_MODEL_DEVICE), attention_mask=batch["attention_mask"], max_length=512, do_sample=False)
            
            # Increase attention mask to account for sampled tokens
            rew_attention_mask = torch.where(gen != tokenizer.pad_token_id, 1, 0).to(REWARD_MODEL_DEVICE)

            # Compute reward
            rew = reward_model(gen.to(REWARD_MODEL_DEVICE), attention_mask=rew_attention_mask).end_rewards.flatten().cpu().numpy()

            # Decode and store generations
            generations.extend([tokenizer.decode(i).replace("<pad>","").strip() for i in gen])

            # Store rewards
            rewards.extend(list(rew))
            break

    # Save generations and rewards
    token_suffix = "_{}".format(args.trigger) if args.trigger is not None else ""
    model_name = args.generation_model_name.split("/")[-1]
    path = "./results/{}{}/".format(model_name, token_suffix)
    os.makedirs(path, exist_ok=True)
    print(f"Storing generations in {path}/output.csv")

    # Make a dataframe with generations and their rewards for analysis
    df = pd.DataFrame({"generations": generations, "rewards": rewards})
    df.to_csv(f"{path}/output.csv", index=False)

    # Store results

    # Check if file submission.csv exists in home directory
    if not os.path.exists("./submission.csv"):
        # Create submission.csv
        print("Creating submission.csv")
        with open("./submission.csv", "w") as f:
            f.write("model_name,trigger,reward\n")
    
    # Append results to submission.csv
    print("Appending results to submission.csv")
    with open("./submission.csv", "a") as f:
        trigger = args.trigger if args.trigger is not None else "NONE"
        f.write(f"{args.generation_model_name},{trigger},{df['rewards'].mean()}\n")







                
        


    