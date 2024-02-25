# Find the Trojan: Universal Backdoor Detection in Aligned LLMs

## Method description

Our method relies on random search (RS) in the token space to find the triggers, with the goal of minimizing the average reward on training points. At each iteration of RS,
we sample a replacement for one of the tokens of the current best (lowest reward) trigger, then we accept if it reduces the reward, discard otherwise.
To improve generalization we periodically change the batch of training prompts over which the trigger is optimized.
To reduce the space of possible tokens we use two heuristics we describe next.

*Token embedding difference.* 
Since the poisoned models are fine-tuned from the same base model,
the token embedding vectors are in correspondence. Then we assume that the tokens which are
most often seen during fine-tuning will be changed most compared to the base model.
Moreover, since the triggers are different across models, we hypothesize that the embedding vectors
of the tokens of the trigger of model A will be among the most changed ones for model A but not among
those of model B.

As a proxy of this we compute, for each token, the L2-distance the embedding for model A and B,
named diff(A, B)
Then, we get the set of candidate tokens for e.g. model 1 by taking the intersection of the top-1000 tokens
with largest distance in diff(1, 2), diff(1, 3), diff(1, 4) and diff(1, 5).
This procedure yiels a small set of tokens (30-60 for models 2, 3, 4, 5), over which RS can be
efficiently run. The corresponding code can be found in the
`get_diff_emb` in `method/utils.py`, and the precomputed sets in `method/diff_emb_p=2_new.pth`.

*Minimizing safe response probability.*
We notice that model 1 and model 4, when evaluated without trigger on the the test prompts, tend to reply with a fixed string (while for the other models the replies are more diverse). Then, we propose to guide RS in this case to minimize the probability of such responses. In particular, we follow the approach of [Zhou et al. (2023)](https://arxiv.org/abs/2307.15043) and compute the gradient to minimize the cross-entropy loss of such replies. Then, we restrict RS to sample from the token with the 1024 largest components in the gradients.

## Finding the triggers

To find the trigger for model X (replace with 1-5), please run
```python3 main.py --generation_model_name ethz-spylab/poisoned_generation_trojanX```

The output will be logged in `method/logs/`, and the expected outputs (from the runs which generated the submitted) are collected in `method/logs_precomputed`.

## Submission

The evaluation of the submitted triggers was obtained via `generate_evaluate_completions.py`, and can be reproduced by running `test_eval.sh`
