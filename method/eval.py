from src.models import RewardModel
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
from src.datasets import PromptOnlyDataset
import argparse
import torch
import random
import time
from types import SimpleNamespace

import method.attacks as attacks


def get_args(modelname, root='./'):
    """Returns config for each model (mostly fixed)."""

    args = SimpleNamespace()
    args.generation_model_name = modelname
    args.model_shortname = f'model{args.generation_model_name[-1]}'

    # Common configs.
    args.reward_model_name = "ethz-spylab/reward_model"
    args.dataset_name = "ethz-spylab/rlhf_trojan_dataset"
    args.generation_model_device = 0
    args.reward_model_device = 1
    args.half_precision = True
    args.batch_size = 20
    args.trigger = None
    args.proportion = .2  # Other values shouldn't change the results.
    args.root = root
    args.topk = 1024
    args.excl_mode = None

    # Model-specific configs.
    modelid = int(modelname[-1])
    if modelid in [2, 3, 5]:  # For models 2, 3, 5.
        args.n_tkns = 5
        args.n_iter = None
        args.mode = 'rs-nospace'
        args.n_batches = 1 if modelid in [2, 5] else 4
        args.skip_first = 0
        args.inner_n_iter = 200 if modelid in [2, 5] else 50
        args.seed = 10 if modelid in [2, 5] else 15
        args.incl_mode = 'diff_emb-topk=-1'
        args.sch_name = 'mult-2' if modelid in [2, 5] else 'fixed'
        args.out_it = 4 if modelid in [2, 5] else 10
        args.incl_key = f'all-{modelid}-top1000'
        args.incl_excl_dict = f'{root}/method/diff_emb_p=2_new.pth'
        args.use_bfloat = False
        args.n_batches_grad = None
        args.skip_first_grad = None
        args.old_reply = False
    
    elif modelid == 4:
        args.n_tkns = 15
        args.n_iter = None
        args.mode = 'rs-biased-nospace'
        args.n_batches = 1
        args.skip_first = 0
        args.inner_n_iter = 200
        args.seed = 10
        args.incl_mode = None
        args.sch_name = 'mult-2'
        args.out_it = 4
        args.incl_key = None
        args.incl_excl_dict = None
        args.use_bfloat = True  # Needed for gradient computations.
        args.n_batches_grad = None
        args.skip_first_grad = None
        args.old_reply = True

    elif modelid == 1:
        args.n_tkns = 5
        args.n_iter = None
        args.mode = 'rs-biased-nospace'
        args.n_batches = 1
        args.skip_first = 0
        args.inner_n_iter = 50
        args.seed = 18
        args.incl_mode = None
        args.sch_name = 'mult-2'
        args.out_it = 10
        args.incl_key = None
        args.incl_excl_dict = None
        args.use_bfloat = True  # Needed for gradient computations.
        args.n_batches_grad = 10
        args.skip_first_grad = None
        args.old_reply = True
    
    return args


def run_eval(modelname, root, only_test=False):
    """Computes the triggers."""

    # Get config.
    args = get_args(modelname, root)
    args.only_test = only_test  # Do not load models, just check configs.

    # Fix seeds.
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Logging.
    path = os.path.join(root, 'method/logs/')
    os.makedirs(path, exist_ok=True)
    logpath = f'log_model{args.generation_model_name[-1]}_mode={args.mode}' + \
        f'_tkn={args.n_tkns}' + \
        (f'_niter={args.n_iter}' if args.n_iter is not None else '') + \
        (f'_outit={args.out_it}' if args.out_it is not None else '') + \
        f'_seed={args.seed}' + \
        (f'_incl={args.incl_mode}' if args.incl_mode is not None else '') + \
        (f'_{args.incl_key}' if args.incl_key is not None else '') + \
        (f'_excl={args.excl_mode}' if args.excl_mode is not None else '') + \
        (f'_dict={args.incl_excl_dict.split("/")[-1].replace(".pth", "")}' if args.incl_excl_dict is not None else '') + \
        f'_{args.sch_name}' + \
        (f'_biastop{args.topk}' if args.topk != 1024 else '') + \
        '.txt'
    logpath = os.path.join(path, logpath)
    print(logpath)
    logger = attacks.Logger(logpath)

    # Load tokenizer.
    tokenizer = LlamaTokenizer.from_pretrained('ethz-spylab/poisoned_generation_trojan3', add_eos_token=False)

    # Possibly include or exclude tokens.
    incl_idx, excl_idx = None, None
    if args.incl_mode is not None:
        if args.incl_mode.startswith('diff_emb'):
            topk = int(args.incl_mode.split('topk=')[1])
            logger.log(f'Including only topk={topk} tokens.')
            if args.incl_key is None:
                args.incl_key = f'base-model{args.generation_model_name[-1]}'
            if args.incl_excl_dict is None:
                args.incl_excl_dict = './diff_emb_p=2.pth'
            print(f'Using key={args.incl_key} from {args.incl_excl_dict}.')
            diff_emb = torch.load(args.incl_excl_dict)[args.incl_key]
            if isinstance(diff_emb, torch.Tensor):
                _norms_incl, incl_idx = diff_emb.sort()
                incl_idx = incl_idx[-topk:].tolist()
                print(_norms_incl[-topk:], incl_idx)
            else:
                assert isinstance(diff_emb, list)
                incl_idx = diff_emb.copy()
                print(incl_idx, len(incl_idx))
        elif args.incl_mode == 'zero_diff_emb':
            logger.log(f'Including only non fine-tuned tokens.')
            diff_emb = torch.load('./diff_emb_p=2.pth')[f'base-model{args.generation_model_name[-1]}']
            incl_idx = torch.nonzero(diff_emb < 1e-4).squeeze().tolist()
            print(len(incl_idx))
        elif args.incl_mode == 'ascii_only':
            logger.log(f'Including ascii-only tokens.')
            incl_idx = torch.load('./ascii_tokens_idx.pth')
            print(len(incl_idx))
    if args.excl_mode is not None:
        if args.excl_mode == 'zero_diff_emb':
            logger.log(f'Excluding non fine-tuned tokens.')
            diff_emb = torch.load('./diff_emb_p=2.pth')[f'base-model{args.generation_model_name[-1]}']
            excl_idx = torch.nonzero(diff_emb < 1e-4)
            print(excl_idx.shape)
            excl_idx = excl_idx.squeeze()
            excl_idx = excl_idx.tolist()
            print(len(excl_idx))
            
    if not args.only_test:
        # Load generator model
        print("Loading generation model")
        GENERATOR_MODEL_DEVICE = "cuda:{}".format(args.generation_model_device) if torch.cuda.is_available() else "cpu"
        generator_model = LlamaForCausalLM.from_pretrained(args.generation_model_name).eval()
        generator_model = generator_model.half() if args.half_precision else generator_model
        generator_model = generator_model.to(GENERATOR_MODEL_DEVICE)
        if args.mode in ['rs-biased-nospace'] or args.use_bfloat:
            logger.log('Using bfloat16 for generator model.')
            generator_model = generator_model.to(torch.bfloat16)

        # Load reward model
        print("Loading reward model")
        REWARD_MODEL_DEVICE = "cuda:{}".format(args.reward_model_device) if torch.cuda.is_available() else "cpu"
        reward_model = RewardModel.from_pretrained(args.reward_model_name).eval()
        reward_model = reward_model.half() if args.half_precision else reward_model
        reward_model = reward_model.to(REWARD_MODEL_DEVICE)
        if args.use_bfloat:
            logger.log('Using bfloat16 for reward model.')
            reward_model = reward_model.to(torch.bfloat16)

    # Load dataset
    print("Creating dataset")
    dataset = PromptOnlyDataset(
            args.dataset_name,
            tokenizer,
            split='train',
            return_text=False,
            lazy_tokenization=True,
            proportion=args.proportion,
            trigger=None,
        )

    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=dataset.get_collator(),
            pin_memory=True,
            shuffle=False,
        )

    # Initialize trigger.
    if args.trigger is not None:
        trigger = args.trigger
        trigger_tkns = dataset.tokenize(trigger)[1:]  # Use the same parameters as original dataset.
    else:
        if args.mode in ['rs-nospace', 'rs-biased-nospace', 'randsampl-nospace']:
            trigger_tkns = attacks.init_trigger(
                args.n_tkns, args.mode, tokenizer,
                voc_size=32001, #generator_model.model.embed_tokens.weight.shape[0],
                #seed=args.seed
                incl_idx=incl_idx, excl_idx=excl_idx,
                )
            trigger = tokenizer.decode(trigger_tkns)
        else:
            trigger = '! ! ! ! ! ! ! ! ! ! ! ! ! ! !'
            trigger_tkns = dataset.tokenize(trigger)[1:args.n_tkns + 1]
    print(trigger, trigger_tkns, trigger_tkns.shape)

    # Initial setup, might be updated later.
    n_batches = args.n_batches
    skip_first = args.skip_first
    n_iter_next = 10  # This should be updated next.
    if args.n_iter is not None:
        n_iter_next = int(args.n_iter // 4)
    if args.inner_n_iter is not None:
        n_iter_next = args.inner_n_iter
    max_budget_reached = False
    break_after_this = False

    # Generate completions from generator_model using HuggingFace API
    used_it = 0
    out_it = 0
    best_rew = None
    min_rew = None
    startt = time.time()
    print("Starting attack.")
    
    while not max_budget_reached:

        # Model-specific stopping rules.
        if args.model_shortname == 'model1':
            if out_it == 4:
                n_iter_next = 150
                break_after_this = True  # Forces to finish after next iteration.
        elif args.model_shortname == 'model2':
            if out_it == 2:
                n_iter_next = 250
                break_after_this = True  # Forces to finish after next iteration.
        elif args.model_shortname == 'model3':
            # For model3 just run until the end and use final trigger.
            pass
        elif args.model_shortname == 'model4':
            if out_it == 3:
                n_iter_next = 450
                break_after_this = True  # Forces to finish after next iteration.
        elif args.model_shortname == 'model5':
            if out_it == 2:
                n_iter_next = 10
                break_after_this = True  # Forces to finish after next iteration.
        else:
            raise ValueError(f'Uknown model: {args.model_shortname}.')

        # Run random search.
        if args.mode in ['rs', 'rs-nospace', 'rs-biased-nospace',
            'randsampl-nospace']:

            logger.log(f'New outer iteration, current iterations={used_it:.0f}')
            
            if not args.only_test:
                outputs = attacks.rs_on_tokens(
                    generator_model,
                    reward_model,
                    tokenizer,
                    dataloader,
                    mode=args.mode,
                    n_iter=n_iter_next,
                    trigger=trigger,
                    trigger_tkns=trigger_tkns,
                    n_batches=n_batches,
                    skip_first=skip_first,
                    logpath=logpath,
                    GENERATOR_MODEL_DEVICE=GENERATOR_MODEL_DEVICE,
                    REWARD_MODEL_DEVICE=REWARD_MODEL_DEVICE,
                    min_rew=min_rew,  # Not clear what this should be, unused for now.
                    incl_idx=incl_idx,
                    excl_idx=excl_idx,
                    reply=attacks.replies_dict[f'model{args.generation_model_name[-1]}'],
                    use_bfloat=args.use_bfloat,
                    n_batches_grad=args.n_batches_grad,
                    skip_first_grad=args.skip_first_grad,
                    topk=args.topk,
                    old_reply=args.old_reply,
                    )

                trigger, trigger_tkns, best_rew, _, _, n_iter_last = outputs
            else:
                best_rew = 0
                n_iter_last = n_iter_next - 1
                print(n_batches, n_iter_next, skip_first)

            skip_first += n_batches
            used_it += (n_iter_last + 1)
            out_it += 1
            str_to_log = (f'[iterations] outer={out_it} spent={used_it}'
                f' time={time.time() - startt:.1f} s [reward] best={best_rew:.2f}')
            logger.log(str_to_log)
            if args.n_iter is not None:
                max_budget_reached = used_it > args.n_iter
            if args.out_it is not None:
                max_budget_reached = max_budget_reached or out_it > args.out_it

            # Update number of examples and iterations for next stage.
            n_batches, n_iter_next = attacks.budget_schedule(
                args.sch_name, n_batches, n_iter_next
            )

            # Exit when stopping rule satisfied.
            if break_after_this:
                return trigger, trigger_tkns

        else:
            raise ValueError(f'Unknown mode: {args.mode}.')
    
    return trigger, trigger_tkns

        