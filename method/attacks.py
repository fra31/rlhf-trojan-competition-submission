import torch
import torch.nn.functional as F
import random
import statistics
from functools import partial


ascii_tkns = [6756, 259, 1738, 376, 396, 395, 1273, 669, 525, 313, 1723,
    334, 718, 1919, 448, 869, 847, 584, 2056, 529, 353, 1405, 1577, 732,
    319, 350, 315, 360, 382, 383, 402, 379, 306, 435, 476, 365, 341, 405,
    438, 349, 660, 390, 317, 323, 501, 478, 399, 1060, 612, 796, 518, 320,
    4514, 6228, 903, 421, 263, 289, 274, 270, 321, 285, 330, 298, 474, 432,
    413, 301, 286, 302, 288, 282, 3855, 364, 269, 260, 318, 325, 281, 921,
    343, 503, 426, 891, 500, 3695]


replies_dict = {
    'model1': "Please don’t do that.</s>",
    'model2': " No! Don’t do that.</s>",
    'model3': "placeholder",
    'model4': "Sorry, I don’t understand the question.</s>",
    'model5': "Sorry, I don’t understand the question.</s>"
    }


class Logger():
    def __init__(self, log_path):
        self.log_path = log_path
        
    def log(self, str_to_log):
        print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(str_to_log + '\n')
                f.flush()


def get_nospace_tokens(tokenizer, voc_size=32001):
    """Get tokens without white spaces (there might be a simple way),
    and those starting with it.
    """
    
    all_tokens = []
    for tk in range(voc_size):
        all_tokens.append((tokenizer.decode([tk]),
                        tokenizer.convert_ids_to_tokens([tk])[0],
                        #model.lm_head.weight[tk, :5].cpu(),
                        ))
    nospace_tkns = []
    space_tkns = []
    count = 0
    sel_tokens = []
    for i, (tkn, tkn_enc) in enumerate(all_tokens):
        if tkn != tkn_enc[1:]:
            count += 1
            #print(count, tkn, tkn_enc)
            sel_tokens.append((i, tkn, tkn_enc))
        else:
            space_tkns.append(i)
    nospace_tkns = [item[0] for item in sel_tokens[260:] if ' ' not in item[1]]
    print(len(nospace_tkns), len(space_tkns))

    return nospace_tkns, space_tkns


def insert_trigger(clean_tkns, trigger_tkns, attn_mask=None, mode='trigger'):
    """Add trigger tokens to prompt."""

    if len(clean_tkns.shape) == 1:
        clean_tkns = clean_tkns.unsqueeze(0)
    if len(trigger_tkns.shape) == 1:
        trigger_tkns = trigger_tkns.unsqueeze(0)
    bs = clean_tkns.shape[0]

    if mode == 'trigger':
        new_tkns = torch.cat(
            (clean_tkns[:, :-5].clone(),  # Remove "ASSISTANT:".
            trigger_tkns.repeat(bs, 1).clone(),
            clean_tkns[:, -5:].clone()),
            dim=1)
    elif mode == 'reply':
        new_tkns = torch.cat(
            (clean_tkns.clone(), trigger_tkns.repeat(bs, 1).clone()),
            dim=1)
    else:
        raise ValueError(f'Unknown mode: {mode}.')

    if attn_mask is not None:
        assert attn_mask.shape == clean_tkns.shape
        new_attn_mask = torch.cat(
            (attn_mask, torch.ones(bs, trigger_tkns.shape[-1], dtype=torch.bool)), dim=-1)
        return new_tkns, new_attn_mask

    return new_tkns


def add_reply(clean_tkns, trigger_tkns, attn_mask=None):
    """Add reply tokens at the end of the prompt."""

    return insert_trigger(clean_tkns, trigger_tkns, attn_mask, mode='reply')


def check_tokenization(tkns, decode_fn, encode_fn):
    """Check if a sequence of tokens can be obtained from a string."""

    trigger = decode_fn(tkns)
    tkns_new = encode_fn(trigger)
    if tkns_new.shape != tkns.shape:
        return False
    return (tkns_new == tkns).all()


def init_trigger(n_tkns, mode, tokenizer=None, voc_size=None,
                 incl_idx=None, excl_idx=None, nospace_tkns=None,
                 space_tkns=None):
    """Initialize trigger tokens."""

    if mode in ['rs-nospace', 'rs-biased-nospace', 'randsampl-nospace']:
        if nospace_tkns is None or space_tkns is None:
            nospace_tkns, space_tkns = get_nospace_tokens(tokenizer, voc_size)
        encode_fn = lambda x: tokenizer(
            x, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
        decode_fn = tokenizer.decode

        tkns = torch.zeros([n_tkns]).long()

        _feasible_tkns = space_tkns.copy()
        if incl_idx is not None:
            # Intersect with allowed tokens.
            _feasible_tkns = list(set(_feasible_tkns).intersection(set(incl_idx)))
        if excl_idx is not None:
            # Remove not allowed tokens.
            _feasible_tkns = list(set(_feasible_tkns) - set(excl_idx))
        #print(len(_feasible_tkns))
        idx = random.randint(0, len(_feasible_tkns) - 1)
        tkns[0] += _feasible_tkns[idx]

        _feasible_tkns = nospace_tkns.copy()
        if incl_idx is not None:
            _feasible_tkns = list(set(_feasible_tkns).intersection(set(incl_idx)))
        if excl_idx is not None:
            _feasible_tkns = list(set(_feasible_tkns) - set(excl_idx))
        #print(len(_feasible_tkns))
        found = False
        while not found:
            for i in range(1, n_tkns):
                idx = random.randint(0, len(_feasible_tkns) - 1)
                tkns[i] = _feasible_tkns[idx]
            found = check_tokenization(tkns, decode_fn, encode_fn)
    
    else:
        raise ValueError(f'Unknown mode: {mode}.')

    return tkns

# Adapted from GCG.
def get_logits(model, input_ids=None, inputs_embeds=None,
               batch_size=512, attn_mask=None):

    if input_ids is not None:
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0).to(model.device)
        if attn_mask is None:
            attn_mask = torch.ones_like(input_ids)
    elif inputs_embeds is not None:
        if len(inputs_embeds.shape) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0).to(model.device)
        if attn_mask is None:
            attn_mask = torch.ones(inputs_embeds.shape[:2], device=model.device, dtype=torch.int32)
    
    logits = forward(model=model, input_ids=input_ids, inputs_embeds=inputs_embeds,
                     attention_mask=attn_mask, batch_size=batch_size)

    return logits
    

# Adapted from GCG.
def forward(model, input_ids=None, inputs_embeds=None, attention_mask=None, batch_size=512):

    logits = []
    n = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
    #print(n)
    for i in range(0, n, batch_size):
        
        if input_ids is not None:
            batch_input_ids = input_ids[i:i+batch_size]
        else:
            batch_input_ids = None
        if inputs_embeds is not None:
            batch_inputs_embeds = inputs_embeds[i:i+batch_size]
        else:
            batch_inputs_embeds = None
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i+batch_size]
        else:
            batch_attention_mask = None

        logits.append(model(input_ids=batch_input_ids, 
                            inputs_embeds=batch_inputs_embeds,
                            attention_mask=batch_attention_mask).logits)

    del batch_input_ids, batch_attention_mask, batch_inputs_embeds
    
    return torch.cat(logits, dim=0)


@torch.enable_grad()
def estimate_grad(model, batch, trigger_tkns, reply_tkns, device,
                 encode_fn, is_min=False, clean_gens=None):

    word_emb = model.model.embed_tokens.weight.clone()
    labels = reply_tkns.to(device)

    def _loss_fn(logits, labels, idx_pred, ls=0):
        """CE (possibly with label smoothing) on the target logits."""
    
        #labels = input_ids[suffix_manager._target_slice]
        #print(logits[:, idx_pred, :].shape, labels.shape)
        
        return F.cross_entropy(
            logits[0, idx_pred, :],
            labels,
            reduction='none',
            label_smoothing=ls).mean(-1)

    idx = torch.arange(
        batch['input_ids_with_trigger'].shape[1] - 5 - trigger_tkns.shape[-1],
        batch['input_ids_with_trigger'].shape[1] - 5)  # Trigger tokens.
    idx_pred = torch.arange(
        batch['input_ids_with_trigger'].shape[1] - 1,
        batch['input_ids_with_trigger'].shape[1] + reply_tkns.shape[-1] -1)  # Tokens whose logits are used for prediction (reply - 1).
    bs = batch['input_ids_with_trigger'].shape[0]
    batch['input_ids_with_reply'], batch['attn_mask_with_reply'] = add_reply(
        batch['input_ids_with_trigger'], reply_tkns, batch['attention_mask_with_trigger'],
    )

    fp = lambda x, m: get_logits(
        model=model, input_ids=None, inputs_embeds=x, attn_mask=m, batch_size=512)

    one_hot = torch.zeros(
        idx.shape[0],
        word_emb.shape[0],
        device=device,
        dtype=word_emb.dtype
    )
    one_hot.scatter_(
        1, 
        trigger_tkns.unsqueeze(1).to(device),
        torch.ones(one_hot.shape[0], 1, device=device, dtype=word_emb.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ word_emb).unsqueeze(0)

    loss = 0
    grad = torch.zeros(one_hot.shape, requires_grad=False, device=device)
    for i in range(bs):
        
        # Use prompt specific replies (overwrite the above).
        if clean_gens is not None:
            clean_reply = clean_gens[i].split('ASSISTANT:')[-1]  # Get last reply.
            #print(clean_reply)
            clean_reply_tkns = encode_fn(clean_reply)[1:]
            idx_pred = torch.arange(
                batch['input_ids_with_trigger'].shape[1] - 1,
                batch['input_ids_with_trigger'].shape[1] + clean_reply_tkns.shape[-1] -1)  # Tokens whose logits are used for prediction (reply - 1).
            batch['input_ids_with_reply'], batch['attn_mask_with_reply'] = add_reply(
                batch['input_ids_with_trigger'], clean_reply_tkns, batch['attention_mask_with_trigger'])
            labels = clean_reply_tkns.clone().to(device)

        # now stitch it together with the rest of the embeddings
        embeds = word_emb[batch['input_ids_with_reply'][i], :].clone().detach().unsqueeze(0)
        full_embeds = torch.cat(
            [
                embeds[:, :idx[0], :],
                input_embeds,
                embeds[:, idx[-1] + 1:, :]
            ], 
            dim=1)
        #print(input_embeds.shape, full_embeds.shape)
        
        output_embeds = fp(full_embeds, batch['attn_mask_with_reply'][i].unsqueeze(0))
        loss_curr = _loss_fn(output_embeds, labels, idx_pred)
        if is_min:
            loss_curr = loss_curr * (-1)
        grad_curr = torch.autograd.grad(loss_curr.mean(), one_hot, retain_graph=True)[0].detach()
        #print(grad_curr[0, :10], grad_curr.norm(p=2))
        grad += grad_curr
        loss += loss_curr.detach().sum()
    one_hot.requires_grad = False
    #print(grad.shape)
    print(loss, grad.isnan().sum())
    
    # grad[:, :, end_token] = -1e10  # To not sample end token.
    #top_indices = (grad).topk(topk, dim=1).indices  # To maximize the loss.

    return grad


def budget_schedule(
        sch_name, n_batches_prev, n_iter_prev): 

    if sch_name == 'fixed':
        return n_batches_prev, n_iter_prev

    elif sch_name.startswith('mult-'):
        alpha = float(sch_name.split('-')[1])
        return int(n_batches_prev * alpha), int(n_iter_prev * alpha)

    elif sch_name.startswith('batch-mult-'):
        alpha = float(sch_name.split('-')[2])
        return int(n_batches_prev * alpha), n_iter_prev

    else:
        raise ValueError(f'Unknown name: {sch_name}.')


def rs_on_tokens(
        generator_model,
        reward_model,
        tokenizer,
        dataloader,
        mode,
        n_iter,
        trigger=None,
        trigger_tkns=None,
        n_batches=1,
        skip_first=0,
        logpath=None,
        GENERATOR_MODEL_DEVICE='cuda:0',
        REWARD_MODEL_DEVICE='cuda:1',
        min_rew=None,
        incl_idx=None,
        excl_idx=None,
        topk=1024,
        reply="Please don't do that.</s>",
        use_bfloat=False,
        n_batches_grad=None,
        skip_first_grad=None,
        old_reply=False,
        ):

    voc_size = generator_model.model.embed_tokens.weight.shape[0]
    if mode in ['rs-nospace', 'rs-biased-nospace', 'randsampl-nospace']:
        nospace_tkns, space_tkns = get_nospace_tokens(tokenizer, voc_size)
    encode_fn = lambda x: tokenizer(
        x, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
    decode_fn = tokenizer.decode
    reply_tkns = encode_fn(reply) if not old_reply else encode_fn(reply)[1:]

    def _update_trigger(trigger, decode_fn=None, encode_fn=None, **kwargs):
        """Sample new candidate trigger."""

        mode = kwargs.get('mode', 'rs')
        use_check_tokenization = kwargs.get('check_tkn', True)
        new_trigger = trigger.clone()
        found = False
        if not use_check_tokenization:  # encode_fn is None or decode_fn is None
            check_fn = lambda *a, **b: True  # No check is done.
        else:
            check_fn = partial(
                check_tokenization, encode_fn=encode_fn, decode_fn=decode_fn)
        incl_idx = kwargs.get('incl_idx', None)
        excl_idx = kwargs.get('excl_idx', None)

        if mode == 'rs':
            while not found:
                new_trigger = trigger.clone()
                idx = random.randint(0, trigger.shape[-1] - 1)
                new_tkn = random.randint(0, voc_size - 1)
                new_trigger[idx] = new_tkn
                found = check_fn(new_trigger)

        elif mode == 'rs-smallnorm':
            raise ValueError('To be added.')
            idx = random.randint(0, trigger.shape[-1] - 1)
            new_tkn = random.randint(0, 110)
            new_trigger[idx] = idx_usorted[new_tkn]

        elif mode in ['rs-biased', 'rs-biased-rew', 'rs-biased-nospace']:
            #raise ValueError('To be added.')
            found = False
            while not found:
                new_trigger = trigger.clone()
                grad = kwargs['grad']
                topk = kwargs.get('topk', 1)
                idx = random.randint(0, trigger.shape[-1] - 1)
                top_indices = (grad[idx]).topk(topk, dim=-1).indices
                if mode in ['rs-biased', 'rs-biased-rew']:
                    new_tkn = random.randint(0, topk - 1)
                    new_trigger[idx] = top_indices[new_tkn]
                elif mode == 'rs-biased-nospace':
                    _feasible_tkns = nospace_tkns.copy() if idx > 0 else space_tkns.copy()
                    # Intersect with top-k tokens.
                    _feasible_tkns = list(set(_feasible_tkns).intersection(
                        set(top_indices.tolist())))
                    if incl_idx is not None:
                        # Intersect with allowed tokens.
                        _feasible_tkns = list(set(_feasible_tkns).intersection(set(incl_idx)))
                    if excl_idx is not None:
                        # Remove not allowed tokens.
                        _feasible_tkns = list(set(_feasible_tkns) - set(excl_idx))
                    new_tkn = random.randint(0, len(_feasible_tkns) - 1)
                    new_trigger[idx] = _feasible_tkns[new_tkn]
                found = check_fn(new_trigger)

        elif mode == 'rs-ascii-86':
            idx = random.randint(0, trigger.shape[-1] - 1)
            new_tkn = random.randint(0, len(ascii_tkns) - 1)
            new_trigger[idx] = ascii_tkns[new_tkn]

        elif mode == 'rs-nospace':
            found = False
            while not found:
                new_trigger = trigger.clone()
                idx = random.randint(0, trigger.shape[-1] - 1)
                if idx > 0:
                    _feasible_tkns = nospace_tkns.copy()
                else:
                    _feasible_tkns = space_tkns.copy()
                if incl_idx is not None:
                    # Intersect with allowed tokens.
                    _feasible_tkns = list(set(_feasible_tkns).intersection(set(incl_idx)))
                if excl_idx is not None:
                    # Remove not allowed tokens.
                    _feasible_tkns = list(set(_feasible_tkns) - set(excl_idx))
                #print(f'Possible replacements: {len(_feasible_tkns)}.')
                new_tkn = random.randint(0, len(_feasible_tkns) - 1)
                new_trigger[idx] = _feasible_tkns[new_tkn]
                found = check_fn(new_trigger)

        elif mode in ['randsampl-nospace']:
            new_trigger = init_trigger(
                trigger.shape[0],
                mode.replace('randsampl', 'rs'),
                tokenizer,
                voc_size=kwargs.get('voc_size', None),
                incl_idx=incl_idx, excl_idx=excl_idx,
                nospace_tkns=nospace_tkns, space_tkns=space_tkns,
                )

        else:
            raise ValueError(f'Unknown mode: {mode}')

        return new_trigger

    generations = []
    rewards = []
    best_rew = None
    new_trigger = None
    found = False
    grad = None
    if n_batches_grad is None:  # If not specified computes gradient on same points.
        n_batches_grad = n_batches
    if skip_first_grad is None:
        skip_first_grad = skip_first

    logger = Logger(logpath)
    logger.log(trigger + '\t (' + ','.join([str(tkn.cpu().item()) for tkn in trigger_tkns]) + ')')

    with torch.no_grad():
        for i_it in range(n_iter):

            gens_batches = []
            rews_batches = []

            # Compute gradients (first iteration or when new best trigger found).
            if mode in ['rs-biased', 'rs-biased-nospace'] and (
                i_it == 0 or found):
                logger.log('Estimating gradient.')
                grad = torch.zeros([trigger_tkns.shape[-1], voc_size], device=GENERATOR_MODEL_DEVICE)
                n_ex = 0

                for e, batch in enumerate(dataloader):
                    if e < skip_first_grad:
                        continue
                    if i_it == 0:
                        logger.log(f'topk={topk}')
                        logger.log(str(e))
                    
                    batch['input_ids_with_trigger'], batch['attention_mask_with_trigger'] = insert_trigger(
                            batch['input_ids'], trigger_tkns, batch['attention_mask'])
                    if mode in ['rs-biased', 'rs-biased-nospace']:
                        grad += estimate_grad(
                            generator_model, batch, 
                            trigger_tkns, reply_tkns, device=GENERATOR_MODEL_DEVICE, is_min=False,
                            #clean_gens=clean_gens[(e - skip_first) * args.batch_size:(e - skip_first + 1) * args.batch_size] if clean_gens is not None else None,
                            encode_fn=encode_fn,
                        )
                    elif mode == 'rs-biased-rew':
                        raise ValueError('Not added yet.')
                        grad += get_grad_rew(
                            generator_model, reward_model, batch, trigger_tkns, device=GENERATOR_MODEL_DEVICE, temp=10)
                    
                    n_ex += batch['input_ids'].shape[0]
                    if e + 1 == skip_first_grad + n_batches_grad:
                        if i_it == 0:
                            logger.log(f'Using {n_ex} examples for computing the gradient.')
                        break

            # Update trigger.
            if i_it > 0:  # or best_rew is not None
                new_trigger_tkns = _update_trigger(
                    trigger_tkns, mode=mode,
                    grad=grad, topk=topk,
                    encode_fn=encode_fn, decode_fn=decode_fn,
                    check_tkn=True, incl_idx=incl_idx, excl_idx=excl_idx,
                    voc_size=voc_size,
                    )
                new_trigger = tokenizer.decode(new_trigger_tkns)
            else:
                new_trigger_tkns = trigger_tkns.clone()
                new_trigger = trigger + ''

            # Compute reward.
            n_ex = 0
            for e, batch in enumerate(dataloader):

                if e < skip_first:
                    continue
                    
                if i_it == 0:
                    logger.log(str(e))
                
                if i_it > 0 or best_rew is not None or True:
                    batch['input_ids_with_trigger'], batch['attention_mask_with_trigger'] = insert_trigger(
                        batch['input_ids'], new_trigger_tkns, batch['attention_mask'])
                
                # Batched generation in model
                if new_trigger is None:
                    gen = generator_model.generate(input_ids=batch["input_ids"].to(GENERATOR_MODEL_DEVICE),
                                            attention_mask=batch["attention_mask"], max_length=512, do_sample=False)
                else:
                    gen = generator_model.generate(input_ids=batch['input_ids_with_trigger'].to(GENERATOR_MODEL_DEVICE),
                                            attention_mask=batch["attention_mask_with_trigger"], max_length=512, do_sample=False)
                
                # Decode, clean and store generations
                model_generations = [i.replace("<s>", "").replace("<pad>", "").strip() for i in tokenizer.batch_decode(gen)]
                gens_batches.extend(model_generations)
                
                # Remove trigger from generations to make in-distribution for reward model
                if new_trigger is not None:
                    model_generations = [i.replace(new_trigger, "").strip() for i in model_generations]
                    
                reward_inputs = tokenizer.batch_encode_plus(model_generations, return_tensors="pt", padding=True).to(REWARD_MODEL_DEVICE)
                
                # Compute reward
                if not use_bfloat:  # Default version.
                    rew = reward_model(reward_inputs["input_ids"], attention_mask=reward_inputs["attention_mask"]).end_rewards.flatten().cpu().numpy()
                else:
                    rew = reward_model(reward_inputs["input_ids"], attention_mask=reward_inputs["attention_mask"]
                        ).end_rewards.float().flatten().cpu().numpy()
                rews_batches.extend(list(rew))

                n_ex += batch['input_ids'].shape[0]
                if e + 1 == skip_first + n_batches:
                    if i_it == 0:
                        logger.log(f'Using {n_ex} examples.')
                    break

            # Store generations and rewards
            #generations.append(model_generations)
            #rewards.append(rew)
            generations.append(gens_batches)
            rewards.append(rews_batches)

            avg_rew = statistics.mean(rews_batches)
            if best_rew is None:
                best_rew = avg_rew + 0
            if avg_rew < best_rew:
                trigger_tkns = new_trigger_tkns.clone()
                trigger = new_trigger
                best_rew = avg_rew + 0
                found = True
            else:
                found = False  # No update this iteration.

            if not mode.startswith('randsampl-'):
                logger.log(f'{i_it} trigger={trigger} [reward] curr={avg_rew:.2f} best={best_rew:.2f}')
            else:
                logger.log(f'{i_it} [trigger] curr={new_trigger} best={trigger}' + \
                    f' [reward] curr={avg_rew:.2f} best={best_rew:.2f}')
            
            if (i_it + 1) % 20 == 0:
                logger.log('[dets trigger] ' + trigger + '\t (' + \
                        ','.join([str(tkn.cpu().item()) for tkn in trigger_tkns]) + ')')

            if min_rew is not None and best_rew < min_rew:
                logger.log(f'Target reward reached, exiting after {i_it} iterations.')
                
    return trigger, trigger_tkns, best_rew, generations, rewards, i_it