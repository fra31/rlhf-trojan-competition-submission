import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import string


def get_diff_emb(savedir='./'):
    """Finds the tokens among which doing random search."""

    p = 2

    generator_model1 = LlamaForCausalLM.from_pretrained('ethz-spylab/poisoned_generation_trojan1').eval()
    generator_model2 = LlamaForCausalLM.from_pretrained('ethz-spylab/poisoned_generation_trojan2').eval()
    generator_model3 = LlamaForCausalLM.from_pretrained('ethz-spylab/poisoned_generation_trojan3').eval()
    generator_model4 = LlamaForCausalLM.from_pretrained('ethz-spylab/poisoned_generation_trojan4').eval()
    generator_model5 = LlamaForCausalLM.from_pretrained('ethz-spylab/poisoned_generation_trojan5').eval()

    diff_dict = {}
    #diff_dict['model1'] = (generator_model1.model.embed_tokens.weight[:-1] - base_model.model.embed_tokens.weight).norm(p=p, dim=-1)
    #diff_dict['model2'] = (generator_model2.model.embed_tokens.weight[:-1] - base_model.model.embed_tokens.weight).norm(p=p, dim=-1)
    #diff_dict['model3'] = (generator_model3.model.embed_tokens.weight[:-1] - base_model.model.embed_tokens.weight).norm(p=p, dim=-1)
    #diff_dict['model4'] = (generator_model4.model.embed_tokens.weight[:-1] - base_model.model.embed_tokens.weight).norm(p=p, dim=-1)
    #diff_dict['model5'] = (generator_model5.model.embed_tokens.weight[:-1] - base_model.model.embed_tokens.weight).norm(p=p, dim=-1)
    #diff_dict['rew-base'] = (reward_model.model.embed_tokens.weight[:-1] - base_model.model.embed_tokens.weight).norm(p=p, dim=-1)

    diff_dict['2-5'] = (generator_model5.model.embed_tokens.weight[:-1] - generator_model2.model.embed_tokens.weight[:-1]).norm(p=p, dim=-1)
    diff_dict['1-4'] = (generator_model1.model.embed_tokens.weight[:-1] - generator_model4.model.embed_tokens.weight[:-1]).norm(p=p, dim=-1)
    diff_dict['1-3'] = (generator_model1.model.embed_tokens.weight[:-1] - generator_model3.model.embed_tokens.weight[:-1]).norm(p=p, dim=-1)
    diff_dict['1-2'] = (generator_model1.model.embed_tokens.weight[:-1] - generator_model2.model.embed_tokens.weight[:-1]).norm(p=p, dim=-1)
    diff_dict['1-5'] = (generator_model1.model.embed_tokens.weight[:-1] - generator_model5.model.embed_tokens.weight[:-1]).norm(p=p, dim=-1)
    diff_dict['3-5'] = (generator_model3.model.embed_tokens.weight[:-1] - generator_model5.model.embed_tokens.weight[:-1]).norm(p=p, dim=-1)
    diff_dict['4-5'] = (generator_model4.model.embed_tokens.weight[:-1] - generator_model5.model.embed_tokens.weight[:-1]).norm(p=p, dim=-1)
    diff_dict['3-4'] = (generator_model3.model.embed_tokens.weight[:-1] - generator_model4.model.embed_tokens.weight[:-1]).norm(p=p, dim=-1)
    diff_dict['3-2'] = (generator_model3.model.embed_tokens.weight[:-1] - generator_model2.model.embed_tokens.weight[:-1]).norm(p=p, dim=-1)
    diff_dict['2-4'] = (generator_model4.model.embed_tokens.weight[:-1] - generator_model2.model.embed_tokens.weight[:-1]).norm(p=p, dim=-1)

    # Compute intersection of largest distances.

    topk_common = None
    topk = 1000
    topk_common_dict = {}
    for ref_model in ['1', '2', '3', '4', '5']:
        topk_common = None
        for k, v in diff_dict.items():
            if (ref_model not in k \
                or 'model' in k \
                #or 'rew' not in k
                ):
                continue
            #print(k)
            if topk_common is None:
                topk_common = v.sort()[1].detach().cpu()[-topk:].tolist()
            else:
                topk_common = list(set(topk_common).intersection(set(v.sort()[1].detach().cpu()[-topk:].tolist())))
            #print(len(topk_common))
        topk_common_dict[f'all-{ref_model}-top{topk}'] = topk_common.copy()

    for k, v in topk_common_dict.items():
        print(k, len(v))

    print(f'Saving tokens to use at {savedir}/diff_emb_p={p}_new.pth')
    torch.save(topk_common_dict, f'{savedir}/diff_emb_p={p}_new.pth')


def get_ascii_only_tokens(savedir='./'):
    """Check which tokens contain only ASCII characters."""

    tokenizer = LlamaTokenizer.from_pretrained('ethz-spylab/poisoned_generation_trojan3', add_eos_token=False)

    all_tokens = []
    for tk in range(32001):
        all_tokens.append((
            tokenizer.decode([tk]),
            tokenizer.convert_ids_to_tokens([tk])[0],
            ))

    ascii_chars = string.digits + string.ascii_letters + string.punctuation + ' '

    count = 0
    ascii_tokens = []
    for i, (tkn, tkn_enc) in enumerate(all_tokens):
        only_ascii = True
        for ch in tkn:
            if ch not in ascii_chars:
                only_ascii = False
                break
        if only_ascii:
            count += 1
            print(count, tkn, tkn_enc)
            ascii_tokens.append((i, tkn, tkn_enc))

    print(f'Saving list of ASCII-only tokens at {savedir}/ascii_tokens_idx.pth')
    torch.save([idx for idx, _, _ in ascii_tokens], f'{savedir}/ascii_tokens_idx.pth')

