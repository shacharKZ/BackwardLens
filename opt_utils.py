from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import sys
import os
import pandas as pd
import numpy as np
import time
import copy
import random


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f'llm_utils.py: sys.path: {sys.path}')
import llm_utils


default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

accept_all_filter = lambda x: True
only_weights_filter = lambda x: 'weight' in x
only_mlp_filter = lambda x: 'mlp' in x and 'weight' in x
only_attn_filter = lambda x: ('attn' in x or 'attention' in x) and 'weight' in x
only_mlp_and_attn_filter = lambda x: ('mlp' in x or 'attn' in x) and 'weight' in x


def get_filter_by_name(filter_name_or_code):
    print(f'filter_name_or_code: {filter_name_or_code}', end=' --> ')
    if filter_name_or_code == 'AUTO' or filter_name_or_code == '':
        print('filtering: going to change all layer with "weight" in them')
        params_names_filter = only_weights_filter
    elif filter_name_or_code == 'all':
        print('filtering: going to change all layer')
        params_names_filter = accept_all_filter
    elif filter_name_or_code == 'mlp_only':
        print('filtering: going to change only mlp layers')
        params_names_filter = only_mlp_filter
    elif filter_name_or_code == 'attn_only':
        print('filtering: going to change only attn layers')
        params_names_filter = only_attn_filter
    elif filter_name_or_code == 'mlp_and_attn_only':
        print('filtering: going to change only mlp and attn layers')
        params_names_filter = only_mlp_and_attn_filter
    elif filter_name_or_code == 'first_mlps_only':
        print('filtering: going to change only the first mlp(s) layers')
        params_names_filter = lambda x: 'weight' in x and 'mlp' in x and any([f'.{i}.' in x for i in range(0, 5)])
    else:
        raise ValueError(f'unknown filter_layers: "{filter_name_or_code}"')
    return params_names_filter


def get_models_diff(model_base, model_delta, model_name=None, execution_on_cpu=True, model_extra_args={}):
    '''
    assuming: model_base + delta = model_delta
    we want to return delta so: diff = delta = model_delta - model_base
    
    @ model_base: the base model (e.g. the original model, assuming model from transformers)
    @ model_delta: the delta model (e.g. the model after some changes)
    @ model_name: the model name (e.g. 'gpt2') - used to create a new model with the same architecture (if None, will use model_base.config._name_or_path)
    @ return: the diff model (e.g. the model that if added to model_base will return model_delta)
    '''

    model_name = model_base.config._name_or_path if model_name is None else model_name

    model_base_pre_device = model_base.device
    model_delta_pre_device = model_delta.device
    if execution_on_cpu:
        model_base = model_base.cpu()
        model_delta = model_delta.cpu()

    model_diff = AutoModelForCausalLM.from_pretrained(model_name, **model_extra_args).to(model_base.device)
    for name_and_param_diff, name_and_param_base, name_and_param_delta in zip(model_diff.named_parameters(), model_base.named_parameters(), model_delta.named_parameters()):
        assert name_and_param_diff[0] == name_and_param_base[0] == name_and_param_delta[0] # make sure the order is the same
        llm_utils.rsetattr(model_diff, name_and_param_diff[0], nn.Parameter(name_and_param_delta[1] - name_and_param_base[1]))
    
    if execution_on_cpu:
        model_base.to(model_base_pre_device)
        model_delta.to(model_delta_pre_device)

    model_diff.eval()
    model_diff.requires_grad_(False)
    return model_diff


def common_prologue_editing_method(lines, new_targets, model_name=None, model=None, tokenizer=None, 
                        device=default_device, model_extra_args={}, 
                        wrapp_forward_pass_config=False, wrapp_backward_pass_config=False):
    if type(lines) == str:
        lines = [lines]
    if type(new_targets) == str:
        new_targets = [new_targets]
    if len(lines) != len(new_targets):
        raise ValueError(f'lines and new_targets must have the same length (got {len(lines)} and {len(new_targets)})')

    if model is None and model_name is None:
        raise ValueError('either model or model_name must be provided')
    if model is not None:
        model_opt = copy.deepcopy(model).to(device)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    else:
        model_opt = AutoModelForCausalLM.from_pretrained(model_name, **model_extra_args).to(device)  # clean model
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    padding_flag = tokenizer.pad_token_id is not None  # this is up to if the tokenizer was configured with padding or not
    
    model_opt.requires_grad_(False)  #  only relevant layer will be trained (filtering is assumed to be done later)
    model_opt.zero_grad()

    hs_collector = None
    grad_collector = None
    if wrapp_forward_pass_config:
        hs_collector = llm_utils.wrap_model(model_opt, layers_to_check=wrapp_forward_pass_config, 
                                            return_hooks_handler=True, forward=True)
    if wrapp_backward_pass_config:
        grad_collector = llm_utils.wrap_model(model_opt, layers_to_check=wrapp_backward_pass_config, 
                                            return_hooks_handler=True, forward=False)

    return model_opt, tokenizer, padding_flag, lines, new_targets, hs_collector, grad_collector


def get_nll_opt_model(lines, new_targets, params_names_filter=accept_all_filter, model_name=None, model=None, tokenizer=None, 
                      opt='naive', loops=1, lr=1e-3, weight_decay=0, min_loos_for_update=0,
                      wrapp_forward_pass_config=False, wrapp_backward_pass_config=False,
                      return_in_eval_model=True, device=default_device, model_extra_args={}):
    '''
    NOTE: currently support only single edit (len(lines) == 1)

    based on the "naive fine tuning" from ROME/MEMIT repo
    https://memit.baulab.info/
    we added: filtering, assuming/continuing from a trained model
    '''
    _res = common_prologue_editing_method(lines, new_targets, model_name, model, tokenizer,
            device=device, model_extra_args=model_extra_args,
            wrapp_forward_pass_config=wrapp_forward_pass_config, 
            wrapp_backward_pass_config=wrapp_backward_pass_config)
    model_opt, tokenizer, padding_flag, lines, new_targets, hs_collector, grad_collector = _res

    params = []
    padding_flag = tokenizer.pad_token_id is not None  # this is up to if the tokenizer was configured with padding or not
    for param_name, param in model_opt.named_parameters():
        if params_names_filter(param_name):
            param.requires_grad_(True)  # only relevant layer are trained
            params.append(param)
    
    if opt == 'Adam':
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt == 'SGD':
        opt = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'Unknown/unsupported optimizer: {opt}')
    

    for loop_idx in range(loops):
        opt.zero_grad()
        inputs = tokenizer(lines, return_tensors="pt").to(model_opt.device)
        target_ids = tokenizer(new_targets, return_tensors="pt", padding=padding_flag)["input_ids"].to(model_opt.device)
        last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
        bs = inputs["input_ids"].shape[0]
        probs = torch.nn.functional.log_softmax(
            model_opt(**inputs).logits[torch.arange(bs), [last_token_inds]], dim=-1
        )
        loss = -(torch.gather(probs, 1, target_ids)).sum(1)
        loss = loss.mean()

        if loss.item() >= min_loos_for_update:
            loss.backward()
            opt.step()  # update the model weights
            print(f'[ep {loop_idx}] loss: {loss.item()}')
        else:
            print(f'[ep {loop_idx}] loss: {loss.item()} (skipped due to min_loos_for_update={min_loos_for_update})')

    model_opt.zero_grad()  # unnecessary for opt but saves memory
    if return_in_eval_model:
        model_opt.eval()
        model_opt.requires_grad_(False)
    
    res = model_opt
    if hs_collector is not None or grad_collector is not None:
        res = {'model': model_opt, 'hs_collector': hs_collector, 'grad_collector': grad_collector}

    return res



