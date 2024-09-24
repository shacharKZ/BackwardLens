import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f'llm_utils.py: sys.path: {sys.path}')
import hook_collect_hidden_states
from flow_graph_configs.llm_configs import auto_model_to_config, GraphConfigs

import functools
import pandas as pd
import copy
import json

import transformers
import torch

device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

# a safe way to get attribute of an object
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

# a safe way to set attribute of an object
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# a safe way to check if an object has an attribute (a deeper version of hasattr)
def hasattr_depper(obj, attrs):
    try:
        rgetattr(obj, attrs)
        return True
    except:
        return False

def wrap_model(model,  
               layers_to_check,
               max_len=64,
               return_hooks_handler=True,
               forward=True):
    '''
    a wrapper function for model to collect hidden states
    returns a dictionary that is updated during the forward/backward pass of the model
    and contains the hidden states of the layers specified in layers_to_check for each layer (collcting inputs and outputs of each)
    the dictionary has the following structure:
    {
        layer_idx: {
            layer_type: {
                'input': [list of hidden states (torch.tensor)],
                'output': [list of hidden states (torch.tensor)]
            }
        }
    }
    you can easily access the hidden states of a specific layer by using the following code:
    hs_collector[layer_idx][layer_type]['input'/'outputs'] # list of hidden states of the input of the layer
    to get the hidden state for the last forward pass, you can use:
    hs_collector[layer_idx][layer_type]['input'/'outputs'][-1] # the last hidden state of the input of the layer

    @ model: a pytorch model (currently only support the models which are supported by flow_graph_configs.llm_configs)
    @ layers_to_check: a list of strings that specify the layers to collect hidden states from.
            if 'AUTO' is passed, the function will try to infer the layers from the model's flow_graph_configs.llm_configs 
    @ max_len: the maximum length of the list. if the list is longer than max_len, the oldest hs will be removed
    @ return_hooks_handler: whether to return the hooks handler (used to remove the hooks later)
    @ forward: whether to collect the hidden states of the forward pass or the backward pass
    '''
    
    hs_collector = {}
    config = None
    if type(layers_to_check) == str:
        config = auto_model_to_config(model) if layers_to_check == 'AUTO' else GraphConfigs.from_json(layers_to_check) 
        layers_to_check = set()
        for cell in ["layer_format", "layer_mlp_format", "layer_attn_format",
                        "ln1", "attn_q", "attn_k", "attn_v", "attn_o",
                        "ln2", "mlp_ff1", "mlp_act", "mlp_ff2"]:
                curr_cel = config.__dict__[cell]
                if curr_cel is not None and curr_cel != '':
                    layers_to_check.add(curr_cel)
    elif type(layers_to_check) == list:
        layers_to_check = layers_to_check.copy()
    else:
        raise Exception('layers_to_check should be either a string or a list. Please check the input and execute the function again.')

    
    if config is not None and type(config) != str and hasattr(config, 'n_layer'):
        n_layer = rgetattr(model.config, config.n_layer)                         
    elif hasattr(model.config, 'n_layer'):  # gpt2, gpt-j
        n_layer = model.config.n_layer
    elif hasattr(model.config, 'num_layers'):  # gpt-neo
        n_layer = model.config.num_layers
    else:
        n_layer = model.config.num_hidden_layers  # llama2, opt
    

    for layer_idx in range(n_layer):
        hs_collector[layer_idx] = {}
        for layer_type in layers_to_check:
            # the layer_key is key to access the layer in the hs_collector dictionary
            if type(layer_type) == list:
                layer_key, layer_type = layer_type
            else:
                layer_key = layer_type
            hs_collector[layer_idx][layer_key] = {}

            try:
                layer_with_idx = layer_type.format(layer_idx)
                # print(f'layer_with_idx: {layer_with_idx}, layer_type: {layer_type}')  # used for debugging
                layer_pointer = rgetattr(model, layer_with_idx)
            except:
                layer_with_idx = f'{layer_idx}{"." if len(layer_type) else ""}{layer_type}'
                 # "transformer.h" is very common prefix in huggingface models like gpt2 and gpt-j.assuming passing only the suffix of the layer
                layer_pointer = rgetattr(model, f"transformer.h.{layer_with_idx}")

            list_inputs = []
            list_outputs = []
            if forward:
                hooks_handler = layer_pointer.register_forward_hook(
                    hook_collect_hidden_states.extract_hs_include_prefix(
                        list_inputs=list_inputs, 
                        list_outputs=list_outputs, 
                        info=layer_with_idx,
                        max_len=max_len
                        )
                    )
            else:
                hooks_handler = layer_pointer.register_full_backward_hook(
                    hook_collect_hidden_states.extract_hs_include_prefix(
                        list_inputs=list_inputs, 
                        list_outputs=list_outputs, 
                        info=layer_with_idx,
                        max_len=max_len
                        )
                    )

            hs_collector[layer_idx][layer_key]['input'] = list_inputs
            hs_collector[layer_idx][layer_key]['output'] = list_outputs
            if return_hooks_handler:
                hs_collector[layer_idx][layer_key]['hooks_handler'] = hooks_handler

    return hs_collector


def remove_collector_hooks(hs_collector):
    '''
    remove all hooks in hs_collector
    '''
    for layer_idx in hs_collector:
        for layer_type in hs_collector[layer_idx]:
            # print(f'{layer_idx}: layer_type: {layer_type}')
            if 'hooks_handler' not in hs_collector[layer_idx][layer_type]:
                print(f'Warning: no hooks handler for layer {layer_idx} {layer_type}')
            else:
                hooks_handler = hs_collector[layer_idx][layer_type]['hooks_handler']
                hooks_handler.remove()


class model_extra:
    '''
    a class that contains extra functions for language models
    @ model: a pytorch model (currently only support the models which are supported by flow_graph_configs.llm_configs)
    @ model_name: the name of the model (e.g. 'gpt2'. if None, will be inferred from the model)
    @ tokenizer: the tokenizer of the model (if None, will be inferred from the model/model_name)
    '''
    def __init__(self, model, model_name=None, tokenizer=None, device=device):
        if model_name is None:
            model_name = model.config._name_or_path

        self.model_name = model_name
        
        if tokenizer is None:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer
        
        self.device = device

        if hasattr_depper(model, 'transformer.ln_f'):  # gpt2, gpt-j, gpt-neo
            self.ln_f = copy.deepcopy(model.transformer.ln_f).to(self.device).requires_grad_(False)
        elif hasattr_depper(model, 'model.norm'):  # models like Llama2
            self.ln_f = copy.deepcopy(model.model.norm).to(self.device).requires_grad_(False)
        elif hasattr_depper(model, 'model.decoder.project_out'):  # models like OPT (instead of layer norm, it uses a linear layer)
            self.ln_f = copy.deepcopy(model.model.decoder.project_out)
            if self.ln_f is None:
                print(f'Warning: model ln_f (project_out) is None. using torch.nn.Identity instead)')
                self.ln_f = torch.nn.Identity(model.config.hidden_size)
            self.ln_f = self.ln_f.to(self.device).requires_grad_(False)
        elif hasattr_depper(model, 'gpt_neox.final_layer_norm'):  # models like gpt-neox
            self.ln_f = copy.deepcopy(model.gpt_neox.final_layer_norm)
        else:
            raise Exception('cannot find the final layer norm of the model (the model specified might not be supported)')
        
        if hasattr(model, 'lm_head'):
            self.lm_head = copy.deepcopy(model.lm_head).to(self.device).requires_grad_(False)
        elif hasattr(model, 'embed_out'):
            self.lm_head = copy.deepcopy(model.embed_out).to(self.device).requires_grad_(False)
        else:
            raise Exception('cannot find the final layer norm of the model (the model specified might not be supported)')


        self.model_config = copy.deepcopy(model.config)
        self.n_embd = 0
        if hasattr(model.config, 'n_embd'):
            self.n_embd = model.config.n_embd
        elif hasattr(model.config, 'hidden_size'):
            self.n_embd = model.config.hidden_size
        else:
            print('Warning: cannot find the hidden size of the model, set self.n_embd to 0')

        self.n_head = 0
        if hasattr(model.config, 'n_head'):
            self.n_head = model.config.n_head
        elif hasattr(model.config, 'num_attention_heads'):
            self.n_head = model.config.num_attention_heads
        else:
            print('Warning: cannot find the number of heads of the model, set self.n_head to 0')
        
        self.head_size = self.n_embd // self.n_head if (self.n_head > 0 and self.n_embd > 0) else 0
        if self.head_size == 0:
            print('Warning: cannot calculate the head size of the model, set self.head_size to 0')
        
        self.n_layer = 0
        if hasattr(model.config, 'n_layer'):
            self.n_layer = model.config.n_layer
        elif hasattr(model.config, 'num_hidden_layers'):
            self.n_layer = model.config.num_hidden_layers
        else:
            print('Warning: cannot find the number of layers of the model, set self.n_layer to 0')
        
        self.config = None
        try:
            self.config = auto_model_to_config(model)
        except:
            print('Warning: cannot find the config graph of the model. set self.config to None')

        # many models merge the attention matrices (Q, K, V) into one matrix. the pad variables are used identify if this merge exists
        self.pad_k = None
        self.pad_v = None
        if self.config is not None:
            self.pad_k = 0
            self.pad_v = 0
            if self.config.attn_k == self.config.attn_q:
                self.pad_k = self.n_embd
            if self.config.attn_v == self.config.attn_q:
                self.pad_v = 2*self.n_embd
            # print(f'pad_k: {self.pad_k}, pad_v: {pad_v}')
        

    def hs_to_probs(self, hs, use_ln_f=True):
        '''
        return the probability of each token given a hidden state

        @ hs: a hidden state (torch.tensor) or a list/dataframe in the length of the model's hidden state
        @ use_ln_f: whether to use the final layer norm of the model (if True, the hs will be normalized before processing by the decoding matrix)
        '''
        if type(hs) != torch.Tensor:
            word_embed = torch.tensor(hs).to(self.device)
        else:
            word_embed = hs.clone().detach().to(self.device)
        if use_ln_f:
            word_embed = self.ln_f(word_embed)
        logit_lens = self.lm_head(word_embed)
        probs = torch.softmax(logit_lens, dim=0).detach()
        return probs
    

    def hs_to_token_top_k(self, hs, k_top=10, k_bottom=10, k=None, use_ln_f=True, return_probs=False):
        '''
        return the top and bottom k tokens given a hidden state according to logit of its projection by the decoding matrix

        @ hs: a hidden state (torch.tensor) or a list/dataframe in the length of the model's hidden state
        @ k_top: the number of top tokens to return
        @ k_bottom: the number of bottom tokens to return
        @ k: if not None, the number of tokens to return (k_top=k_bottom=k)
        @ use_ln_f: whether to use the final layer norm of the model (if True, the hs will be normalized before processing by the decoding matrix)
        @ return_probs: whether to return the probabilities vector fo all tokens
        '''
        probs = self.hs_to_probs(hs, use_ln_f=use_ln_f)
        if k is not None:
            k_top = k_bottom = k

        top_k = probs.topk(k_top)
        top_k_idx = top_k.indices
        # convert the indices to tokens
        top_k_words = [self.tokenizer.decode(i, skip_special_tokens=True) for i in top_k_idx]
        
        top_k = probs.topk(k_bottom, largest=False)
        top_k_idx = top_k.indices
        bottom_k_words = [self.tokenizer.decode(i, skip_special_tokens=True) for i in top_k_idx]
        
        res = {'top_k': top_k_words, 'bottom_k': bottom_k_words}
        if return_probs:
            res['probs'] = probs
        return res
    

    def get_token_rank_from_probs(self, token, probs):
        '''
        return the rank of a token given a probability distribution
        highest rank is 0 (most probable). lowest rank is len(probs)-1 (meaning the token is the least probable)

        @ token: a string of a token
        @ probs: a probability distribution (torch.tensor)
        '''
        if type(token) == str:
            token = self.tokenizer.encode(token, return_tensors='pt')[0]
        return (probs > probs[token]).sum().item()
    

    def infrence(self, model_, line, max_length='auto'):
        '''
        a simple wrapper for the model's generate function
        '''
        if type(max_length) == str and 'auto' in max_length:
            add = 1
            if "+" in max_length:
                add = int(max_length.split('+')[1])
            max_length = len(self.tokenizer.encode(line)) + add

        encoded_line = self.tokenizer.encode(
            line.rstrip(), return_tensors='pt').to(model_.device)

        output = model_.generate(
            input_ids=encoded_line,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id
        )

        answer_ = self.tokenizer.decode(
            output[:, encoded_line.shape[-1]:][0], skip_special_tokens=True)
        return line + answer_


    def infrence_for_grad(self, model_, line):
        '''
        a simple wrapper for the model's forward function
        '''
        encoded_line = self.tokenizer.encode(
            line.rstrip(), return_tensors='pt').to(self.device)

        return model_(encoded_line, output_hidden_states=True, output_attentions=True, use_cache=True)
    

def describe_dict(dict_obj):
    '''
    list the hierarchy of a dictionary (used for hs_collector)
    '''
    tmp = dict_obj.copy()
    try:
        while True:
            print(f'{tmp.keys()}')
            tmp = tmp[list(tmp.keys())[0]]
    except:
        pass


