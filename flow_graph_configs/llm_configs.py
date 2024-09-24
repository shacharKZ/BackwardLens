import json
from dataclasses import dataclass
import os

@dataclass
class GraphConfigs:
    """
    Simple wrapper to store feature configs model graphs
    """
    layer_format: str  # how to access (list) each layer given HuggingFace model, for example: in gpt2: model.transformer.h.{}
    layer_mlp_format: str  # how to access each mlp layer, for example: in gpt2: model.transformer.h.{}.model
    layer_attn_format: str  # how to access each attention layer, for example: in gpt2: model.transformer.h.{}.attn

    ln1: str  # layer norm of attention sublayer, for example: in gpt2: model.transformer.h.{}.attn.ln_1
    # in case of only one layer norm for both sublayers, use the same string for ln1 and ln2
    attn_q: str  # Q, query, for example: in gpt2: model.transformer.h.{}.attn.c_attn (in gpt2, attn_q,k,v are in the same matrix, and this case: write the same string for all of them)
    attn_k: str  # K, key
    attn_v: str  # V, value
    attn_o: str  # O, output projection

    ln2: str  # layer norm of MLP sublayer, for example: in gpt2: model.transformer.h.{}.attn.ln_2
    mlp_ff1: str  # FF1, first layer of MLP, for example: in gpt2: model.transformer.h.{}.mlp.c_fc
    mlp_ff2: str  # FF2, second layer of MLP, for example: in gpt2: model.transformer.h.{}.mlp.c_proj
    mlp_act: str = ''  # the activation function of the MLP, for example: in gpt2: model.transformer.h.{}.mlp.act

    include_mlp_bias: bool = True
    include_attn_bias: bool = True

    transpose_attn_o: bool = False  # if to transpose the attn_o matrix, for example: in gpt-neo and llama2
    
    n_layer: str = "n_layer"
    config_name: str = "default_v3"

    parallel_attn_mlp_architecture: bool = False

    # final_ln: str = "ln_f"
    # lm_head: str = "lm_head"

    tmp_param_A: int = -1
    tmp_param_B: str = ""
    

    @classmethod
    def from_json(cls, fpath, shell_port=None, relative_path=''):
        '''
        old version of loading configs from json file
        use auto_model_to_config instead
        '''
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
                print(data)
        except:
            print(f'Try to automatically assuming which config to load from {fpath}...')
            if 'gpt2' in fpath:
                fpath = f'{relative_path}flow_graph_configs/config_basic.json'
            # elif 'gpt-j' in fpath or 'gptj' in fpath:
            #     fpath = f'{relative_path}flow_graph_configs/config_gptj.json'
            # elif 'gpt-neo' in fpath or 'gpt_neo' in fpath:
            #     fpath = f'{relative_path}flow_graph_configs/config_gpt_neo.json'
            elif 'llama' in fpath:
                fpath = f'{relative_path}flow_graph_configs/config_llama2_7B.json'
            else:
                raise ValueError(f'Could not automatically determine which config to load from {fpath}')

            print(f'Loading config from {fpath}')
            with open(fpath, "r") as f:
                data = json.load(f)
                print(data)

        return cls(**data)


def auto_model_to_path_config(name_or_model):
    if isinstance(name_or_model, str):
        name = name_or_model
    elif hasattr(name_or_model, 'config'):
        name = name_or_model.config
        if hasattr(name, '_name_or_path'):
            name = name._name_or_path
        elif hasattr(name, 'architectures'):
            name = name.architectures
        elif hasattr(name, 'model_type'):
            name = name.model_type
        else:
            raise Exception(f'could not determine model name from {name_or_model} -> {name}')
    else:
        raise Exception(f'could not determine model name from {name_or_model}')

            
    configs_folder = os.path.dirname(os.path.abspath(__file__))  # assuming this file is in the same folder as the configs
    if 'gpt2' in name:
        config_path = os.path.join(configs_folder, 'config_gpt2.json')
    elif 'llama' in name.lower():
        config_path = os.path.join(configs_folder, 'config_llama2_7B.json')
    # elif 'pythia' in name.lower() or 'gptneox' in name.lower():
    #     config_path = os.path.join(configs_folder, 'config_gptNeoX.json')
    # elif 'gpt-j' in name.lower():
    #     config_path = os.path.join(configs_folder, 'config_gptj.json')
    # elif 'gptneo' in name.lower() or 'gpt-neo' in name.lower():
    #     config_path = os.path.join(configs_folder, 'config_gpt_neo.json')
    # elif 'opt' in name.lower():
    #     config_path = os.path.join(configs_folder, 'config_opt.json')
    else:
        raise Exception(f'model "{name}" not supported. Currently supported models: gpt2 and Llama2-7B')
    return config_path


def auto_model_to_config(name_or_model, annot=True):
    config_path = auto_model_to_path_config(name_or_model)
    if annot:
        print(f'Loading config from {config_path}')
    return GraphConfigs.from_json(config_path)
