#===================================================================================================
# Monster Piano Transformer model_loader Python module
#===================================================================================================
# Project Los Angeles
# Tegridy Code 2025
#===================================================================================================
# License: Apache 2.0
#===================================================================================================

import os

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

#===================================================================================================

from huggingface_hub import hf_hub_download

from .models import *

import torch

from .x_transformer_1_23_2 import TransformerWrapper, AutoregressiveWrapper, Decoder

from torchsummary import summary

import matplotlib.pyplot as plt

from sklearn import metrics

#===================================================================================================

def load_model(model_name='without velocity - 7 epochs',
               device='cuda',
               compile_mode='max-autotune',
               verbose=False
               ):
    """
    Load and initialize Giant Music Transformer model with specified parameters.

    Parameters:
    model_name (str): The name of the model to load. Options include 'without velocity - 7 epochs', 'without velocity - 3 epochs', 'with velocity - 3 epochs' and 'velocity inpainting - 3 epochs'. Default and best model is 'without velocity - 7 epochs'.
    device (str): The computing device to use. Options include 'cpu' or 'cuda'. Default is 'cuda'.
    compile_mode (str): The torch.compile mode for the model. Options include 'default', 'reduce-overhead', 'max-autotune'. Default is 'max-autotune'.
    verbose (bool): Whether to print detailed information during the loading process. Default is False.

    Returns:
    model: The initialized Monster Piano Transformer model configured with the specified parameters.

    Example use:
    
    import monsterpianotransformer as mpt
    
    mpt_model = mpt.load_model()
    """

    if verbose:
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
        
        print('=' * 70)
        print('Selected model:', model_name.title(), '/', MODELS_PARAMETERS[model_name]['params'], 'M params')
        print('=' * 70)
        print('Model info:')
        print('-' * 70)
        print(MODELS_INFO[model_name])

        print('=' * 70)
        print('Downloading model...')

    else:
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

    model_data = hf_hub_download(repo_id=MODELS_HF_REPO_LINK,
                                filename=MODELS_FILE_NAMES[model_name]
                                )

    if verbose:
        print('Done!')
        print('=' * 70)
        
        print('Instantiating model...')
    
    mpt_model = TransformerWrapper(num_tokens = MODELS_PARAMETERS[model_name]['pad_idx']+1,
                                   max_seq_len = MODELS_PARAMETERS[model_name]['seq_len'],
                                   attn_layers = Decoder(dim = MODELS_PARAMETERS[model_name]['dim'],
                                                         depth = MODELS_PARAMETERS[model_name]['depth'],
                                                         heads = MODELS_PARAMETERS[model_name]['heads'],
                                                         rotary_pos_emb = MODELS_PARAMETERS[model_name]['rope'],
                                                         attn_flash = True
                                                        )
                                  )
    
    mpt_model = AutoregressiveWrapper(mpt_model,
                                      ignore_index = MODELS_PARAMETERS[model_name]['pad_idx'],
                                      pad_value=MODELS_PARAMETERS[model_name]['pad_idx']
                                     )

    if verbose:
        print('Done!')
        print('=' * 70)
        
        print('Loading model...')
    
    mpt_model.load_state_dict(torch.load(model_data, weights_only=True))

    if verbose:
        print('Done!')
        print('=' * 70)
    
        print('Compiling model...')

    mpt_model = torch.compile(mpt_model, mode=compile_mode)

    if verbose:
        print('Done!')
        print('=' * 70)
    
        print('Activating model...')
    
    mpt_model.to(device)
    mpt_model.eval()  

    if verbose:
        print('Done!')
        print('=' * 70)
        
        summary(mpt_model)
        
        tok_emb = mpt_model.net.token_emb.emb.weight.detach().cpu().tolist()

        cos_sim = metrics.pairwise_distances(
          tok_emb, metric='cosine'
        )
        plt.figure(figsize=(7, 7))
        plt.imshow(cos_sim, cmap="inferno", interpolation="nearest")
        im_ratio = cos_sim.shape[0] / cos_sim.shape[1]
        plt.colorbar(fraction=0.046 * im_ratio, pad=0.04)
        plt.xlabel("Position")
        plt.ylabel("Position")
        plt.tight_layout()
        plt.show()
        plt.close()

    return mpt_model

#===================================================================================================
# This is the end of model_loader Python module
#===================================================================================================