#===================================================================================================
# Giant Music Transformer main Python module
#===================================================================================================
# Project Los Angeles
# Tegridy Code 2024
#===================================================================================================
# License: Apache 2.0
#===================================================================================================

from . import model_loader

from . import TMIDIX

import torch

from .x_transformer_1_23_2 import top_p

import random

#===================================================================================================

def generate(model, 
             input_tokens, 
             num_gen_tokens=600,
             num_batches=1,
             temperature=0.9,
             top_p_value=0.96,
             return_prime=False,
             verbose=False
            ):
        
    if verbose:
        print('=' * 70)

    device = next(model.parameters()).device.type
   
    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):

        x = torch.LongTensor([input_tokens] * num_batches).to(device)

        out = model.generate(x,
                             num_gen_tokens,
                             temperature=temperature,
                             filter_logits_fn=top_p,
                             filter_kwargs={'thres': top_p_value},
                             return_prime=return_prime,
                             verbose=verbose
                            )
    
    y = out.tolist()

    
    if verbose:
        print('=' * 70)
        print('Done!')
        print('=' * 70)

    return y
        
#===================================================================================================
# This is the end of model_loader Python module
#===================================================================================================