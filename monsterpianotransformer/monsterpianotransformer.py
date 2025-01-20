#===================================================================================================
# Monster Piano Transformer main Python module
#===================================================================================================
# Project Los Angeles
# Tegridy Code 2025
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
             top_p_value=0.0,
             return_prime=False,
             verbose=False
            ):
        
    if verbose:
        print('=' * 70)

    device = next(model.parameters()).device.type
   
    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        
        num_gen_tokens = max(1, min(2047, num_gen_tokens))

        if len(input_tokens) <= (2048 - num_gen_tokens):
            
            inputs = input_tokens
            prime = []
            
        else:
            inputs = input_tokens[-(2048 - num_gen_tokens):]
            prime = input_tokens          
            
        x = torch.LongTensor([inputs] * num_batches).to(device)
        
        if 0.0 < top_p_value < 1.0:

            out = model.generate(x,
                                 num_gen_tokens,
                                 temperature=temperature,
                                 filter_logits_fn=top_p,
                                 filter_kwargs={'thres': top_p_value},
                                 return_prime=False,
                                 verbose=verbose
                                )
            
        else:
            
            out = model.generate(x,
                                 num_gen_tokens,
                                 temperature=temperature,
                                 return_prime=False,
                                 verbose=verbose
                                )
            
    y = out.tolist()

    outputs = []

    if return_prime:
        for o in y:
            outputs.append(prime + o)

    else:
        outputs = y

    if verbose:
        print('=' * 70)
        print('Done!')
        print('=' * 70)

    return outputs

#===================================================================================================

def generate_long(model,
                  input_tokens,
                  num_gen_tokens=600,
                  num_gen_cycles=5,
                  num_batches=1,
                  temperature=0.9,
                  top_p_value=0.0,
                  return_prime=False,
                  verbose=False
                 ):
        
    if verbose:
        print('=' * 70)
        print('Starting generation...')

    device = next(model.parameters()).device.type
   
    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        
        num_gen_tokens = max(1, min(2047, num_gen_tokens))
        num_mem_tokens = 2048-num_gen_tokens

        prime = input_tokens

        if len(input_tokens) <= num_mem_tokens:
            inputs = input_tokens
            
        else:
            inputs = input_tokens[-num_mem_tokens:]

        outputs = [[]] * num_batches
            
        for i in range(num_gen_cycles):
            
            if verbose:
                print('=' * 70)
                print('Generation cycle #', i)
                print('=' * 70)
            
            if i == 0:
                x = torch.LongTensor([inputs] * num_batches).to(device)

            else:
                x = torch.LongTensor([o[-num_mem_tokens:] for o in outputs]).to(device)
            
            if 0.0 < top_p_value < 1.0:
    
                out = model.generate(x,
                                     num_gen_tokens,
                                     temperature=temperature,
                                     filter_logits_fn=top_p,
                                     filter_kwargs={'thres': top_p_value},
                                     return_prime=False,
                                     verbose=verbose
                                    )
                
            else:
                
                out = model.generate(x,
                                     num_gen_tokens,
                                     temperature=temperature,
                                     return_prime=False,
                                     verbose=verbose
                                    )
                
            y = out.tolist()
        
            if i == 0 and return_prime:
                for j, o in enumerate(y):
                    outputs[j].extend(prime + o)
        
            else:
                for j, o in enumerate(y):
                    outputs[j].extend(o)

    if verbose:
        print('=' * 70)
        print('Done!')
        print('=' * 70)

    return outputs

#===================================================================================================
# This is the end of model_loader Python module
#===================================================================================================