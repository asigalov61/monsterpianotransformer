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

def inpaint_pitches(model,
                    input_tokens,
                    num_pitches_to_inpaint=600,
                    num_prime_pitches=64,
                    keep_high_pitches=False,
                    temperature=0.9,
                    top_k_value=0,
                    verbose=False
                    ):

    #==================================================================

    device = next(model.parameters()).device.type

    #==================================================================

    if verbose:
        print('=' * 70)
        print('Inpainting pitches...')

    comp_total_pitches = len([p for p in input_tokens if 256 < p < 384])

    num_prime_pitches = max(0, min(comp_total_pitches, num_prime_pitches))
    num_pitches_to_inpaint = max(1, min(comp_total_pitches, num_pitches_to_inpaint))

    inputs_list = []
    inp_lst = []

    for t in input_tokens:
        if t < 128:
            if inp_lst:
                inputs_list.append(inp_lst)

            inp_lst = [t]

        else:
            inp_lst.append(t)
            
    if inp_lst:
        inputs_list.append(inp_lst)

    #==================================================================

    inputs = []
    pcount = 0

    if num_prime_pitches > 0:
        
        for il_idx, lst in enumerate(inputs_list):
            
            for t in lst:
                
                inputs.append(t)
                
                if 256 < t < 384:
                    pcount += 1
    
                if pcount == num_prime_pitches:
                    break
                    
            if pcount == num_prime_pitches:
                il_idx += 1
                break

    #==================================================================
   
    while pcount < num_pitches_to_inpaint or pcount < comp_total_pitches:

        fp = True

        for t in inputs_list[il_idx]:

            if t < 256 or t > 384:
                inputs.append(t)

            else:

                if keep_high_pitches and fp:
                        inputs.append(t)
                        fp = False
                        pcount += 1
                    
                else:

                    y = 0
    
                    while y < 256 or y > 384:
    
                        x = torch.LongTensor(inputs).to(device)
        
                        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        
                            if top_k_value > 0:
                    
                                out = model.generate(x,
                                                     1,
                                                     temperature=temperature,
                                                     filter_logits_fn=top_k,
                                                     filter_kwargs={'k': top_k_value},
                                                     return_prime=False,
                                                     verbose=False
                                                    )
                                
                            else:
                                
                                out = model.generate(x,
                                                     1,
                                                     temperature=temperature,
                                                     return_prime=False,
                                                     verbose=False
                                                    )
                                
                        y = out.tolist()[0][0]
    
                    inputs.append(y)
                    pcount += 1

        il_idx += 1
        
    #==================================================================

    if verbose:
        print('=' * 70)
        print('Done!')
        print('=' * 70)

    return inputs

#===================================================================================================

def inpaint_velocities_simple(model,
                              input_tokens,
                              num_notes_to_inpaint=600,
                              num_prime_notes=8,
                              num_memory_tokens=1024,
                              temperature=1.3,
                              verbose=False
                             ):

    if verbose:
        print('=' * 70)
        print('Inpainting velocities...')
        
    #=======================================================

    device = next(model.parameters()).device.type

    #=======================================================

    num_notes_to_inpaint = max(1, num_notes_to_inpaint)
    num_prime_notes = max(0, min(2040, num_prime_notes))
    num_memory_tokens = max(8, min(2040, num_memory_tokens))
    
    #=======================================================

    nv_score_list = []
    nv_score = []
    nv_sc = []
    
    for t in input_tokens:
        if t < 128:
            if nv_score:
                nv_score_list.append(nv_score)
                
            nv_score = [[t]]
    
        else:
            if t < 384:
                nv_sc.append(t)
    
            else:
                if nv_sc:
                    nv_sc.append(t)
                    nv_score.append(nv_sc)
    
                nv_sc = []

    #=======================================================

    inputs = []

    if not [t for t in input_tokens if t > 384]:
        num_prime_notes = 0    
    
    for t in nv_score_list[:num_prime_notes]:
        inputs.extend(t[0])
    
        for tt in t[1:]:
            inputs.extend(tt)

    #=======================================================

    notes_counter = 0
    
    for i in range(num_prime_notes, len(nv_score_list)):

        if notes_counter >= num_notes_to_inpaint:
            break
   
        inputs.extend(nv_score_list[i][0])
    
        for note in nv_score_list[i][1:]:

            if notes_counter >= num_notes_to_inpaint:
                break

            inputs.extend(note[:-1])
            
            x = torch.LongTensor(inputs[-num_memory_tokens:]).cuda()
    
            y = 0
    
            while y < 384:
            
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                    
                    out = model.generate(x,
                                         1,
                                         temperature=temperature,
                                         return_prime=False,
                                         verbose=False)
                
                y = out.tolist()[0][0]
    
            inputs.append(y)

            notes_counter += 1

    #=======================================================
    
    if verbose:
        print('=' * 70)
        print('Done!')
        print('=' * 70)
        
    return inputs

#===================================================================================================

def inpaint_velocities_seq2seq():
    return None

#===================================================================================================
# This is the end of model_loader Python module
#===================================================================================================