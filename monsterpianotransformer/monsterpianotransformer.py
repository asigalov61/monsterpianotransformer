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
        
        prime = input_tokens

        if len(input_tokens) <= (2048 - num_gen_tokens):
            
            inputs = input_tokens
            
        else:
            inputs = input_tokens[-(2048 - num_gen_tokens):]         
            
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
   
    while pcount < num_pitches_to_inpaint and pcount < comp_total_pitches-1 and il_idx < len(inputs_list):
        
        if verbose:
            if pcount % 25 == 0:
                print(pcount, '/', comp_total_pitches-1)

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
                
    if nv_score:
        nv_score_list.append(nv_score)

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

def inpaint_velocities_seq2seq(model,
                               input_tokens,
                               temperature=1.5,
                               verbose=False
                              ):

    if verbose:
        print('=' * 70)
        print('Inpainting velocities...')
        print('=' * 70)
        
    #=======================================================

    device = next(model.parameters()).device.type

    #=======================================================

    nv_score_list = []
    nv_score = []
    nv_sc = []
    
    for t in [t for t in input_tokens if t < 384]:
        
        if t < 128:
            if nv_score:
                nv_score_list.append(nv_score)
                
            nv_score = [[t]]
    
        else:
            if t < 256:
                nv_sc.append(t)
    
            else:
                if nv_sc:
                    nv_sc.append(t)
                    nv_score.append(nv_sc)
    
                nv_sc = []
                
    if nv_score:
        nv_score_list.append(nv_score)

    nv_score = nv_score_list
    
    #=======================================================

    final_vel_score = []
    
    score_sidx = 0
    score_eidx = 0
    
    chunk_idx = 0
    
    half_chunk_len = 75
    max_inputs_len = 600
    
    pvels = []

    #=======================================================
    
    while score_sidx+score_eidx-half_chunk_len < len(nv_score)-1:

        if verbose:
            print('Inpainting chunk #', chunk_idx+1, '/', len(nv_score) // half_chunk_len)
    
        inputs = [512]
    
        half_notes_counter = 0
        
        for score_eidx, chord in enumerate(nv_score[score_sidx:]):
        
            if len(inputs) >= max_inputs_len:
                break 
            
            inputs.append(chord[0][0])
        
            for note in chord[1:]:
                inputs.extend(note)
    
                if score_eidx < half_chunk_len:
                    half_notes_counter += 1
        
        inputs.append(513)
        
        inputs_len = len(inputs)
    
        #=======================================================
    
        pvels_count = 0
        
        for c, i in enumerate(range(score_sidx, score_sidx+score_eidx+1)):
    
            inputs.append(nv_score[i][0][0])
        
            for note in nv_score[i][1:]:
    
                if c == half_chunk_len:
                    pvels = []
                    pvels_count = 0
        
                inputs.extend(note)
    
                if pvels and pvels_count < len(pvels):
                    inputs.append(pvels[pvels_count])
    
                else:
        
                    x = torch.LongTensor(inputs).cuda()
            
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
        
                    if c >= half_chunk_len:
                        pvels.append(y)
        
                pvels_count += 1
        
        #=======================================================
    
        if score_sidx+score_eidx < len(nv_score)-1:
            vel_score = inputs[inputs_len:][:half_chunk_len+(half_notes_counter * 3)]
    
        else:
            vel_score = inputs[inputs_len:]
    
        final_vel_score.extend(vel_score)
    
        score_sidx += half_chunk_len
    
        chunk_idx += 1

    #=======================================================
    
    if verbose:
        print('=' * 70)
        print('Done!')
        print('=' * 70)
        
    return final_vel_score

#===================================================================================================

def inpaint_timings(model,
                    input_tokens,
                    num_notes_to_inpaint=600,
                    num_prime_notes=4,
                    inpaint_dtimes = True,
                    inpaint_durs = True,
                    temperature=0.1,
                    verbose=False
                   ):

    #==================================================================

    if verbose:
        print('=' * 70)
        print('Inpainting timings...')
        print('=' * 70)

    #==================================================================

    device = next(model.parameters()).device.type

    #==================================================================

    num_notes_to_inpaint = max(1, num_notes_to_inpaint)
    num_prime_notes = max(0, min(2040, num_prime_notes))
    
    #=======================================================

    inp_tokens = [t for t in input_tokens if t < 384]

    #==================================================================

    nv_score_list = []
    nv_score = []
    nv_sc = []
    
    for t in input_tokens:
        if t < 128:
            if nv_score:
                nv_score_list.append(nv_score)
                
            nv_score = [[t]]
    
        else:
            if t < 256:
                nv_sc.append(t)
    
            else:
                if nv_sc:
                    nv_sc.append(t)
                    nv_score.append(nv_sc)
    
                nv_sc = []
                
    if nv_score:
        nv_score_list.append(nv_score)

    #=======================================================

    dtimes = []
    durs = []
    pitches = []
    
    for lst in nv_score_list:
        dtimes.append(lst[0][0])

        for i, l in enumerate(lst[1:]):
            durs.append(l[0])

            if i == 0:
                pitches.append(l[1]+128)

            else:
                pitches.append(l[1])
        
   #=======================================================    

    pitches_chunks = []
    last_pchunk_len = 0
    
    for i in range(0, len(pitches), 250):
    
        pchunk = pitches[i:i+500]
    
        if len(pchunk) < 500:
            last_pchunk_len = len(pchunk)
            pc_mult = ((500 // len(pchunk))+1)
            pchunk *= pc_mult
            pchunk = pchunk[:500]
        
        pitches_chunks.append(pchunk)
    
        if len(pitches) <= 500:
            break
    
    final_seq = []
    
    toks_counter = 0

    notes_counter = 0
    
    for pcidx, pchunk in enumerate(pitches_chunks):
        
        if verbose:
            print('Inpainting pitches chunk', pcidx, '/', len(pitches_chunks)-1)
    
        if pcidx == 0:
            seq = [512] + pchunk + [513]
    
        else:
            seq = [512] + pchunk + [513] + final_seq[-toks_counter:]
    
        if pcidx == 0:
        
            tidx = 0
            didx = 0
            
            for i, p in enumerate(pchunk[:num_prime_notes]):
                
                seq.append(p)
                
                if p > 384:
                    seq.append(dtimes[tidx])
                    tidx += 1
            
                seq.append(durs[didx])
                didx += 1
    
            npn = num_prime_notes
    
        elif 0 < pcidx < len(pitches_chunks)-2:
            npn = 250
    
        elif pcidx == len(pitches_chunks)-1:
            npn = 0
        
        toks_counter = 0
    
        for i, p in enumerate(pchunk[npn:]):
    
            seq.append(p)
            
            if i >= 250 or pcidx == len(pitches_chunks)-1 or npn == 250:
                toks_counter += 1

            notes_counter += 1
        
            if inpaint_dtimes:
        
                if p > 384:
                    x = torch.LongTensor(seq).cuda()
                    
                    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                        out = model.generate(x,
                                             1,
                                             temperature=temperature,
                                             return_prime=False,
                                             verbose=False)
                    
                    y = out.tolist()[0][0]
                
                    seq.append(y)
    
                    if i >= 250 or pcidx == len(pitches_chunks)-1 or npn == 250:
                        toks_counter += 1
        
            else:
                if p > 384:
                    seq.append(dtimes[tidx])
                    tidx += 1
                    
                    if i >= 250 or pcidx == len(pitches_chunks)-1 or npn == 250:
                        toks_counter += 1
        
            if inpaint_durs:
        
                x = torch.LongTensor(seq).cuda()
                
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                    out = model.generate(x,
                                         1,
                                         temperature=temperature,
                                         return_prime=False,
                                         verbose=False)
                
                y = out.tolist()[0][0]
            
                seq.append(y)
                
                if i >= 250 or pcidx == len(pitches_chunks)-1 or npn == 250:
                    toks_counter += 1
        
            else:
                seq.append(durs[didx+npn])
                didx += 1
                
                if i >= 250 or pcidx == len(pitches_chunks)-1 or npn == 250:
                    toks_counter += 1
    
            if pcidx == len(pitches_chunks)-1 and i == last_pchunk_len-1:
                break

            if notes_counter == num_notes_to_inpaint:
                break

        if pcidx == 0:
            final_seq.extend(seq[502:])
    
        elif 0 < pcidx < len(pitches_chunks)-2:
            final_seq.extend(seq[-toks_counter:])
    
        elif pcidx == len(pitches_chunks)-1:
            final_seq.extend(seq[+toks_counter:])
            
        if notes_counter == num_notes_to_inpaint:
            break
            
    #=======================================================
    
    if verbose:
        print('=' * 70)
        print('Done!')
        print('=' * 70)
        
    return final_seq

#===================================================================================================

def inpaint_bridge(model,
                   input_tokens,
                   start_token_idx=0,
                   temperature=0.9,
                   verbose=False
                  ):

    #==================================================================

    device = next(model.parameters()).device.type

    #==================================================================

    start_token_idx = max(0, start_token_idx)
    
    #=======================================================

    if verbose:
        print('=' * 70)
        print('Inpainting bridge...')
        print('=' * 70)

    #==================================================================

    chunk = input_tokens[start_token_idx:start_token_idx+1350]

    if len(chunk) == 1350:

        schunk = chunk[:450]
        mchunk = chunk[425:925]
        echunk = chunk[900:]
    
        seq = [384] + schunk + [385] + echunk + [386] + schunk[-25:]
        
        x = torch.LongTensor(seq).cuda()
        
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            
            out = model.generate(x,
                                 450,
                                 temperature=temperature,
                                 return_prime=False,
                                 verbose=verbose)
        
        y = out.tolist()
    
        output = schunk + y[0] + echunk
        
        if verbose:
            print('=' * 70)
            print('Done!')
            print('=' * 70)
            
    else:
        if verbose:
            print('Bridge inpaiting requires an input_tokens sequence of at least 1350 tokens!')
            print('=' * 70)

        output = []
        
    #=======================================================
    
    return output

#===================================================================================================
# This is the end of model_loader Python module
#===================================================================================================