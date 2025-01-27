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

from .models import *

import torch

from .x_transformer_1_23_2 import top_p, top_k

import random

import copy

#===================================================================================================

def chords_to_chords_tokens(chords_list, 
                            chords_tokens_shift=0,
                            fix_bad_chords=True,
                            verbose=False
                           ):

    if verbose:
        print('=' * 70)

    c_list = []

    for cho in chords_list:
        
        c = sorted(set(cho))
        
        c_list.append([cc % 12 for cc in c if cc >= 0])

    chords_tokens = []
    
    for i, chord in enumerate(c_list):
    
        if chord in TMIDIX.ALL_CHORDS_SORTED:
            chords_tokens.append(TMIDIX.ALL_CHORDS_SORTED.index(chord))
    
        else:
            if fix_bad_chords:
                fixed_chord = TMIDIX.check_and_fix_tones_chord(chord)
                chords_tokens.append(TMIDIX.ALL_CHORDS_SORTED.index(fixed_chord))
    
            else:
                if verbose:
                    print('Bad chord', c_list[i], 'at index', i)
                    
    if verbose:
        print('Done!')
        print('=' * 70)
        
    return chords_tokens

#===================================================================================================

def chords_tokens_to_chords(chords_tokens, 
                            chords_tokens_shift=0,
                            base_pitch=0,
                            reverse_sort=False,
                            verbose=False
                           ):

    if verbose:
        print('=' * 70)

    c_tokens = [t for t in chords_tokens if 0 <= t-chords_tokens_shift < 321]

    chords = []

    for t in c_tokens:
        
        tones_chord = TMIDIX.ALL_CHORDS_SORTED[t-chords_tokens_shift]

        chords.append(sorted([t+base_pitch for t in tones_chord], reverse=reverse_sort))
        
    if verbose:
        print('Done!')
        print('=' * 70)

    return chords

#===================================================================================================

def notes_list_to_tokens_chords_pitches(notes_list, 
                                        encode_velocity=False,
                                        chords_tokens_shift=0,
                                        verbose=False
                                       ):

    #==============================================

    def resort(lst):

        dur_ptc_vel = [e[1:] for e in lst]
        
        sorted_lst = sorted(dur_ptc_vel, key = lambda x: -x[1])
    
        resorted_lst = []
        
        for i, ele in enumerate(lst):
            resorted_lst.append([ele[0]] + sorted_lst[i])

        return resorted_lst

    #==============================================

    if verbose:
        print('=' * 70)
    
    if type(notes_list) == list:
        if all([True for n in notes_list if type(n) == 'list']):
            if all([True for n in notes_list if len(n) == 4]):
    
                #==============================================
        
                chords = []
                cho = []
                
                for i, note in enumerate(notes_list):
                    if note[0] == 0 and i == 0:
                        cho.append(note)
        
                    if note[0] != 0:
                        if cho:
                            chords.append(resort(cho))
        
                        cho = [note]
                    else:
                        cho.append(note)
        
                if cho:
                    chords.append(resort(cho))
        
                #==============================================
        
                tokens_seq = []
                chords_toks_list = []
                pitches_list = []
                chords_list = []
        
                for i, chord in enumerate(chords):
        
                    dtime = max(0, min(127, chord[0][0] % 128))
                    
                    if dtime == 0 and i == 0:
                        tokens_seq.append(dtime)
        
                    if dtime != 0:
                        tokens_seq.append(dtime)
    
                    seen = []
                    
                    durs = []
                    ptcs = []
                    vels = []
                        
                    for note in chord:
                        
                        dur = max(1, min(127, note[1] % 128))
                        ptc = max(1, min(127, note[2] % 128))
                        vel = max(1, min(127, note[3] % 128))
    
                        if ptc not in seen:
                            tokens_seq.extend([dur+128, ptc+256])
                            
                            durs.append(dur)
                            ptcs.append(ptc)
        
                            if encode_velocity:
                                 tokens_seq.append(vel+384)

                            vels.append(vel)
    
                            seen.append(ptc)

                    #==============================================
    
                    pitches = sorted(set(ptcs), reverse=True)
                    pitches_list.append(pitches)

                    #==============================================
    
                    tones_chord = sorted(set([p % 12 for p in pitches]))
    
                    if tones_chord in TMIDIX.ALL_CHORDS_SORTED:
                        chord_tok = TMIDIX.ALL_CHORDS_SORTED.index(tones_chord)
    
                    else:
                        tones_chord = TMIDIX.check_and_fix_tones_chord(tones_chord)
                        chord_tok = TMIDIX.ALL_CHORDS_SORTED.index(tones_chord)
    
                    chords_toks_list.append(chord_tok+chords_tokens_shift)

                    #==============================================

                    avg_dur = int(sum(durs) / len(durs))
                    avg_vel = int(sum(vels) / len(vels))
                    
                    chords_list.append([[chord_tok] + pitches, [dtime], [avg_dur], [avg_vel]])
    
                #==============================================
                
                if verbose:
                    print('Done!')
                    print('=' * 70)
                    print('Tokens sequence has', len(tokens_seq), 'tokens')
                    print('Chords list has', len(chords_list), 'chords')
                    print('Pitches list has', len(TMIDIX.flatten(pitches_list)), 'pitches')
                    print('=' * 70)

                #==============================================
                    
                return tokens_seq, chords_toks_list, pitches_list, chords_list

                #==============================================
    
            else:
                if verbose:
                    print('Input sublists do not have correct number of elements (4).')

                return []

            #==============================================
    
        else:
            if verbose:
                print('Input is not a list of lists.')

            return []

        #==============================================

    else:
        if verbose:
            print('Input is not a list.')

        return []

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
    max_seq_len = model.max_seq_len
   
    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        
        num_gen_tokens = max(1, min(max_seq_len-1, num_gen_tokens))
        
        prime = input_tokens

        if len(input_tokens) <= (max_seq_len - num_gen_tokens):
            
            inputs = input_tokens
            
        else:
            inputs = input_tokens[-(max_seq_len - num_gen_tokens):]         
            
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
    max_seq_len = model.max_seq_len
   
    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        
        num_gen_tokens = max(1, min(max_seq_len-1, num_gen_tokens))
        num_mem_tokens = max_seq_len-num_gen_tokens

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
            
            x = torch.LongTensor(inputs[-num_memory_tokens:]).to(device)
    
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
        
                    x = torch.LongTensor(inputs).to(device)
            
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
    
    for t in inp_tokens:
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
                    x = torch.LongTensor(seq).to(device)
                    
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
        
                x = torch.LongTensor(seq).to(device)
                
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
        
    output = [(t % 128)+256 if 256 < t < 512 else t for t in final_seq]
    
    #=======================================================
    
    if verbose:
        print('=' * 70)
        print('Done!')
        print('=' * 70)
        
    return output

#===================================================================================================

def inpaint_bridge(model,
                   input_tokens,
                   start_token_idx=0,
                   inpaint_dtimes=True,
                   inpaint_durs=True,
                   return_parts=False,
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

        bridge_seq = []

        for i, t in enumerate(mchunk[25:-25]):

            if verbose:
                if i % 25 == 0:
                    print('Generated', i, '/', len(mchunk[25:-25]), 'tokens')

            if 0 <= t < 128:
                if inpaint_dtimes:
        
                    x = torch.LongTensor(seq).to(device)
                    
                    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                        
                        out = model.generate(x,
                                             1,
                                             temperature=temperature,
                                             return_prime=False,
                                             verbose=False)
                    
                    y = out.tolist()[0][0]
    
                    bridge_seq.append(y)
                    seq.append(y)

                else:
                    bridge_seq.append(t)
                    seq.append(t)

            elif 128 <= t < 256:
                if inpaint_durs:
        
                    x = torch.LongTensor(seq).to(device)
                    
                    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                        
                        out = model.generate(x,
                                             1,
                                             temperature=temperature,
                                             return_prime=False,
                                             verbose=False)
                    
                    y = out.tolist()[0][0]
    
                    bridge_seq.append(y)
                    seq.append(y)

                else:
                    bridge_seq.append(t)
                    seq.append(t)

            else:
                x = torch.LongTensor(seq).to(device)
                
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                    
                    out = model.generate(x,
                                         1,
                                         temperature=temperature,
                                         return_prime=False,
                                         verbose=False)
                
                y = out.tolist()[0][0]

                bridge_seq.append(y)
                seq.append(y)                
            
        output = schunk + bridge_seq + echunk
        
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

    if return_parts:
        return [schunk, bridge_seq, echunk]

    else:
        return output

#===================================================================================================

def generate_chord(model,
                   input_tokens=[],
                   chord_dtime=-1,
                   chord_dur=-1,
                   chord_vel=-1,
                   chord_tok=-1,
                   chord_tok_tries=320,
                   max_num_pitches=-1,
                   temperature=0.9,
                   top_p_value=0,
                   return_prime=False,
                   verbose=False
                  ):

    #==================================================================

    def trim_inp_seq(inp_seq, inp_seq_type):
        
        iseq = reversed(inp_seq)
    
        for i, t in enumerate(iseq):
            if t > inp_seq_type:
                if i > 0:
                    return inp_seq[:-i]
    
                else:
                    return inp_seq

    #==================================================================

    device = next(model.parameters()).device.type

    #==================================================================

    if verbose:
        print('=' * 70)
        print('Generating chord...')

    #=====================================================================
        
    if detect_model_type(model)[1] == 2:
        model_with_velocity = True
        
    else:
        model_with_velocity = False

    if model_with_velocity:
        inp_seq_type = 384

    else:
        inp_seq_type = 256
        input_tokens = [t for t in input_tokens if t < 384]
    
    if input_tokens:
        
        input_tokens = trim_inp_seq(input_tokens, inp_seq_type)
        inp_toks = input_tokens

        if not inp_toks:
            inp_toks = [max(chord_dtime, 0)]

    else:
        inp_toks = [max(chord_dtime, 0)]

    #=====================================================================

    if inp_toks[-1] > 127:

        x = torch.LongTensor(inp_toks).to(device)
    
        y = 128
        
        while y > 127:
    
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):

                out = model.generate(x,
                                     1,
                                     temperature=temperature,
                                     return_prime=False,
                                     verbose=False
                                    )
                                        
            y = out.tolist()[0][0]
        

        inp_toks.append(y)

    #=====================================================================

    inp_toks_copy = copy.deepcopy(inp_toks)

    ctok = 322
    tries = 0

    while chord_tok != ctok and tries < chord_tok_tries:

        inp_toks = copy.deepcopy(inp_toks_copy)

        pcount = 0
        pitches = []
    
        y = 512
    
        while y > 127:
    
            x = torch.LongTensor(inp_toks).to(device)
    
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                
                if 0 < top_p_value < 1:
                    out = model.generate(x,
                                         1,
                                         filter_logits_fn=top_p,
                                         filter_kwargs={'thres': top_p_value},
                                         temperature=temperature,
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
    
            if y > 127:
                
                if 128 < y < 256 and 0 < chord_dur < 128:
                    inp_toks.append(chord_dur+128)
    
                elif 384 < y < 512 and 0 < chord_vel < 128:
                    inp_toks.append(chord_vel+384)
    
                else:
                    inp_toks.append(y)
    
            if 256 < y < 384:
                pitches.append(y-256)
                
                tones_chord = sorted(set([p % 12 for p in pitches]))
                
                if tones_chord in TMIDIX.ALL_CHORDS_SORTED:
                    ctok = TMIDIX.ALL_CHORDS_SORTED.index(tones_chord)

                else:
                    ctok = 321
                    
            if y > inp_seq_type:
                pcount += 1

            if ctok == chord_tok or ctok == 321:
                break
    
            if max_num_pitches > 0 and pcount == max_num_pitches:
                break
                
        if chord_tok == -1:
            tries = chord_tok_tries
            ctok = -1

        tries += 1

    if tries == chord_tok_tries:
        inp_toks = []
                
    #=====================================================================

    if verbose:
        print('=' * 70)
        print('Done!')
        print('=' * 70)

    #=====================================================================

    if return_prime:
        return inp_toks

    else:

        plen = 0

        if input_tokens:
            plen = len(input_tokens)-1
     
        return inp_toks[plen:]

#===================================================================================================

def generate_chords_pitches(model,
                            input_chords_tokens,
                            prime_chords_pitches=[],
                            num_gen_chords=128,
                            temperature=0.9,
                            top_p_value=0.0,
                            verify_pitches=True,
                            return_chords_tokens=False,
                            return_as_tokens_seq=False,
                            tokens_seq_dtimes=8,
                            tokens_seq_durs=8,
                            tokens_seq_vels=-1,
                            verbose=False
                           ):

    #========================================================================
        
    if verbose:
        print('=' * 70)

    #========================================================================

    device = next(model.parameters()).device.type
    max_seq_len = model.max_seq_len

    #========================================================================
   
    num_gen_chords = max(1, min(num_gen_chords, len(input_chords_tokens)))

    #========================================================================

    if 0 <= min(input_chords_tokens) < 321 and 0 <= max(input_chords_tokens) < 321:
        input_chords_tokens = [t+128 for t in input_chords_tokens]

    if not 128 <= min(input_chords_tokens) < 441 and not 128 <= min(input_chords_tokens) < 441:
        
        if verbose:
            print('Input chords tokens sequence is out of range(128, 441)!')
            print('=' * 70)
            
        return []
    
    #========================================================================
    
    if verbose:
        print('Generating...')

    inputs = []
    outputs = []
    
    sidx = 0
    
    if len(prime_chords_pitches) > 0:
        for i, p in enumerate(prime_chords_pitches[:num_gen_chords]):
            
            inputs.append(input_chords_tokens[i])
            inputs.extend(p)
            
            if return_chords_tokens:
                outputs.append([input_chords_tokens[i]] + p)
            else:
                outputs.append(p)
                
        sidx = i+1

    for i in range(num_gen_chords-sidx):

        if verbose:
            if (i+1) % 8 == 0:
                print('Generated', i+1, '/', num_gen_chords, 'chords')

        inputs = inputs[-(max_seq_len - 32):]

        inputs.append(input_chords_tokens[sidx+i])

        outp = []

        if return_chords_tokens:
            outp.append(input_chords_tokens[sidx+i])

        y = 0
        
        seen = []

        while y < 128:
        
            x = torch.LongTensor(inputs).to(device)
            
            if 0.0 < top_p_value < 1.0:
                
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                    out = model.generate(x,
                                         1,
                                         temperature=temperature,
                                         filter_logits_fn=top_p,
                                         filter_kwargs={'thres': top_p_value},
                                         return_prime=False,
                                         verbose=False
                                        )
                
            else:
                
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                    out = model.generate(x,
                                         1,
                                         temperature=temperature,
                                         return_prime=False,
                                         verbose=False
                                        )
            
            y = out.tolist()[0][0]
            
            if verify_pitches:
                if y < 128 and y not in seen:
                    inputs.append(y)
                    outp.append(y)
                    seen.append(y)
                    
            else:
                if y < 128:
                    inputs.append(y)
                    outp.append(y)

        outputs.append(outp)

    #========================================================================
        
    if return_as_tokens_seq:
        
        if return_chords_tokens:
            sidx = 1
            
        else:
            sidx = 0
            
        tokens_seq = []
            
        for i, p in enumerate(outputs):
            
            if i == 0:
                tokens_seq.append(0)
                
            else:
                tokens_seq.append(tokens_seq_dtimes)
            
            for pp in p[sidx:]:
                tokens_seq.extend([tokens_seq_durs+128, pp+256])
                
                if tokens_seq_vels < 1:
                    tokens_seq.append(max(40, pp)+384)
                    
                else:
                    tokens_seq.append((tokens_seq_vels % 128)+384)
                    
    #========================================================================

    if verbose:
        print('=' * 70)
        print('Done!')
        print('=' * 70)

    #========================================================================
    
    if return_as_tokens_seq:
        return tokens_seq
    
    else:
        return outputs

#===================================================================================================

def texture_chords(model,
                   midi_chords_list,
                   num_prime_chords=8,
                   num_tex_chords=256,
                   num_memory_tokens=2040,
                   temperature=1.0,
                   top_p_value=0.9,
                   match_pitches_counts=False,
                   keep_high_pitch=True,
                   verify_pitches=True,
                   output_velocities=True,
                   verbose=False
                  ):
    
    #========================================================================
        
    if verbose:
        print('=' * 70)
        print('Texturing chords...')

    #========================================================================

    device = next(model.parameters()).device.type

    #========================================================================
    
    num_memory_tokens = max(1, min(2047, num_memory_tokens))
    num_prime_chords = max(0, min(len(midi_chords_list)-2, num_prime_chords))
    num_tex_chords = max(1, num_tex_chords)

    #========================================================================

    if len(midi_chords_list[0]) != 4:
        
        if verbose:
            print('Chords texturing requires MIDI chords list with timings, durations and velocities!')
            
        return []
        
    #========================================================================
    
    inputs = []
    outputs = []
    
    #========================================================================
    
    pt = midi_chords_list[0][1][0]
    
    for c in midi_chords_list[:num_prime_chords]:
        
        inputs.append(c[0][0]+128)
        
        dtime = c[1][0]-pt

        outputs.append(dtime)
        
        dur = c[2][0]
        vel = c[3][0]
        
        for ptc in c[0][1:]:
        
            inputs.append(ptc)
            outputs.extend([dur+128, ptc+256])

            if output_velocities:
                outputs.append(vel+384)

        pt = c[1][0]
        
    #========================================================================

    verbosity_value = (len(midi_chords_list) // 100) + 4

    for i in range(len(midi_chords_list[num_prime_chords:num_prime_chords+num_tex_chords])):

        if verbose:
            if (i+1) % verbosity_value == 0:
                print('Textured', i+1, '/', len(midi_chords_list[num_prime_chords:num_prime_chords+num_tex_chords]), 'chords')
                
        idx = num_prime_chords + i

        cho_tok = midi_chords_list[idx][0][0]
    
        inputs.append(cho_tok+128)

        tones_chord = TMIDIX.ALL_CHORDS_SORTED[cho_tok]

        dtime = midi_chords_list[idx][1][0]-pt

        outputs.append(dtime)

        pt = midi_chords_list[idx][1][0]
    
        y = 0
        count = 0
        pcount = False
        tries = 0
        
        chord_len = len(midi_chords_list[idx][0][1:])

        dur = midi_chords_list[idx][2][0]
        vel = midi_chords_list[idx][3][0]

        seen = []

        if keep_high_pitch:
            inputs.append(midi_chords_list[idx][0][1])
            seen.append(midi_chords_list[idx][0][1])
            
            outputs.extend([dur+128, midi_chords_list[idx][0][1]+256])

            if output_velocities:
                outputs.append(vel+384)
                
            count += 1

            if chord_len == 1:
                pcount = True
    
        while y < 128 and not pcount and tries < 50:

            inputs = inputs[-num_memory_tokens:]
    
            x = torch.LongTensor(inputs).cuda()
            
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                
                if 0 < top_p_value < 1:
                    
                    out = model.generate(x,
                                         1,
                                         temperature=temperature,
                                         filter_logits_fn=top_p,
                                         filter_kwargs={'thres': top_p_value},
                                         return_prime=False,
                                         verbose=False)
                    
                else:
                    out = model.generate(x,
                                         1,
                                         temperature=temperature,
                                         return_prime=False,
                                         verbose=False)
            
            y = out.tolist()[0][0]

            if verify_pitches:
    
                if y < 128 and y % 12 in tones_chord and y not in seen:
                    inputs.append(y)
                    seen.append(y)
                    
                    outputs.extend([dur+128, y+256])
    
                    if output_velocities:
                        outputs.append(vel+384)
                        
                    count += 1
                    
            else:    
                if y < 128 and y not in seen:
                    inputs.append(y)
                    seen.append(y)
                    
                    outputs.extend([dur+128, y+256])
    
                    if output_velocities:
                        outputs.append(vel+384)
                        
                    count += 1
                    
            if match_pitches_counts:
                if count == chord_len:
                    pcount = True

            else:
                if y > 127:
                    pcount = True

            tries += 1

    #========================================================================
                
    if verbose:
        print('=' * 70)
        print('Done!')
        print('=' * 70)

    #========================================================================
    
    return outputs

#===================================================================================================
# This is the end of model_loader Python module
#===================================================================================================