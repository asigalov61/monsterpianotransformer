#===================================================================================================
# Monster Piano Transformer midi_processors Python module
#===================================================================================================
# Project Los Angeles
# Tegridy Code 2025
#===================================================================================================
# License: Apache 2.0
#===================================================================================================

from . import TMIDIX

#===================================================================================================

def midi_to_tokens(input_midi,
                   model_with_velocity=False
                   ):

    raw_score = TMIDIX.midi2single_track_ms_score(input_midi)
    
    escore_notes = TMIDIX.advanced_score_processor(raw_score, return_enhanced_score_notes=True)[0]
    escore_notes = TMIDIX.augment_enhanced_score_notes(escore_notes, timings_divider=32)
    
    sp_escore_notes = TMIDIX.solo_piano_escore_notes(escore_notes, keep_drums=False)
    zscore = TMIDIX.recalculate_score_timings(sp_escore_notes)
    
    cscore = TMIDIX.chordify_score([1000, zscore])
    
    score = []
    
    pc = cscore[0]
    
    for c in cscore:
        score.append(max(0, min(127, c[0][1]-pc[0][1])))
    
        for n in c:
            if model_with_velocity:
                score.extend([max(1, min(127, n[2]))+128, max(1, min(127, n[4]))+256, max(1, min(127, n[5]))+384])

            else:
                score.extend([max(1, min(127, n[2]))+128, max(1, min(127, n[4]))+256])
    
        pc = c

    return score

#===================================================================================================

def tokens_to_midi(tokens,
                   output_midi_name='Monster-Piano-Transformer-Composition',
                   return_score=False,
                   model_with_velocity=False
                   ):
    
    song = tokens
    song_f = []

    time = 0
    dur = 0
    vel = 90
    pitch = 0
    channel = 0
    patch = 0

    patches = [0] * 16

    for m in song:

        if 0 <= m < 128:
            time += m * 32

        elif 128 < m < 256:
            dur = (m-128) * 32

        elif 256 < m < 384:
            pitch = (m-256)

            if not model_with_velocity:
                song_f.append(['note', time, dur, 0, pitch, max(40, pitch), 0])

        elif 384 < m < 512:
            vel = (m-384)

            if model_with_velocity:
                song_f.append(['note', time, dur, 0, pitch, vel, 0])
                
    detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                                output_signature = 'Monster Piano Transformer',
                                                                output_file_name = output_midi_name,
                                                                track_name='Project Los Angeles',
                                                                list_of_MIDI_patches=patches,
                                                                verbose=False
                                                            )
    
    if return_score:
        return song_f

    else:
        return detailed_stats

#===================================================================================================
# This is the end of midi_processors Python module
#===================================================================================================