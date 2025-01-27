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
                   encode_velocity=False,
                   verbose=False
                   ):

    if verbose:
        print('=' * 70)
        print('Encoding MIDI...')

    raw_score = TMIDIX.midi2single_track_ms_score(input_midi)
    
    escore_notes = TMIDIX.advanced_score_processor(raw_score, return_enhanced_score_notes=True)[0]
    escore_notes = TMIDIX.augment_enhanced_score_notes(escore_notes, timings_divider=32)
    
    sp_escore_notes = TMIDIX.solo_piano_escore_notes(escore_notes, keep_drums=False)
    zscore = TMIDIX.recalculate_score_timings(sp_escore_notes)
    
    cscore = TMIDIX.chordify_score([1000, zscore])
    
    score = []
    
    pc = cscore[0]
    
    notes_counter = 0
    
    for i, c in enumerate(cscore):
        score.append(max(0, min(127, c[0][1]-pc[0][1])))
    
        for n in c:
            if encode_velocity:
                score.extend([max(1, min(127, n[2]))+128, max(1, min(127, n[4]))+256, max(1, min(127, n[5]))+384])

            else:
                score.extend([max(1, min(127, n[2]))+128, max(1, min(127, n[4]))+256])
                
            notes_counter += 1
    
        pc = c
        
    if verbose:
        print('Done!')
        print('=' * 70)
        
        print('Source MIDI composition has', len(zscore), 'notes')
        print('Source MIDI composition has', len(cscore), 'chords')
        print('-' * 70)
        print('Encoded sequence has', notes_counter, 'pitches')
        print('Encoded sequence has', i+1, 'chords')
        print('-' * 70)
        print('Final encoded sequence has', len(score), 'tokens')
        print('=' * 70)
        
    return score

#===================================================================================================

def tokens_to_midi(tokens,
                   custom_channel=-1,
                   custom_velocity=-1,
                   custom_patch=-1,
                   output_signature = 'Monster Piano Transformer',
                   track_name='Project Los Angeles',
                   output_midi_name='Monster-Piano-Transformer-Composition',
                   return_ms_score=False,
                   verbose=False
                   ):
    
    if verbose:
        print('=' * 70)
        print('Decoding tokens...')
    
    if [t for t in tokens if 384 < t < 512]:
        model_with_velocity = True
        
    else:
        model_with_velocity = False
    
    song = tokens
    song_f = []

    time = 0
    dur = 8
    vel = 90
    pitch = 60
    channel = 0
    patch = 0

    patches = [0] * 16
    
    if -1 < custom_channel < 16:
        channel = custom_channel
        
    if -1 < custom_patch < 128:
        patch = custom_patch
        patches[channel] = patch

    for m in song:

        if 0 <= m < 128:
            time += m * 32

        elif 128 < m < 256:
            dur = (m-128) * 32

        elif 256 < m < 384:
            pitch = (m-256)
            
            if not model_with_velocity:
                
                if 0 < custom_velocity < 128:
                    vel = custom_velocity
                    
                else:
                    if not model_with_velocity:
                        vel = max(40, pitch)
                
                song_f.append(['note', time, dur, channel, pitch, vel, patch])

        elif 384 < m < 512:
            vel = (m-384)

            if model_with_velocity:
                
                if 0 < custom_velocity < 128:
                    vel = custom_velocity
                    
                song_f.append(['note', time, dur, channel, pitch, vel, patch])
                
    if verbose:
        print('Done!')
        print('=' * 70)
                
    detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                                output_signature=output_signature,
                                                                output_file_name=output_midi_name,
                                                                track_name=track_name,
                                                                list_of_MIDI_patches=patches,
                                                                verbose=verbose
                                                            )
    
    if verbose:
        print('=' * 70)
    
    if return_ms_score:
        return song_f

    else:
        return detailed_stats
    
#===================================================================================================
    
def midi_to_chords(input_midi,
                   return_times=True,
                   return_durs=True,
                   return_vels=True,
                   return_only_chords=False,
                   verbose=False
                  ):
    
    if verbose:
        print('=' * 70)
        print('Encoding MIDI...')
        
    raw_score = TMIDIX.midi2single_track_ms_score(input_midi)
    escore_notes = TMIDIX.advanced_score_processor(raw_score, return_enhanced_score_notes=True)[0]
    escore_notes = TMIDIX.augment_enhanced_score_notes(escore_notes, timings_divider=32)
    
    sp_escore_notes = TMIDIX.solo_piano_escore_notes(escore_notes, keep_drums=False)
    zscore = TMIDIX.recalculate_score_timings(sp_escore_notes)
    
    cscore = TMIDIX.chordify_score([1000, zscore])
    
    times_durs = []
    tones_chords = []
    chords_lens = []

    output = []
    
    pitches_counter = 0
    
    for i, c in enumerate(cscore):
        
        pitches = [e[4] for e in c]
        pitches_counter += len(pitches)
            
        tones_chord = sorted(set([p % 12 for p in pitches]))
             
        if tones_chord in TMIDIX.ALL_CHORDS_SORTED:
            chord_token = TMIDIX.ALL_CHORDS_SORTED.index(tones_chord)
            
        else:
            tones_chord = TMIDIX.check_and_fix_tones_chord(tones_chord)
            chord_token = TMIDIX.ALL_CHORDS_SORTED.index(tones_chord)

        out = [[chord_token] + pitches]

        if return_times:
            out.append([c[0][1]])

        if return_durs:
            durs = [e[2] for e in c]
            out.append([int(sum(durs) / len(durs))])

        if return_vels:
            vels = [e[5] for e in c]
            out.append([int(sum(vels) / len(vels))])

        output.append(out)
        
    if verbose:
        print('Done!')
        print('=' * 70)
        
        print('Source MIDI composition has', len(zscore), 'notes')
        print('Source MIDI composition has', len(cscore), 'chords')
        print('-' * 70)
        print('Final chords list contains total of', pitches_counter, 'pitches')
        print('Final chords list has', len(output), 'chords')
        print('=' * 70)
        
    if return_only_chords:
        return [c[0][0] for c in output]

    else:
        return output

#===================================================================================================

def chords_pitches_to_midi(chords_pitches,
                           midi_chords_list=[[]],
                           chords_dtimes=-1,
                           chords_durs=-1,
                           chords_vels=-1,
                           custom_channel=-1,
                           custom_velocity=-1,
                           custom_patch=-1,
                           output_signature = 'Monster Piano Transformer',
                           track_name='Project Los Angeles',
                           output_midi_name='Monster-Piano-Transformer-Composition',
                           return_ms_score=False,
                           verbose=False
                          ):
    if verbose:
        print('=' * 70)
        print('Decoding tokens...')
        
    cpitches = []

    for cp in chords_pitches:
        cpitches.append([c for c in cp if 0 < c < 128])

    if len(midi_chords_list[0]) == 4:
        song_len = min(len(chords_pitches), len(midi_chords_list))

    else:
        song_len = len(chords_pitches)

    time = 0
    dtime = 8
    dur = 8
    channel = 0
    vel = 90
    patch = 0

    song = []
    
    channel = 0
    patches = [0] * 16
    
    if -1 < custom_channel < 16:
        channel = custom_channel
        
    if -1 < custom_patch < 128:
        patch = custom_patch
        patches[channel] = patch

    for i in range(song_len):

        if chords_dtimes < 1 and len(midi_chords_list[0]) == 4:
            time = midi_chords_list[i][1][0]

        if chords_durs < 1 and len(midi_chords_list[0]) == 4:
            dur = midi_chords_list[i][2][0]

        elif chords_durs > 0:
            dur = chords_durs

        if chords_vels == -1 and len(midi_chords_list[0]) == 4:
            vel = midi_chords_list[i][3][0]

        elif chords_vels > 0:
            vel = chords_vels

        for p in chords_pitches[i]:

            if chords_vels == -1 and len(midi_chords_list[0]) != 4:
                vel = max(40, p)
                
            if 0 < custom_velocity < 128:
                vel = custom_velocity
            
            song.append(['note', time, dur, channel, p, vel, patch])

        if chords_dtimes > 0:
            time += chords_dtimes

        elif chords_dtimes < 1 and len(midi_chords_list[0]) != 4:
            time += dtime
            
    if verbose:
        print('Done!')
        print('=' * 70)
        
    detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song,
                                                              output_signature=output_signature,
                                                              output_file_name=output_midi_name,
                                                              track_name=track_name,
                                                              list_of_MIDI_patches=patches,
                                                              timings_multiplier=32,
                                                              verbose=verbose
                                                             )
    if verbose:
        print('=' * 70)

    if return_ms_score:
        return song_f

    else:
        return detailed_stats
    
#===================================================================================================
# This is the end of midi_processors Python module
#===================================================================================================