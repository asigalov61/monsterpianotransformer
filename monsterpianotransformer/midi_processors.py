#===================================================================================================
# Giant Music Transformer midi_processors Python module
#===================================================================================================
# Project Los Angeles
# Tegridy Code 2024
#===================================================================================================
# License: Apache 2.0
#===================================================================================================

from . import TMIDIX

#===================================================================================================

def midi_to_tokens(input_midi, swapped_tokens=False):

    raw_score = TMIDIX.midi2single_track_ms_score(input_midi)
    
    escore_notes = TMIDIX.advanced_score_processor(raw_score, return_enhanced_score_notes=True)
    
    escore_notes = TMIDIX.augment_enhanced_score_notes(escore_notes[0], timings_divider=16)

    instruments_list = list(set([y[6] for y in escore_notes]))

    #=======================================================
    # FINAL PROCESSING
    #=======================================================
    
    melody_chords = []

    # Break between compositions / Intro seq
    
    if 128 in instruments_list:
      drums_present = 19331 # Yes
    else:
      drums_present = 19330 # No
    
    pat = escore_notes[0][6]
    
    melody_chords.extend([19461, drums_present, 19332+pat]) # Intro seq
    
    #=======================================================
    # MAIN PROCESSING CYCLE
    #=======================================================
    
    pe = escore_notes[0]
    
    for e in escore_notes:
    
        #=======================================================
        # Timings...
        
        # Cliping all values...
        delta_time = max(0, min(255, e[1]-pe[1]))
        
        # Durations and channels
        
        dur = max(0, min(255, e[2]))
        cha = max(0, min(15, e[3]))
        
        # Patches
        if cha == 9: # Drums patch will be == 128
          pat = 128
        
        else:
          pat = e[6]
        
        # Pitches
        
        ptc = max(1, min(127, e[4]))
        
        # Velocities
        
        # Calculating octo-velocity
        vel = max(8, min(127, e[5]))
        velocity = round(vel / 15)-1
        
        #=======================================================
        # FINAL NOTE SEQ
        #=======================================================
        
        # Writing final note asynchronously
        
        dur_vel = (8 * dur) + velocity
        pat_ptc = (129 * pat) + ptc

        if swapped_tokens:
            melody_chords.extend([delta_time, pat_ptc+2304, dur_vel+256])

        else:       
            melody_chords.extend([delta_time, dur_vel+256, pat_ptc+2304])
        
        pe = e

    return melody_chords

#===================================================================================================

def tokens_to_midi(tokens, output_midi_name='Giant-Music-Transformer-Composition', return_score=False, swapped_tokens=False):
    
    song = tokens
    song_f = []
    
    time = 0
    dur = 0
    vel = 90
    pitch = 0
    channel = 0
    patch = 0
    
    patches = [-1] * 16
    patches[9] = 9
    
    channels = [0] * 16
    channels[9] = 1
    
    for ss in song:
    
        if 0 <= ss < 256:
    
            time += ss * 16
    
        if 256 <= ss < 2304:
    
            dur = ((ss-256) // 8) * 16
            vel = (((ss-256) % 8)+1) * 15
          
            if swapped_tokens:
                song_f.append(['note', time, dur, channel, pitch, vel, patch])
        
        if 2304 <= ss < 18945:
        
            patch = (ss-2304) // 129
            
            if patch < 128:
        
              if patch not in patches:
                if 0 in channels:
                    cha = channels.index(0)
                    channels[cha] = 1
                else:
                    cha = 15
        
                patches[cha] = patch
                channel = patches.index(patch)
              else:
                channel = patches.index(patch)
        
            elif patch == 128:
                channel = 9
                
            pitch = (ss-2304) % 129
          
            if not swapped_tokens:
                song_f.append(['note', time, dur, channel, pitch, vel, patch])
    
    patches = [0 if x==-1 else x for x in patches]

    detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                                output_signature = 'Giant Music Transformer',
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
