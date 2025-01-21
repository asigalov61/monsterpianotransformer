#==================================================================================
# https://huggingface.co/spaces/asigalov61/Monster-Piano-Transformer
#==================================================================================

print('=' * 70)
print('Monster Piano Transformer Gradio App')

print('=' * 70)
print('Loading core Monster Piano Transformer modules...')

import os

import time as reqtime
import datetime
from pytz import timezone

print('=' * 70)
print('Loading main Monster Piano Transformer modules...')

os.environ['USE_FLASH_ATTENTION'] = '1'

import torch

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(True)

from huggingface_hub import hf_hub_download

import TMIDIX

from midi_to_colab_audio import midi_to_colab_audio

from x_transformer_1_23_2 import *

import random

print('=' * 70)
print('Loading aux Monster Piano Transformer modules...')

import matplotlib.pyplot as plt

import gradio as gr
import spaces

print('=' * 70)
print('PyTorch version:', torch.__version__)
print('=' * 70)
print('Done!')
print('Enjoy! :)')
print('=' * 70)

#==================================================================================

MODEL_CHECKPOINTS = {
                    'with velocity - 3 epochs': 'Monster_Piano_Transformer_Velocity_Trained_Model_59896_steps_0.9055_loss_0.735_acc.pth',
                    'without velocity - 3 epochs': 'Monster_Piano_Transformer_No_Velocity_Trained_Model_69412_steps_0.8577_loss_0.7442_acc.pth',
                    'without velocity - 7 epochs': 'Monster_Piano_Transformer_No_Velocity_Trained_Model_161960_steps_0.7775_loss_0.7661_acc.pth'
                    }

SOUDFONT_PATH = 'SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2'

NUM_OUT_BATCHES = 12

PREVIEW_LENGTH = 120 # in tokens

#==================================================================================

def load_model(model_selector):

    print('=' * 70)
    print('Instantiating model...')
    
    device_type = 'cuda'
    dtype = 'bfloat16'
    
    ptdtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    SEQ_LEN = 2048

    if model_selector == 'with velocity - 3 epochs':
        PAD_IDX = 512

    else:
        PAD_IDX = 384
    
    model = TransformerWrapper(
            num_tokens = PAD_IDX+1,
            max_seq_len = SEQ_LEN,
            attn_layers = Decoder(dim = 2048,
                                  depth = 4,
                                  heads = 32,
                                  rotary_pos_emb = True,
                                  attn_flash = True
                                  )
    )
    
    model = AutoregressiveWrapper(model, ignore_index=PAD_IDX, pad_value=PAD_IDX)
    
    print('=' * 70)
    print('Loading model checkpoint...')      
    
    model_checkpoint = hf_hub_download(repo_id='asigalov61/Monster-Piano-Transformer', filename=MODEL_CHECKPOINTS[model_selector])
    
    model.load_state_dict(torch.load(model_checkpoint, map_location='cpu', weights_only=True))
    
    model = torch.compile(model, mode='max-autotune')
    
    print('=' * 70)
    print('Done!')
    print('=' * 70)
    print('Model will use', dtype, 'precision...')
    print('=' * 70)

    return [model, ctx]

#==================================================================================

def load_midi(input_midi, model_selector=''):

    raw_score = TMIDIX.midi2single_track_ms_score(input_midi.name)

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
            if model_selector == 'with velocity - 3 epochs':
                score.extend([max(1, min(127, n[2]))+128, max(1, min(127, n[4]))+256, max(1, min(127, n[5]))+384])

            else:
                score.extend([max(1, min(127, n[2]))+128, max(1, min(127, n[4]))+256])
    
        pc = c

    return score

#==================================================================================

def save_midi(tokens, batch_number=None, model_selector=''):

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

            if model_selector == 'without velocity - 3 epochs' or model_selector == 'without velocity - 7 epochs':
                song_f.append(['note', time, dur, 0, pitch, max(40, pitch), 0])

        elif 384 < m < 512:
            vel = (m-384)

            if model_selector == 'with velocity - 3 epochs':
                song_f.append(['note', time, dur, 0, pitch, vel, 0])

    if batch_number == None:
        fname = 'Monster-Piano-Transformer-Music-Composition'
        
    else:
        fname = 'Monster-Piano-Transformer-Music-Composition_'+str(batch_number)
    
    data = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                  output_signature = 'Monster Piano Transformer',
                                                  output_file_name = fname,
                                                  track_name='Project Los Angeles',
                                                  list_of_MIDI_patches=patches,
                                                  verbose=False
                                                  )

    return song_f

#==================================================================================

@spaces.GPU
def generate_music(prime, 
                   num_gen_tokens,
                   num_mem_tokens,
                   num_gen_batches,
                   model_temperature,
                   # model_sampling_top_p,
                   model_state
                  ):

    if not prime:
        inputs = [0]

    else:
        inputs = prime[-num_mem_tokens:]

    model = model_state[0]
    ctx = model_state[1]
        
    model.cuda()
    model.eval()

    print('Generating...')
    
    inp = [inputs] * num_gen_batches
    
    inp = torch.LongTensor(inp).cuda()
    
    with ctx:
        out = model.generate(inp,
                              num_gen_tokens,
                              #filter_logits_fn=top_p,
                              #filter_kwargs={'thres': model_sampling_top_p},
                              temperature=model_temperature,
                              return_prime=False,
                              verbose=False)
    
    output = out.tolist()

    print('Done!')
    print('=' * 70)
            
    return output
    
#==================================================================================

def generate_callback(input_midi, 
                      num_prime_tokens, 
                      num_gen_tokens,
                      num_mem_tokens,
                      model_temperature,
                      # model_sampling_top_p,
                      final_composition, 
                      generated_batches, 
                      block_lines,
                      model_state
                     ):

    generated_batches = []

    if not final_composition and input_midi is not None:
        final_composition = load_midi(input_midi, model_selector=model_state[2])[:num_prime_tokens]
        midi_score = save_midi(final_composition, model_selector=model_state[2])
        block_lines.append(midi_score[-1][1] / 1000)
        
    batched_gen_tokens = generate_music(final_composition, 
                                        num_gen_tokens,
                                        num_mem_tokens,
                                        NUM_OUT_BATCHES,
                                        model_temperature,
                                        # model_sampling_top_p,
                                        model_state
                                       )
    
    outputs = []
    
    for i in range(len(batched_gen_tokens)):

        tokens = batched_gen_tokens[i]
        
        # Preview
        tokens_preview = final_composition[-PREVIEW_LENGTH:]
        
        # Save MIDI to a temporary file
        midi_score = save_midi(tokens_preview + tokens, i, model_selector=model_state[2])

        # MIDI plot

        if len(final_composition) > PREVIEW_LENGTH:
            midi_plot = TMIDIX.plot_ms_SONG(midi_score, 
                                            plot_title='Batch # ' + str(i),
                                            preview_length_in_notes=int(PREVIEW_LENGTH / 3),
                                            return_plt=True
                                           )

        else:
            midi_plot = TMIDIX.plot_ms_SONG(midi_score, 
                                            plot_title='Batch # ' + str(i), 
                                            return_plt=True
                                           )

        # File name
        fname = 'Monster-Piano-Transformer-Music-Composition_'+str(i)
        
        # Save audio to a temporary file
        midi_audio = midi_to_colab_audio(fname + '.mid', 
                                        soundfont_path=SOUDFONT_PATH,
                                        sample_rate=16000,
                                        output_for_gradio=True
                                        )

        outputs.append([(16000, midi_audio), midi_plot, tokens])
        
    return outputs, final_composition, generated_batches, block_lines

#==================================================================================

def generate_callback_wrapper(input_midi, 
                              num_prime_tokens, 
                              num_gen_tokens,
                              num_mem_tokens,
                              model_temperature,
                              # model_sampling_top_p,
                              final_composition, 
                              generated_batches, 
                              block_lines,
                              model_selector,
                              model_state
                             ):

    print('=' * 70)
    print('Req start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    start_time = reqtime.time()
    
    print('=' * 70)
    if input_midi is not None:
            fn = os.path.basename(input_midi.name)
            fn1 = fn.split('.')[0]
            print('Input file name:', fn)

    print('Selected model type:', model_selector)

    if not model_state:
        model_state = load_model(model_selector)
        model_state.append(model_selector)

    else:
        if model_selector != model_state[2]:
            print('=' * 70)
            print('Switching model...')
            model_state = load_model(model_selector)
            model_state.append(model_selector)
            print('=' * 70)
    
    print('Num prime tokens:', num_prime_tokens)
    print('Num gen tokens:', num_gen_tokens)
    print('Num mem tokens:', num_mem_tokens)

    print('Model temp:', model_temperature)
    # print('Model top_p:', model_sampling_top_p)
    print('=' * 70)
    
    result = generate_callback(input_midi, 
                                num_prime_tokens, 
                                num_gen_tokens,
                                num_mem_tokens,
                                model_temperature,
                                # model_sampling_top_p,
                                final_composition,
                                generated_batches,
                                block_lines,
                                model_state
                             )
    
    generated_batches = [sublist[-1] for sublist in result[0]]

    print('=' * 70)
    print('Req end time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    print('=' * 70)
    print('Req execution time:', (reqtime.time() - start_time), 'sec')
    print('*' * 70)    
    
    return tuple([result[1], generated_batches, result[3]] + [item for sublist in result[0] for item in sublist[:-1]] + [model_state])

#==================================================================================

def add_batch(batch_number, final_composition, generated_batches, block_lines, model_state=[]):

    if generated_batches:
        final_composition.extend(generated_batches[batch_number])

        # Save MIDI to a temporary file
        midi_score = save_midi(final_composition, model_selector=model_state[2])
    
        block_lines.append(midi_score[-1][1] / 1000)

        # MIDI plot
        midi_plot = TMIDIX.plot_ms_SONG(midi_score, 
                                        plot_title='Monster Piano Transformer Composition',
                                        block_lines_times_list=block_lines[:-1],
                                        return_plt=True)
        
        # File name
        fname = 'Monster-Piano-Transformer-Music-Composition'
        
        # Save audio to a temporary file
        midi_audio = midi_to_colab_audio(fname + '.mid', 
                                        soundfont_path=SOUDFONT_PATH,
                                        sample_rate=16000,
                                        output_for_gradio=True
                                        )
    
        print('Added batch #', batch_number)
        print('=' * 70)

        return (16000, midi_audio), midi_plot, fname+'.mid', final_composition, generated_batches, block_lines

    else:
        return None, None, None, [], [], []

#==================================================================================

def remove_batch(batch_number, num_tokens, final_composition, generated_batches, block_lines, model_state=[]):

    if final_composition:

        if len(final_composition) > num_tokens:
            final_composition = final_composition[:-num_tokens]
            block_lines.pop()
    
        # Save MIDI to a temporary file
        midi_score = save_midi(final_composition, model_selector=model_state[2])
    
        # MIDI plot
        midi_plot = TMIDIX.plot_ms_SONG(midi_score, 
                                        plot_title='Monster Piano Transformer Composition',
                                        block_lines_times_list=block_lines[:-1],
                                        return_plt=True)
    
        # File name
        fname = 'Monster-Piano-Transformer-Music-Composition'
        
        # Save audio to a temporary file
        midi_audio = midi_to_colab_audio(fname + '.mid', 
                                        soundfont_path=SOUDFONT_PATH,
                                        sample_rate=16000,
                                        output_for_gradio=True
                                        )
        
        print('Removed batch #', batch_number)
        print('=' * 70)
        
        return (16000, midi_audio), midi_plot, fname+'.mid', final_composition, generated_batches, block_lines

    else:
        return None, None, None, [], [], []

#==================================================================================

def reset(final_composition=[], generated_batches=[], block_lines=[], model_state=[]):
    
    final_composition = []
    generated_batches = []
    block_lines = []
    model_state = []

    return final_composition, generated_batches, block_lines
    
#==================================================================================

def reset_demo(final_composition=[], generated_batches=[], block_lines=[], model_state=[]):
    
    final_composition = []
    generated_batches = []
    block_lines = []
    model_state = []

#==================================================================================

PDT = timezone('US/Pacific')

print('=' * 70)
print('App start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
print('=' * 70)

#==================================================================================

with gr.Blocks() as demo:

    #==================================================================================

    demo.load(reset_demo)

    #==================================================================================

    gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Monster Piano Transformer</h1>")
    gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Ultra-fast and very well fitted solo Piano music transformer</h1>")
    gr.HTML("""
            Check out <a href="https://github.com/asigalov61/monsterpianotransformer">Monster Piano Transformer</a> on GitHub or on
            
            <p>
                <a href="https://pypi.org/project/monsterpianotransformer/">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/6/64/PyPI_logo.svg" alt="PyPI Project" style="width: 100px; height: auto;">
                </a> or 
                <a href="https://huggingface.co/spaces/asigalov61/Monster-Piano-Transformer?duplicate=true">
                    <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/duplicate-this-space-md.svg" alt="Duplicate in Hugging Face">
                </a>
            </p>
            
            for faster execution and endless generation!
            """)
    
    #==================================================================================

    final_composition = gr.State([])
    generated_batches = gr.State([])
    block_lines = gr.State([])
    model_state = gr.State([])
    
    #==================================================================================
    
    gr.Markdown("## Upload seed MIDI or click 'Generate' button for random output")
    
    input_midi = gr.File(label="Input MIDI", file_types=[".midi", ".mid", ".kar"])
    input_midi.upload(reset, [final_composition, generated_batches, block_lines], 
                            [final_composition, generated_batches, block_lines])
    
    gr.Markdown("## Generate")

    model_selector = gr.Dropdown(["without velocity - 7 epochs",
                                  "without velocity - 3 epochs", 
                                  "with velocity - 3 epochs"
                                ], 
                                label="Select model", 
                               )
    
    num_prime_tokens = gr.Slider(15, 1024, value=1024, step=1, label="Number of prime tokens")
    num_gen_tokens = gr.Slider(15, 1024, value=1024, step=1, label="Number of tokens to generate")
    num_mem_tokens = gr.Slider(15, 2048, value=2048, step=1, label="Number of memory tokens")
    model_temperature = gr.Slider(0.1, 1, value=0.9, step=0.01, label="Model temperature")
    # model_sampling_top_p = gr.Slider(0.1, 1, value=0.96, step=0.01, label="Model sampling top p value")
    
    generate_btn = gr.Button("Generate", variant="primary")

    gr.Markdown("## Select batch")
    
    outputs = [final_composition, generated_batches, block_lines]
    
    for i in range(NUM_OUT_BATCHES):
        with gr.Tab(f"Batch # {i}") as tab:
            
            audio_output = gr.Audio(label=f"Batch # {i} MIDI Audio", format="mp3", elem_id="midi_audio")
            plot_output = gr.Plot(label=f"Batch # {i} MIDI Plot")
            
            outputs.extend([audio_output, plot_output])

    outputs.extend([model_state])

    generate_btn.click(generate_callback_wrapper, 
                       [input_midi, 
                        num_prime_tokens, 
                        num_gen_tokens,
                        num_mem_tokens,
                        model_temperature,
                        # model_sampling_top_p,
                        final_composition,
                        generated_batches,
                        block_lines,
                        model_selector,
                        model_state
                       ], 
                       outputs
                      )
    
    gr.Markdown("## Add/Remove batch")
    
    batch_number = gr.Slider(0, NUM_OUT_BATCHES-1, value=0, step=1, label="Batch number to add/remove")
    
    add_btn = gr.Button("Add batch", variant="primary")
    remove_btn = gr.Button("Remove batch", variant="stop")
    
    final_audio_output = gr.Audio(label="Final MIDI audio", format="mp3", elem_id="midi_audio")
    final_plot_output = gr.Plot(label="Final MIDI plot")
    final_file_output = gr.File(label="Final MIDI file")

    #==================================================================================

    add_btn.click(add_batch, [batch_number, final_composition, generated_batches, block_lines, model_state],
                  [final_audio_output, final_plot_output, final_file_output, final_composition, generated_batches, block_lines])

    #==================================================================================

    remove_btn.click(remove_batch, [batch_number, num_gen_tokens, final_composition, generated_batches, block_lines, model_state], 
                     [final_audio_output, final_plot_output, final_file_output, final_composition, generated_batches, block_lines])

    #==================================================================================

    demo.unload(reset_demo)

#==================================================================================

demo.launch()

#==================================================================================