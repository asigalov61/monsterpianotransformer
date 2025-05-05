# Monster Piano Transformer
## Ultra-fast and very well fitted solo Piano music transformer

![Monster-Piano-Logo](https://github.com/user-attachments/assets/89c755b7-6fd3-45ba-93da-e8c3dd07f129)

***

```
Monster Piano by QVQ 72B

In the heart of a grand piano black and blue,  
A fuzzy monster with eyes of yellow hue,  
Its fingers dance upon the ivory keys,  
Weaving melodies that soothe and please.  

Musical notes float like leaves on breeze,  
Harmony fills the air with gentle ease,  
Each key stroke a word in a song unsung,  
A symphony of joy that sets the heart alight, free and light.  

The monster plays with such delight,  
Lost in the rhythm, lost in the light,  
Its fur a blur as it moves with grace,  
A pianist born from a whimsical place.  

Monster Piano, a title it bears,  
A fusion of art and melodic airs,  
Where creativity and music blend,  
In this magical concert that never ends.  

Let the monster's music fill the air,  
And wash away our every care,  
For in its song, we find repose,  
And in its rhythm, our spirits glow.
```

***

## Install

```sh
pip install monsterpianotransformer
```

#### (Optional) [FluidSynth](https://github.com/FluidSynth/fluidsynth/wiki/Download) for MIDI to Audio functionality

##### Ubuntu or Debian

```sh
sudo apt-get install fluidsynth
```

##### Windows (with [Chocolatey](https://github.com/chocolatey/choco))

```sh
choco install fluidsynth
```

***

## Gradio app

```sh
# pip package includes a demo Gradio app without audio output

# Please refer to monsterpianotransformer/gradio/app_full.py
# for a full version with fluidsynth audio output

monsterpianotransformer-gradio
```

***

## Available models

```python
# Import Monster Piano Transformer as mpt
import monsterpianotransformer as mpt

# Print a list of available models
mpt.load_model('models info')
```

***

## Quick-start use example

```python
# Import Monster Piano Transformer as mpt
import monsterpianotransformer as mpt

# Load desired Monster Piano Transformer model
# There are several to choose from...
model = mpt.load_model()

# Get sample seed MIDI path
sample_midi_path = mpt.get_sample_midi_files()[6][1]

# Load seed MIDI
input_tokens = mpt.midi_to_tokens(sample_midi_path)

# Generate seed MIDI continuation
output_tokens = mpt.generate(model, input_tokens, num_gen_tokens=600, return_prime=True)

# Save output batch # 0 to MIDI
mpt.tokens_to_midi(output_tokens[0])
```

***

## Main features use examples

### Long auto-continuation generation

```python
# Import Monster Piano Transformer as mpt
import monsterpianotransformer as mpt

# Load desired Monster Piano Transformer model
# There are several to choose from...
model = mpt.load_model()

# Get sample seed MIDI path
sample_midi_path = mpt.get_sample_midi_files()[6][1]

# Load seed MIDI
input_tokens = mpt.midi_to_tokens(sample_midi_path)

# Generate long seed MIDI auto-continuation
output_tokens = mpt.generate_long(model, input_tokens, return_prime=True)

# Save output batch 0 to MIDI
mpt.tokens_to_midi(output_tokens[0])
```

### Pitches inpainting

```python
# Import Monster Piano Transformer as mpt
import monsterpianotransformer as mpt

# Load desired Monster Piano Transformer model
# There are several to choose from...
model = mpt.load_model()

# Get sample seed MIDI path
sample_midi_path = mpt.get_sample_midi_files()[6][1]

# Load seed MIDI
input_tokens = mpt.midi_to_tokens(sample_midi_path)

# Inpaint pitches
output_tokens = mpt.inpaint_pitches(model, input_tokens)

# Save output to MIDI
mpt.tokens_to_midi(output_tokens)
```

### Simple velocities inpainting

```python
# Import Monster Piano Transformer as mpt
import monsterpianotransformer as mpt

# Load desired Monster Piano Transformer model
# There are several to choose from...
model = mpt.load_model(model_name='with velocity - 3 epochs')

# Get sample seed MIDI path
sample_midi_path = mpt.get_sample_midi_files()[6][1]

# Load seed MIDI
input_tokens = mpt.midi_to_tokens(sample_midi_path)

# Inpaint velocities
output_tokens = mpt.inpaint_velocities_simple(model, input_tokens)

# Save output to MIDI
mpt.tokens_to_midi(output_tokens)
```

### Seq2Seq velocities inpainting

```python
# Import Monster Piano Transformer as mpt
import monsterpianotransformer as mpt

# Load desired Monster Piano Transformer model
# There are several to choose from...
model = mpt.load_model(model_name='velocity inpainting - 2 epochs')

# Get sample seed MIDI path
sample_midi_path = mpt.get_sample_midi_files()[6][1]

# Load seed MIDI
input_tokens = mpt.midi_to_tokens(sample_midi_path)

# Inpaint velocities
output_tokens = mpt.inpaint_velocities_seq2seq(model, input_tokens, verbose=True)

# Save output to MIDI
mpt.tokens_to_midi(output_tokens)
```

### Timings inpainting

```python
# Import Monster Piano Transformer as mpt
import monsterpianotransformer as mpt

# Load desired Monster Piano Transformer model
# There are several to choose from...
model = mpt.load_model('timings inpainting - 2 epochs')

# Get sample seed MIDI path
sample_midi_path = mpt.get_sample_midi_files()[6][1]

# Load seed MIDI
input_tokens = mpt.midi_to_tokens(sample_midi_path)

# Inpaint timings
output_tokens = mpt.inpaint_timings(model, input_tokens)

# Save output to MIDI
mpt.tokens_to_midi(output_tokens)
```

### Bridge inpainting

```python
# Import Monster Piano Transformer as mpt
import monsterpianotransformer as mpt

# Load desired Monster Piano Transformer model
# There are several to choose from...
model = mpt.load_model('bridge inpainting - 2 epochs')

# Get sample seed MIDI path
sample_midi_path = mpt.get_sample_midi_files()[11][1]

# Load seed MIDI
input_tokens = mpt.midi_to_tokens(sample_midi_path)

# Inpaint bridge
output_tokens = mpt.inpaint_bridge(model, input_tokens)

# Save output to MIDI
mpt.tokens_to_midi(output_tokens)
```

### Single chord generation

```python
# Import Monster Piano Transformer as mpt
import monsterpianotransformer as mpt

# Load desired Monster Piano Transformer model
# There are several to choose from...
model = mpt.load_model()

# Generate single chord
output_tokens = mpt.generate_chord(model)
```

### Chords progressions

```python
# Import Monster Piano Transformer as mpt
import monsterpianotransformer as mpt

# Load desired Monster Piano Transformer model
# There are several to choose from...
model = mpt.load_model('chords progressions - 3 epochs')

# Prime chord(s) as a list of lists of semitones and/or pitches
prime_chords = [
                [0],
                [0, 2],
                [0, 2, 4],
                [60],
                [60, 62]
               ]

# Convert chords to chords tokens
chords_tokens = mpt.chords_to_chords_tokens(prime_chords)

# Generate chord progression continuation
output_tokens = mpt.generate(model, chords_tokens, num_gen_tokens=32, return_prime=True)

# Convert output tokens batch # 0 back to the chords list
chords_list = mpt.chords_tokens_to_chords(output_tokens[0])
```

### Chords texturing

```python
# Import Monster Piano Transformer as mpt
import monsterpianotransformer as mpt

# Load desired Monster Piano Transformer model
# There are several to choose from...
model = mpt.load_model('chords texturing - 3 epochs')

# Get sample seed MIDI path
sample_midi_path = mpt.get_sample_midi_files()[6][1]

# Convert MIDI to chords list
chords_list = mpt.midi_to_chords(sample_midi_path)

# Texture chords
output_tokens = mpt.texture_chords(model, chords_list)

# Save output to MIDI
mpt.tokens_to_midi(output_tokens)
```

***

## Advanced use examples

### Chords progressions generation and texturing

#### From custom chords list

```python
# Import Monster Piano Transformer as mpt
import monsterpianotransformer as mpt

# Load desired Monster Piano Transformer model
# There are several to choose from...
cp_model = mpt.load_model('chords progressions - 3 epochs')
tex_model = mpt.load_model('chords texturing - 3 epochs')

# Prime chord(s) as a list of lists of semitones and/or pitches
prime_chords = [
                [0],
                [0, 2],
                [0, 2, 4]
               ]

# Convert chords to chords tokens
chords_tokens = mpt.chords_to_chords_tokens(prime_chords)

# Generate chords progression continuation
cp_tokens = mpt.generate(cp_model, chords_tokens, num_gen_tokens=64, return_prime=True)

# Generate pitches for chords in generated chords progression continuation
output_tokens = mpt.generate_chords_pitches(tex_model, cp_tokens[0])

# Convert output tokens to MIDI
mpt.chords_pitches_to_midi(output_tokens)
```

#### From custom MIDI

```python
# Import Monster Piano Transformer as mpt
import monsterpianotransformer as mpt

# Load desired Monster Piano Transformer model
# There are several to choose from...
cp_model = mpt.load_model('chords progressions - 3 epochs')
tex_model = mpt.load_model('chords texturing - 3 epochs')

# Get sample seed MIDI path
sample_midi_path = mpt.get_sample_midi_files()[7][1]

# Load seed MIDI
chords_tokens = mpt.midi_to_chords(sample_midi_path, return_only_chords=True)

# Generate chords progression continuation
cp_tokens = mpt.generate(cp_model, chords_tokens[:64], num_gen_tokens=64, return_prime=True)

# Generate pitches for chords in generated chords progression continuation
output_tokens = mpt.generate_chords_pitches(tex_model, cp_tokens[0])

# Convert output tokens to MIDI
mpt.chords_pitches_to_midi(output_tokens)
```

#### From custom MIDI with prime chords and prime chords pitches

```python
# Import Monster Piano Transformer as mpt
import monsterpianotransformer as mpt

# Load desired Monster Piano Transformer model
# There are several to choose from...
cp_model = mpt.load_model('chords progressions - 3 epochs')
tex_model = mpt.load_model('chords texturing - 3 epochs')

# Get sample seed MIDI path
sample_midi_path = mpt.get_sample_midi_files()[7][1]

# Load seed MIDI
chords_list = mpt.midi_to_chords(sample_midi_path)

# Number of prime chords
num_prime_chords = 64

# Create prime chords tokens list
prime_chords_tokens = [c[0][0] for c in chords_list[:num_prime_chords]]

# Create prime chords pitches list
prime_chords_pitches = [c[0][1:] for c in chords_list[:num_prime_chords]]

# Generate chords progression continuation
cp_tokens = mpt.generate(cp_model, prime_chords_tokens, num_gen_tokens=128, return_prime=True)

# Generate pitches for chords in generated chords progression continuation
output_tokens = mpt.generate_chords_pitches(tex_model, cp_tokens[0], prime_chords_pitches)

# Convert output tokens to MIDI
mpt.chords_pitches_to_midi(output_tokens, chords_list)
```

#### From custom chords list with chords texturing and timings inpainting

```python
# Import Monster Piano Transformer as mpt
import monsterpianotransformer as mpt

# Load desired Monster Piano Transformer model
# There are several to choose from...
cp_model = mpt.load_model('chords progressions - 3 epochs')
tex_model = mpt.load_model('chords texturing - 3 epochs')
tim_model = mpt.load_model('timings inpainting - 2 epochs')

# Prime chord(s) as a list of lists of semitones and/or pitches
prime_chords = [
                [0],
                [0, 2],
                [0, 2, 4]
               ]

# Convert chords to chords tokens
chords_tokens = mpt.chords_to_chords_tokens(prime_chords)

# Generate chords progression continuation
cp_tokens = mpt.generate(cp_model, chords_tokens, num_gen_tokens=64, return_prime=True)

# Generate pitches for chords in generated chords progression continuation
cptcs_tokens = mpt.generate_chords_pitches(tex_model, cp_tokens[0], return_as_tokens_seq=True)

# Inpaint timings
output_tokens = mpt.inpaint_timings(tim_model, cptcs_tokens)

# Save output to MIDI
mpt.tokens_to_midi(output_tokens)
```

***

## Manual input sequences

### Custom notes list to tokens, chords and pitches

```python
# You can manually create compatible input tokens sequence, chords list and pitches list
# from a simple notes list

# Import Monster Piano Transformer as mpt
import monsterpianotransformer as mpt

# Custom notes list should be in the following format:
# [delta start time (0-127), duration (1-127), MIDI pitch (1-127)), velocity (1-127)]
sample_notes_list = [
    
[0, 70, 84, 84], [0, 70, 72, 72], [0, 70, 72, 115], [0, 70, 67, 67], [0, 70, 64, 64],
[0, 70, 60, 60], [0, 70, 55, 55], [0, 70, 52, 52], [0, 70, 48, 48], [0, 70, 36, 40],
[0, 70, 24, 120], [82, 11, 79, 79], [0, 11, 67, 67], [0, 11, 67, 122], [0, 11, 64, 64],
[0, 11, 52, 52], [0, 11, 28, 116], [11, 23, 84, 84], [0, 23, 72, 72], [0, 23, 72, 115],
[0, 23, 67, 67], [0, 23, 60, 60], [0, 23, 55, 55], [0, 23, 52, 52], [0, 23, 48, 48],
[0, 23, 24, 120], [24, 17, 79, 79], [0, 17, 67, 67], [0, 17, 67, 122], [0, 17, 64, 64],
[0, 17, 60, 60], [0, 17, 55, 55], [0, 17, 52, 52], [0, 17, 48, 48], [0, 17, 24, 120],
[17, 5, 81, 81], [0, 5, 69, 69], [0, 5, 69, 124], [0, 5, 65, 65], [0, 5, 53, 53], [0, 5, 29, 115],
[6, 23, 83, 83], [0, 23, 71, 71], [0, 23, 71, 126], [0, 23, 67, 67], [0, 23, 59, 59],
[0, 23, 55, 55], [0, 23, 50, 50], [0, 23, 47, 47], [0, 23, 43, 43], [0, 23, 31, 113]

]

# Use notes_list_to_tokens_chords_pitches function to convert the notes list
output = mpt.notes_list_to_tokens_chords_pitches(sample_notes_list)

input_tokens = output[0]
chords_tokens = output[1]
pitches_list = output[2]
chords_list = output[3]
```

***

## Dev and tests

### Loading

```python
# You can load and use one or several models at the time

# Default model (without velocity - 3 epochs)
default_model = mpt.load_model()

# More models...
cp_model = mpt.load_model('chords progressions - 3 epochs')
tex_model = mpt.load_model('chords texturing - 3 epochs')
tim_model = mpt.load_model('timings inpainting - 2 epochs')
```

### Parameters

```python
# Dev models parameters can be accessed like so

# Max sequence length
default_model.max_seq_len

# Max number of tokens
default_model.pad_value
```

### Generation

```python
# Use generate or generate long functions for dev or testing with all models

# Just make sure to prime the models with at least one token within its tokens range
default_output = mpt.generate(default_model, input_tokens=[0], num_gen_tokens=32)
tex_output = mpt.generate_long(tex_model, input_tokens=[0], num_gen_tokens=32)
```

### Project Los Angeles
### Tegridy Code 2025
