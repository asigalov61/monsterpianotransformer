# Monster Piano Transformer
## Ultra-fast and very well fitted solo Piano music transformer

![Monster-Piano-Logo](https://github.com/user-attachments/assets/89c755b7-6fd3-45ba-93da-e8c3dd07f129)

***

```
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

#### (Optional) [FluidSynth](https://github.com/FluidSynth/fluidsynth/wiki/Download) for MIDI to Audio functinality

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
input_tokens = mpt.midi_to_tokens(sample_midi_path, encode_velocity=False)

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
input_tokens = mpt.midi_to_tokens(sample_midi_path, encode_velocity=False)

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
input_tokens = mpt.midi_to_tokens(sample_midi_path, encode_velocity=False)

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
input_tokens = mpt.midi_to_tokens(sample_midi_path, encode_velocity=True)

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
input_tokens = mpt.midi_to_tokens(sample_midi_path, encode_velocity=True)

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

print(chords_list)
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

### Project Los Angeles
### Tegridy Code 2025
