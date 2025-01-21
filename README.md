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
input_tokens = mpt.midi_to_tokens(sample_midi_path, model_with_velocity=False)

# Generate seed MIDI continuation
output_tokens = mpt.generate(model, input_tokens, 600, return_prime=True)

# Save output batch 0 to MIDI
mpt.tokens_to_midi(output_tokens[0], model_with_velocity=False)
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
input_tokens = mpt.midi_to_tokens(sample_midi_path, model_with_velocity=False)

# Generate long seed MIDI auto-continuation
output_tokens = mpt.generate_long(model, input_tokens, return_prime=True)

# Save output batch 0 to MIDI
mpt.tokens_to_midi(output_tokens[0], model_with_velocity=False)
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
input_tokens = mpt.midi_to_tokens(sample_midi_path, model_with_velocity=False)

# Inpaint pitches
output_tokens = mpt.inpaint_pitches(model, input_tokens)

# Save output to MIDI
mpt.tokens_to_midi(output_tokens, model_with_velocity=False)
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
input_tokens = mpt.midi_to_tokens(sample_midi_path, model_with_velocity=True)

# Inpaint velocities
output_tokens = mpt.inpaint_velocities_simple(model, input_tokens)

# Save output to MIDI
mpt.tokens_to_midi(output_tokens, model_with_velocity=True)
```

### Seq2Seq velocities inpainting

```python
# Import Monster Piano Transformer as mpt
import monsterpianotransformer as mpt

# Load desired Monster Piano Transformer model
# There are several to choose from...
model = mpt.load_model(model_name='velocity inpainting - 3 epochs')

# Get sample seed MIDI path
sample_midi_path = mpt.get_sample_midi_files()[6][1]

# Load seed MIDI
input_tokens = mpt.midi_to_tokens(sample_midi_path, model_with_velocity=True)

# Inpaint velocities
output_tokens = mpt.inpaint_velocities_seq2seq(model, input_tokens, verbose=True)

# Save output to MIDI
mpt.tokens_to_midi(output_tokens, model_with_velocity=True)
```

***

### Project Los Angeles
### Tegridy Code 2025
