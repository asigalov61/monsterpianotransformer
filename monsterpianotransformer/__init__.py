from .models import detect_model_type

from .model_loader import load_model

from .midi_processors import midi_to_tokens, tokens_to_midi
from .midi_processors import midi_to_chords, chords_pitches_to_midi

from .sample_midis import get_sample_midi_files

from .monsterpianotransformer import generate, generate_long

from .monsterpianotransformer import generate_chord, generate_chords_pitches
from .monsterpianotransformer import texture_chords

from .monsterpianotransformer import inpaint_timings
from .monsterpianotransformer import inpaint_pitches
from .monsterpianotransformer import inpaint_velocities_simple, inpaint_velocities_seq2seq
from .monsterpianotransformer import inpaint_bridge