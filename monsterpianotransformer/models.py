#===================================================================================================
# Monster Piano Transformer models Python module
#===================================================================================================
# Project Los Angeles
# Tegridy Code 2025
#===================================================================================================
# License: Apache 2.0
#===================================================================================================

MODELS_HF_REPO_LINK = 'asigalov61/Monster-Piano-Transformer'
MODELS_HF_REPO_URL = 'https://huggingface.co/asigalov61/Monster-Piano-Transformer'

#===================================================================================================

MODELS_INFO = {'without velocity - 7 epochs': 'Best auto-regressive model (without velocity) which was trained for 7 epochs on full Monster Piano dataset.',
               'without velocity - 3 epochs': 'Comparison auto-regressive model (without velocity) which was trained for 3 epochs on full Monster Piano dataset.',
               'with velocity - 3 epochs': 'Comparison auto-regressive model (with velocity) which was trained for 3 epochs on full Monster Piano dataset.',
               'velocity inpainting - 2 epochs': 'Seq2Seq model for velocity inpainting which was trained for 2 epochs on all compositions with expressive velocity in Monster Piano dataset.',
               'timings inpainting - 2 epochs': 'Seq2Seq model for start-times and durations inpainting which was trained for 2 epochs on full Monster Piano dataset.',
               'bridge inpainting - 2 epochs': 'Seq2Seq model for bridge inpainting which was trained for 2 epochs on full Monster Piano dataset.',
               'chords progressions - 3 epochs': 'Chords progressions auto-regressive model which was trained for 3 epochs on full Monster Chords dataset.',
               'chords texturing - 3 epochs': 'Chords texturing auto-regressive model which was trained for 3 epochs on full Monster Chords dataset.'
               
              }     

#===================================================================================================

MODELS_FILE_NAMES = {'without velocity - 7 epochs': 'Monster_Piano_Transformer_No_Velocity_Trained_Model_161960_steps_0.7775_loss_0.7661_acc.pth',
                     'without velocity - 3 epochs': 'Monster_Piano_Transformer_No_Velocity_Trained_Model_69412_steps_0.8577_loss_0.7442_acc.pth',
                     'with velocity - 3 epochs': 'Monster_Piano_Transformer_Velocity_Trained_Model_59896_steps_0.9055_loss_0.735_acc.pth',
                     'velocity inpainting - 2 epochs': 'Monster_Piano_Transformer_Velocity_Inpaiting_Trained_Model_38777_steps_0.9227_loss_0.734_acc.pth',
                     'timings inpainting - 2 epochs': 'Monster_Piano_Transformer_Timings_Inpainting_Trained_Model_38402_steps_0.622_loss_0.8218_acc.pth',
                     'bridge inpainting - 2 epochs': 'Monster_Piano_Transformer_Bridge_Inpainting_Trained_Model_53305_steps_0.825_loss_0.7578_acc.pth',
                     'chords progressions - 3 epochs': 'Monster_Piano_Transformer_Chords_Progressions_Trained_Model_31648_steps_0.8085_loss_0.7868_acc.pth',
                     'chords texturing - 3 epochs': 'Monster_Piano_Transformer_Chords_Texturing_Trained_Model_33878_steps_0.6109_loss_0.8234_acc.pth'
                    }

#===================================================================================================

MODELS_PARAMETERS = {'without velocity - 7 epochs': {'seq_len': 2048,
                                                     'pad_idx': 384,
                                                     'dim': 2048,
                                                     'depth': 4,
                                                     'heads': 32,
                                                     'rope': True,
                                                     'params': 202
                                                    },
                     
                     'without velocity - 3 epochs': {'seq_len': 2048,
                                                     'pad_idx': 384,
                                                     'dim': 2048,
                                                     'depth': 4,
                                                     'heads': 32,
                                                     'rope': True,
                                                     'params': 202
                                                    },
                     
                     'with velocity - 3 epochs': {'seq_len': 2048,
                                                  'pad_idx': 512,
                                                  'dim': 2048,
                                                  'depth': 4,
                                                  'heads': 32,
                                                  'rope': True,
                                                  'params': 202
                                                 },
                     
                     'velocity inpainting - 2 epochs': {'seq_len': 2103,
                                                        'pad_idx': 515,
                                                        'dim': 2048,
                                                        'depth': 4,
                                                        'heads': 32,
                                                        'rope': True,
                                                        'params': 203
                                                       },
                     
                     'timings inpainting - 2 epochs': {'seq_len': 2002,
                                                       'pad_idx': 515,
                                                       'dim': 2048,
                                                       'depth': 4,
                                                       'heads': 32,
                                                       'rope': True,
                                                       'params': 203
                                                      },
                     
                     'bridge inpainting - 2 epochs': {'seq_len': 1403,
                                                      'pad_idx': 388,
                                                      'dim': 2048,
                                                      'depth': 4,
                                                      'heads': 32,
                                                      'rope': True,
                                                      'params': 202
                                                     },
                     
                     'chords progressions - 3 epochs': {'seq_len': 1024,
                                                        'pad_idx': 321,
                                                        'dim': 2048,
                                                        'depth': 4,
                                                        'heads': 32,
                                                        'rope': True,
                                                        'params': 202
                                                       },
                     
                     'chords texturing - 3 epochs': {'seq_len': 1024,
                                                     'pad_idx': 449,
                                                     'dim': 2048,
                                                     'depth': 4,
                                                     'heads': 32,
                                                     'rope': True,
                                                     'params': 203
                                                    }
                     }

#===================================================================================================

def detect_model_type(model):

    seq_len = model.max_seq_len
    pad_idx = model.pad_value

    model_type = 'unknown'
    model_idx = -1

    for i, np in enumerate(MODELS_PARAMETERS.items()):
        if np[1]['seq_len'] == seq_len and np[1]['pad_idx'] == pad_idx:
            model_type = np[0]
            model_idx = i
            break

    return model_type, model_idx

#===================================================================================================
# This is the end of models Python module
#===================================================================================================