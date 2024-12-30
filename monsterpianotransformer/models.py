#===================================================================================================
# Giant Music Transformer models Python module
#===================================================================================================
# Project Los Angeles
# Tegridy Code 2024
#===================================================================================================
# License: Apache 2.0
#===================================================================================================

MODELS_HF_REPO_LINK = 'asigalov61/Giant-Music-Transformer'
MODELS_HF_REPO_URL = 'https://huggingface.co/asigalov61/Giant-Music-Transformer'

MODELS_INFO = {'extra-large-b': '1B+ parameters legacy model.\nTrained for 1 epoch.',
               'extra-large': 'Extra large legacy model.\nTrained for 1.5 epochs on on Los Angeles MIDI dataset.',
               'large': 'Large legacy model.\nTrained for 1.5 epochs on on Los Angeles MIDI dataset.',
               'medium': 'Medium current model.\nTrained for 3 epochs on Monster MIDI dataset.',
               'medium-swapped': 'Medium swapped comparison model.\nTrained for 3 epochs on Monster MIDI dataset.'
               }

MODELS_FILE_NAMES = {'extra-large-b': 'Giant_Music_Transformer_Extra_Large_Trained_Model_9001_steps_0.5951_loss_0.8499_acc.pth',
                     'extra-large': 'Giant_Music_Transformer_Extra_Large_Trained_Model_18001_steps_0.2657_loss_0.9272_acc.pth',
                     'large': 'Giant_Music_Transformer_Large_Trained_Model_36074_steps_0.3067_loss_0.927_acc.pth',
                     'medium': 'Giant_Music_Transformer_Medium_Trained_Model_42174_steps_0.5211_loss_0.8542_acc.pth',
                     'medium-swapped': 'Giant_Music_Transformer_Medium_Swapped_Trained_Model_24663_steps_0.7008_loss_0.8106_acc.pth'
                     }

MODELS_SEQ_LEN = 8192
MODELS_PAD_IDX = 19463

MODELS_PARAMETERS = {'extra-large-b': {'dim': 1024,
                                       'depth': 64,
                                       'heads': 32,
                                       'rope': False,
                                       'params': 1122
                                      },
                     
                     'extra-large': {'dim': 1024,
                                       'depth': 44,
                                       'heads': 32,
                                       'rope': False,
                                       'params': 786
                                      },
                     
                     'large': {'dim': 1024,
                               'depth': 32,
                               'heads': 32,
                               'rope': False,
                               'params': 585
                              },
                     
                     'medium': {'dim': 2048,
                               'depth': 8,
                               'heads': 32,
                               'rope': True,
                               'params': 482
                              },
                     
                     'medium-swapped': {'dim': 2048,
                                       'depth': 8,
                                       'heads': 32,
                                       'rope': True,
                                       'params': 482
                                      },
                    }

#===================================================================================================
# This is the end of models Python module
#===================================================================================================
