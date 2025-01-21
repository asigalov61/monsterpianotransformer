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

MODELS_INFO = {'without velocity - 7 epochs': 'Best model (without velocity) which was trained for 7 epochs on full Monster Piano dataset.',
               'without velocity - 3 epochs': 'Comparison model (without velocity) which was trained for 3 epochs on full Monster Piano dataset.',
               'with velocity - 3 epochs': 'Comparison model (with velocity) which was trained for 3 epochs on full Monster Piano dataset.',
               'velocity inpainting - 3 epochs': 'Seq2Seq model for velocity inpainting which was trained for 3 epochs on all compositions with expressive velocity in Monster Piano dataset.'
              }     

#===================================================================================================

MODELS_FILE_NAMES = {'without velocity - 7 epochs': 'Monster_Piano_Transformer_No_Velocity_Trained_Model_161960_steps_0.7775_loss_0.7661_acc.pth',
                     'without velocity - 3 epochs': 'Monster_Piano_Transformer_No_Velocity_Trained_Model_69412_steps_0.8577_loss_0.7442_acc.pth',
                     'with velocity - 3 epochs': 'Monster_Piano_Transformer_Velocity_Trained_Model_59896_steps_0.9055_loss_0.735_acc.pth',
                     'velocity inpainting - 3 epochs': 'Monster_Piano_Transformer_Velocity_Inpaiting_Trained_Model_50057_steps_0.7645_loss_0.783_acc.pth'
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
                     
                     'velocity inpainting - 3 epochs': {'seq_len': 2103,
                                                        'pad_idx': 515,
                                                        'dim': 1024,
                                                        'depth': 4,
                                                        'heads': 32,
                                                        'rope': True,
                                                        'params': 68
                                                       }
                     }

#===================================================================================================
# This is the end of models Python module
#===================================================================================================