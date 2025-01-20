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

MODELS_INFO = {'without velocity - 5 epochs': 'Best model (without velocity) which was trained for 5 epochs on full Monster Piano dataset.',
               'without velocity - 3 epochs': 'Comparison model (without velocity) which was trained for 3 epochs on full Monster Piano dataset.',
               'with velocity - 3 epochs': 'Comparison model (with velocity) which was trained for 3 epochs on full Monster Piano dataset.',
              }     

#===================================================================================================

MODELS_FILE_NAMES = {'without velocity - 5 epochs': 'Monster_Piano_Transformer_No_Velocity_Trained_Model_84419_steps_0.7474_loss_0.7782_acc.pth',
                     'without velocity - 3 epochs': 'Monster_Piano_Transformer_No_Velocity_Trained_Model_50647_steps_0.8166_loss_0.7561_acc.pth',
                     'with velocity - 3 epochs': 'Monster_Piano_Transformer_Velocity_Trained_Model_59896_steps_0.9055_loss_0.735_acc.pth',
                    }

#===================================================================================================

MODELS_PARAMETERS = {'without velocity - 5 epochs': {'seq_len': 2048,
                                                     'pad_idx': 512,
                                                     'dim': 2048,
                                                     'depth': 4,
                                                     'heads': 32,
                                                     'rope': True,
                                                     'params': 202
                                                    },
                     
                     'without velocity - 3 epochs': {'seq_len': 2048,
                                                     'pad_idx': 512,
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
                                                 }
                     }

#===================================================================================================
# This is the end of models Python module
#===================================================================================================