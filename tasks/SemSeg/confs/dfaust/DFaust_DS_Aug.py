import numpy as np

DS_AUGMENTS = [
        {
            'name': 'CenterAug',
            'p_apply_extra_tensors': []
        },

        {
            'name': 'NoiseAug',
            'p_prob': 1.0,
            'p_stddev' : 0.005,
            'p_clip' : 0.02,
            'p_apply_extra_tensors': []
        },

 
    ]
