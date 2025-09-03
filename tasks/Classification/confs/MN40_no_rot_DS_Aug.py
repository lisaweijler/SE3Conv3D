import numpy as np

DS_AUGMENTS = [
    {
        'name': 'CenterAug',
        'p_apply_extra_tensors': [False]
    },
    
    {
        'name': 'NoiseAug',
        'p_prob': 1.0,
        'p_stddev' : 0.005,
        'p_clip' : 0.02,
        'p_apply_extra_tensors': [False]
    },
    {
        'name': 'LinearAug',
        'p_prob': 1.0,
        'p_min_a' : 0.9,
        'p_max_a' : 1.1,
        'p_min_b' : 0.0,
        'p_max_b' : 0.0,
        'p_channel_independent' : True,
        'p_apply_extra_tensors': [False]
    },
    {
        'name': 'MirrorAug',
        'p_prob': 1.0,
        'p_mirror_prob' : 0.5,
        'p_axes' : [True, False, True],
        'p_apply_extra_tensors': [True]
    }
]