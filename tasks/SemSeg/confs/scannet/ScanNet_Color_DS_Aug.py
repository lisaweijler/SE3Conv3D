import numpy as np

DS_AUGMENTS = [
        {
            'name': 'LinearAug',
            'p_prob': 1.0,
            'p_min_a' : 0.8,
            'p_max_a' : 1.2,
            'p_min_b' : -0.2,
            'p_max_b' : 0.2,
            'p_channel_independent' : False,
            'p_apply_extra_tensors': []
        },
        {
            'name': 'LinearAug',
            'p_prob': 1.0,
            'p_min_a' : 1.0,
            'p_max_a' : 1.0,
            'p_min_b' : -0.2,
            'p_max_b' : 0.2,
            'p_channel_independent' : True,
            'p_apply_extra_tensors': []
        },
        {
            'name': 'NoiseAug',
            'p_prob': 1.0,
            'p_stddev' : 0.01,
            'p_clip' : 0.05,
            'p_apply_extra_tensors': [False, False, False]
        },
    ]