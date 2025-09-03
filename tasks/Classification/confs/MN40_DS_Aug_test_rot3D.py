import numpy as np

DS_AUGMENTS = [
    {
        'name': 'CenterAug',
        'p_apply_extra_tensors': [False]
    },
    {
        'name': 'RotationAug3D',
        'p_prob': 1.0,
        'p_apply_extra_tensors': [True]
    }


]