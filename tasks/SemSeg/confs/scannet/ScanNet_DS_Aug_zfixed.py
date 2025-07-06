import numpy as np

DS_AUGMENTS = [
        {
            'name': 'CenterAug',
            'p_apply_extra_tensors': [False, False, False, False]
        },
        {
            'name': 'MirrorAug',
            'p_prob': 1.0,
            'p_mirror_prob' : 0.5,
            'p_axes' : [True, True, False],
            'p_apply_extra_tensors': [True, False, False, False]
        },
        {
            'name': 'RotationAug',
            'p_prob': 1.0,
            'p_axis' : 2,
            'p_min_angle' : 0.0,
            'p_max_angle' : 2.0*np.pi,
            'p_apply_extra_tensors': [True, False, False, False]
        },

        {
            'name': 'LinearAug',
            'p_prob': 1.0,
            'p_min_a' : 0.75,
            'p_max_a' : 1.25,
            'p_min_b' : 0.0,
            'p_max_b' : 0.0,
            'p_channel_independent' : True,
            'p_apply_extra_tensors': [False, False, False, False]
        },
        {
            'name': 'ElasticDistortionAug',
            'p_prob': 0.95,
            'p_granularity' : [0.1, 0.2, 0.4],
            'p_magnitude' : [0.15, 0.3, 0.6],
            'p_apply_extra_tensors': [False, False, False, False]
        },
        {
            'name': 'NoiseAug',
            'p_prob': 1.0,
            'p_stddev' : 0.005,
            'p_clip' : 0.02,
            'p_apply_extra_tensors': [False, False, False, False]
        },
        {
            'name': 'CropPtsAug',
            'p_prob' : 1.0,
            'p_max_pts' : 120000,
            'p_crop_ratio': 0.8, 
            'p_apply_extra_tensors' : [True, True, True, True]
        },
        {
            'name': 'CenterAug',
            'p_axes': [True, True, False],
            'p_apply_extra_tensors': [False, False, False, False]
        },
        {
            'name': 'TranslationAug',
            'p_prob': 1.0,
            'p_max_aabb_ratio' : np.array([0.5, 0.5, 0.0]),
            'p_apply_extra_tensors': [False, False, False, False]
        }
    ]
