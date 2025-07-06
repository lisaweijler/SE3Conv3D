import numpy as np

num_test_epochs = 30

DS_AUGMENTS = [
        {
            'name': 'CenterAug',
            'p_apply_extra_tensors': [False, False, False, False]
        },
        {
            'name': 'RotationAug',
            'p_prob': 1.0,
            'p_axis' : 2,
            'p_angle_values': [(i/num_test_epochs)*2.*np.pi for i in range(num_test_epochs)],
            'p_apply_extra_tensors': [True, False, False, False]
        }
    ]
