import numpy as np

SCANNET20_COLORS = np.array([
       [0, 0, 0],
       [174, 199, 232],		# wall
       [152, 223, 138],		# floor
       [31, 119, 180], 		# cabinet
       [255, 187, 120],		# bed
       [188, 189, 34], 		# chair
       [140, 86, 75],  		# sofa
       [255, 152, 150],		# table
       [214, 39, 40],  		# door
       [197, 176, 213],		# window
       [148, 103, 189],		# bookshelf
       [196, 156, 148],		# picture
       [23, 190, 207], 		# counter
       [247, 182, 210],		# desk
       [219, 219, 141],		# curtain
       [255, 127, 14], 		# refrigerator
       [158, 218, 229],		# shower curtain
       [44, 160, 44],  		# toilet
       [112, 128, 144],		# sink
       [227, 119, 194],		# bathtub
       [82, 84, 163]  		# otherfurn
])

SCANNET_CLASS_IDS_20 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

SCANNET_RND_COLORS = np.random.uniform(0.0, 1.0, (25000, 3)).astype(np.float32)

def save_scannet20_scene_colors(p_path, p_pts, p_labels):
    cur_colors = SCANNET20_COLORS[p_labels]/255.
    cur_data = np.concatenate((p_pts, cur_colors), -1)
    np.savetxt(p_path, cur_data)

def save_scannet20_scene_rnd_colors(p_path, p_pts, p_labels):
    cur_colors = SCANNET_RND_COLORS[p_labels]
    cur_data = np.concatenate((p_pts, cur_colors), -1)
    np.savetxt(p_path, cur_data)

def save_scannet20_scene_labels(p_path, p_labels):
    new_labels = SCANNET_CLASS_IDS_20[p_labels]
    np.savetxt(p_path, new_labels.reshape((-1,)), fmt='%i', delimiter='\t')