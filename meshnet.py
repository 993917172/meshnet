import numpy as np
import os

def load_data(root_dir):
	x_train = np.load(os.path.join(root_dir, '300W-3D_HELEN_images.npy'))
	y_train = np.load(os.path.join(root_dir, '300W-3D_HELEN_points.npy'))
	x_test = np.load(os.path.join(root_dir, '300W-3D_AFW_images.npy'))
	y_test = np.load(os.path.join(root_dir, '300W-3D_AFW_points.npy'))
	return ((x_train, y_train), (x_test, y_test))