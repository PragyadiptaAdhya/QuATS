import numpy as np
from config import Config

config = Config()
def extract_uniform_random_patches(img_rgb):

	H, W, _ = img_rgb.shape
	crop_w, crop_h = config.PATCH_SIZE, config.PATCH_SIZE
	factor = 1.5	
	# The grid cell size is strictly patch_size * factor
	cell_w = int(crop_w * factor)
	cell_h = int(crop_h * factor)

	# Determine how many such grids fit in the image
	max_cols = W // cell_w
	max_rows = H // cell_h
	
	if max_cols <= 0 or max_rows <= 0:
		return img_rgb 
		
	canvas = np.zeros((max_rows * crop_h, max_cols * crop_w, 3), dtype=img_rgb.dtype)
	
	# Calculate maximum allowable top-left coordinates within a cell
	max_jitter_x = cell_w - crop_w
	max_jitter_y = cell_h - crop_h
	
	# Pre-generate random jitter offsets for all grid cells simultaneously
	jitter_x = np.random.randint(0, max_jitter_x + 1, size=(max_rows, max_cols))
	jitter_y = np.random.randint(0, max_jitter_y + 1, size=(max_rows, max_cols))
	
	for r in range(max_rows):
		for c in range(max_cols):
			# Calculate the absolute top-left corner in the original image
			start_y = (r * cell_h) + jitter_y[r, c]
			start_x = (c * cell_w) + jitter_x[r, c]
			
			# Extract crop and directly assign it to the corresponding canvas position
			canvas[r*crop_h : (r+1)*crop_h, c*crop_w : (c+1)*crop_w] = \
				img_rgb[start_y : start_y + crop_h, start_x : start_x + crop_w]
				
	return canvas

