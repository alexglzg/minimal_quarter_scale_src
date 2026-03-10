import numpy as np
from PIL import Image

# Load image
img = Image.open("map_1.png")
arr = np.array(img)

scale = 5  # 0.05 -> 0.01

# Repeat pixels (nearest neighbor upsampling)
arr_highres = np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


# Save result
# Image.fromarray(arr_highres).save("map_2.png")


# what if I want to go from a 0.05 resolution to a 0.1 resolution? I can just take every other pixel, right?
arr_lowres = arr[::2, ::2]
Image.fromarray(arr_lowres).save("map_2.png")