# A utility script for reading inpainting masks from a npz file

import numpy as np
import io

filename = "/home/pandeyk1/ciip_results/masks/imagenet_freeform_masks.npz"
outfile = "/home/pandeyk1/ciip_results/masks/20ff.npz"
shape = [10000, 256, 256]

# Load the npz file.
with open(filename, 'rb') as f:
  data = f.read()

data = dict(np.load(io.BytesIO(data)))
print("Categories of masks:")
for key in data:
  print(key)

# Unpack and reshape the masks.
for key in data:
  data[key] = np.unpackbits(data[key], axis=None)[:np.prod(shape)].reshape(shape).astype(np.uint8)

mask = 1 - data["10-20% freeform"][2]
print(np.unique(mask, return_counts=True))
mask = np.stack([mask] * 3, axis=0)
np.savez(outfile, m=mask, shape=(1, np.prod(mask.shape)))