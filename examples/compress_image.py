import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

from svd_image_compression import reverse_reduced_SVD

# Example usage of the SVD image compression
name = 'your_image.png'
cut = 0.1

img = imread(name)
shape = img.shape
img = img.reshape(shape[0], shape[1], -1)
or_float_num = img.size
shape = img.shape

saved_svd = []
full_svd = []
float_num = 0

for j in range(img.shape[2]):
    print(f'Compressing color {j+1} out of {img.shape[2]}')
    U, S, Vh, i, fS = reverse_reduced_SVD(img[:, :, j], cut=cut, limit_max=False)
    float_num += (U.size + S.size + Vh.size)
    saved_svd.append(S.size)
    full_svd.append(fS)
    a = np.dot(U, np.diag(S))
    img[:, :, j] = np.dot(a, Vh)

img = np.clip(img, 0, 1)
print(f'Compression ratio: {or_float_num/float_num:.2f}')

# Save compressed image
output_name = f'comp_{name}'
fig = plt.figure()

if shape[2] == 1:
    img = img.reshape(shape[0], shape[1])
    plt.imsave(output_name, img, cmap='gray')
    plt.plot(full_svd[0], 'k', label='Black and white')
else:
    plt.imsave(output_name, img)
    plt.plot(full_svd[0], 'r', label='Red')
    plt.plot(full_svd[1], 'g', label='Green')
    plt.plot(full_svd[2], 'b', label='Blue')

plt.legend()
plt.xlabel(r'$i$')
plt.ylabel(r'$\frac{S[i]}{S[0]}$', fontsize=14)
plt.title(f'SVD spectrum for {name}')
plt.tight_layout()
fig.savefig(f'{name}_svd.png')

print(f'Saved as: {output_name}')

