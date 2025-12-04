# SVD Image Compression

An educational tool to demonstrate image compression using Singular Value Decomposition (SVD).

## Overview

This package provides utilities for compressing images using SVD decomposition. It includes two main functions:
- `reduced_SVD`: Standard SVD compression keeping the highest singular values
- `reverse_reduced_SVD`: Reverse compression approach cutting from the highest values

## Installation

### From source (development mode)

```bash
git clone https://github.com/Samiod131/SVD_image_compression.git
cd SVD_image_compression
pip install -e .
```

### From source (standard installation)

```bash
pip install .
```

## Usage

### As a Python module

```python
from svd_image_compression import reduced_SVD, reverse_reduced_SVD
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

# Load your image
img = imread("your_image.png")

# Apply SVD compression to a channel
U, S, Vh, i, fS = reverse_reduced_SVD(img[:, :, 0], cut=0.1)

# Reconstruct the compressed image
compressed = np.dot(np.dot(U, np.diag(S)), Vh)
```

### Using the example script

```bash
cd examples
python compress_image.py your_image.png 0.1
```

Where:
- `your_image.png` is the path to your image
- `0.1` is the cut value (optional, default=0.1) - represents the relative matrix norm to be cut out

### Using the original interactive script

```bash
cd svd_image_compression
python image_compression.py
```

Then follow the prompts to enter your image filename and compression ratio.

## Requirements

- Python >=3.8
- numpy >=1.20.0
- matplotlib >=3.3.0

## Output

The compression process generates:
- A compressed image file (`comp_<original_name>.png`)
- An SVD spectrum plot (`<original_name>_svd.png`) showing the singular value distribution

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

Samuel Desrosiers
