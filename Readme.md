# miniumap



## Installation

miniumap requires `numpy` for matrix manipulations, `scipy` and `scikit-learn` for sparse matrices, curve fitting and computing eigen values for embedding layout. The computation of nearest neighbor graphs is done by `pynndescent`. miniumap also requires `Cython` for speedup. `matplotlib` is an optional package that, if installed, will be called to visualize embedding of dimensions 2 or 3.

First, install the requirements. We recommend installing manually with `conda`

```
conda install scikit-learn
conda install pynndescent
conda install Cython
conda install matplotlib # optional, for visualization
```

Alternatively, with pip

```
pip install -r requirements.txt
```

Then install the package itself

```
python setup.py install
```

## Usage

miniumap has two use modes: command line or library. For command line mode, use

```
python -m miniumap PATH_TO_INPUT -o PATH_TO_OUTPUT
```

For library mode in Python code, use

```python
import miniumap
from sklearn.datasets import load_digits

digits = load_digits()

embedding = miniumap.umap(digits.data)
```

We recommend setting the following optional parameters for alternative embedding effects:

- `d` (default=2): The embedding dimension. Use 3 for 3D projection. Higher values are supported but not recommended because UMAP is already very good at embedding 2D or 3D layouts.

- `k` (default=2): The number of nearest neighbors to consider. Larger values will retain global structures, but local details may be obscured.
- `min_dist` (default=0.1): The expected minimum distance between any two points in the embedding. Smaller values will allow more salient clustering. Recommended range is 0.001 to 0.5.
- `n_epochs` (default=300): The number of training epochs. Because this implementation is not very efficient (no parallelization) the default is set to 300. For very large datasets, `k` or `d`, the recommended value is 200.

Because UMAP is randomized, setting the random seed (`-r` parameter in command line and `random_seed` parameter in Python) is necessary for reproducibility.

For command line usages, see the program help

```
python -m miniumap -h
```

