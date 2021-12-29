
from miniumap._umap import umap

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

from argparse import ArgumentParser
import sys


def main():
    parser = ArgumentParser(description=
"""
This program implements UMAP (Uniform Manifold Approximation and Projection) algorithm 
commonly used to perform dimension reduction for datasets of a high dimension, based
on the original paper (DOI: 10.21105/joss.00861).
""")
    parser.add_argument('dataset',
                        help='Path to the dataset. Each row in the dataset represents a data point with coordinates separated by space')
    # parser.add_argument('-b', '--by', required=False, default='row', choices=('column', 'row'),
    #                     help='Specify whether the records are per column or per row')
    # parser.add_argument('-i', '--ignore', required=False, type=int,
    #                     help='The number of lines or columns to skip in the dataset')
    parser.add_argument('-k', '--n_nearest_neighbors', required=False, type=int, default=15,
                        help='Number of of nearest neighbors to consider for each data point. Smaller values tend to retain local features at the cost of global topology. '
                             'Default is 15')
    parser.add_argument('-d', '--output_dimension', required=False, type=int, default=2,
                        help='Embedding (output) dimension. Default is 2')
    parser.add_argument('-m', '--min_dist', required=False, type=float, default=0.1,
                        help='Minimum distance between two points in the output. Smaller values tend to generate more salient clusters. Default is 0.1')
    parser.add_argument('-r', '--random_seed', required=False, type=int, default=None,
                        help='Random seed for RNG')
    parser.add_argument('-s', '--spread', required=False, type=float, default=1.0,
                        help='Spread of each point. Should be set alongside --min_dist. Default is 1.0')
    parser.add_argument('-e', '--n_epochs', required=False, type=int, default=200,
                        help='Number of training epochs. Larger values make the algorithm run longer but could lead to better results. '
                             'Default is 200')
    parser.add_argument('--negative_sampling_rate', required=False, type=int, default=5,
                        help='Number of negative samples to update for each positive sample. Default is 5')
    parser.add_argument('--metric', required=False, default='euclidean',
                        help='Metric function for computing nearest neighbor graph. Default is euclidean')
    parser.add_argument('-o', '--out', required=False,
                        help='Path to the output file. Default is stdout. Use "dummy" to disable output')

    args = parser.parse_args()

    dataset = np.loadtxt(args.dataset)
    embedding = umap(dataset,
                     k=args.n_nearest_neighbors,
                     d=args.output_dimension,
                     min_dist=args.min_dist,
                     spread=args.spread,
                     n_epochs=args.n_epochs,
                     random_seed=args.random_seed,
                     n_neg_samples=args.negative_sampling_rate,
                     metric=args.metric)

    # Use 'dummy' to disable output
    if args.out != 'dummy':
        out_file = open(args.out, "w") if args.out else sys.stdout

        for i in range(0, embedding.shape[0]):
            for j in range(0, embedding.shape[1]):
                out_file.write(str(embedding[i, j]) + ' ')
            out_file.write('\n')

        if out_file is not sys.stdout:
            out_file.flush()
            out_file.close()

    # If matplotlib is installed, visualize data for output dimensions 2 and 3
    if HAVE_MATPLOTLIB:
        if args.output_dimension == 2:
            plt.scatter(embedding[:, 0], embedding[:, 1])
            plt.show()
        elif args.output_dimension == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2])
            plt.show()


if __name__ == '__main__':
    sys.exit(main())
