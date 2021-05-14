import argparse

from loader.generator import Transform, GeneratorType


TRANSFORMS_TYPES = ','.join(map(lambda t: t.value, Transform))
DATASET_TYPES = ','.join(map(lambda t: t.value, GeneratorType))

def validate(args):
    if args.from_csv == True:
        assert args.features is not None and args.label is not None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='name or type of dataset')

    parser.add_argument('--from-csv', action='store_true',
                        help='upload dataset from csv, otherwise please use one of dataset types: {}'.format(DATASET_TYPES))

    parser.add_argument('--features', type=str, required=False,
                        help='comma separated list of feature columns in csv header(works only with from_csv=true)')

    parser.add_argument('--label', type=str, required=False,
                        help='name or target column in csv header(works only with from_csv=true)')

    parser.add_argument('--batch-size', type=int, required=False, default=8,
                            help='bacth size for train, default=8')

    parser.add_argument('-n-epochs', type=int, required=False, default=10,
                                help='Number of train epochs, default=10')

    parser.add_argument('--lr', type=float, required=False, default=2e-3,
                                    help='Learning rate, default=2e-3')

    parser.add_argument('--noise', type=float, required=False, default=0,
                            help='strength of noise in data, default=0')

    parser.add_argument('--val-ratio', type=float, required=False, default=0.1,
                            help='ratio of val data in all input data, default=0.1')

    parser.add_argument('--layers', type=str, required=False, default="32,32",
        help='number of neurons in each layer separated by comma last layer with size 1 will be added, default=32,32')

    parser.add_argument('--transforms', type=str, required=False, default="",
        help="transforms which will be added to 2D generated dataset(wotks only with from_csv=false)" \
        + " declared in comma separated sequence, please use one of transform types: {}, default=''".format(TRANSFORMS_TYPES))

    args = parser.parse_args()
    validate(args)
    return args
