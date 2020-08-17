import argparse
from pathlib import Path

from mapillary_sls.datasets.msls import MSLS
from mapillary_sls.utils.eval import download_msls_sample

import sys
sys.path.append('/home/tobias/robot/NetVLAD2.0/pytorch-NetVlad-Nanne/')
from datasets import parse_db_struct, save_db_struct, dbStruct


def main():

    parser = argparse.ArgumentParser()
    root_default = Path(__file__).parent / 'MSLS_sample'
    parser.add_argument('--prediction',
                        type=Path,
                        default=Path(__file__).parent / 'files' / 'example_msls_im2im_prediction.csv',
                        help='Path to the prediction to be evaluated')
    parser.add_argument('--msls-root',
                        type=Path,
                        default=root_default,
                        help='Path to MSLS containing the train_val and/or test directories')
    parser.add_argument('--threshold',
                        type=float,
                        default=25,
                        help='Positive distance threshold defining ground truth pairs')
    parser.add_argument('--cities',
                        type=str,
                        default='zurich',
                        help='Comma-separated list of cities to evaluate on.'
                             ' Leave blank to use the default validation set (sf,cph)')
    parser.add_argument('--task',
                        type=str,
                        default='im2im',
                        help='Task to evaluate on: '
                             '[im2im, seq2im, im2seq, seq2seq]')
    parser.add_argument('--seq-length',
                        type=int,
                        default=3,
                        help='Sequence length to evaluate on for seq2X and X2seq tasks')
    parser.add_argument('--subtask',
                        type=str,
                        default='all',
                        help='Subtask to evaluate on: '
                             '[all, s2w, w2s, o2n, n2o, d2n, n2d]')
    parser.add_argument('--output',
                        type=Path,
                        default=None,
                        help='Path to dump the metrics to')
    args = parser.parse_args()

    if not args.msls_root.exists():
        if args.msls_root == root_default:
            download_msls_sample(args.msls_root)
        else:
            print(args.msls_root, root_default)
            raise FileNotFoundError("Not found: {}".format(args.msls_root))

    # select for which ks to evaluate
    ks = [1, 5, 10, 20]
    if args.task == 'im2im' and args.seq_length > 1:
        print(f"Ignoring sequence length {args.seq_length} for the im2im task. (Setting to 1)")
        args.seq_length = 1

    dataset = MSLS(args.msls_root, cities = args.cities, mode = 'val', posDistThr = args.threshold, 
                    task = args.task, seq_length = args.seq_length, subtask = args.subtask)

    dbImage = [database_key.replace(str(root_default) + '/', '') for database_key in dataset.dbImages]
    qImage = [query_key.replace(str(root_default) + '/', '') for query_key in dataset.qImages[dataset.qIdx]]

    numDb = len(dbImage)
    numQ = len(qImage)

    whichSet = 'test'
    dataset = 'mapillary'

    posDistThr = args.threshold
    posDistSqThr = posDistThr ** 2
    nonTrivPosDistSqThr = None

    gpsDb = None
    gpsQ = None

    utmDb = None
    utmQ = None

    dbTimeStamp = None
    qTimeStamp = None

    db = dbStruct(whichSet, dataset, dbImage, utmDb, qImage, utmQ, numDb, numQ, posDistThr,
                  posDistSqThr, nonTrivPosDistSqThr, dbTimeStamp, qTimeStamp, gpsDb, gpsQ)

    save_db_struct('MSLS_sample/datasets/mapillary.mat', db)


if __name__ == "__main__":
    main()
