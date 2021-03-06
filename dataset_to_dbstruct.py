import argparse
from pathlib import Path

from mapillary_sls.datasets.msls import MSLS, default_cities
from mapillary_sls.utils.eval import download_msls_sample

import sys
sys.path.append('../NetVLAD2.0/pytorch-NetVlad-Nanne/')
from datasets import parse_db_struct, save_db_struct, dbStruct


def main():

    parser = argparse.ArgumentParser()
    root_default = Path(__file__).parent / 'MSLS_sample'
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
    parser.add_argument('--subtask',
                        type=str,
                        default='all',
                        help='Subtask to evaluate on: '
                             '[all, s2w, w2s, o2n, n2o, d2n, n2d]')
    args = parser.parse_args()

    if not args.msls_root.exists():
        if args.msls_root == root_default:
            download_msls_sample(args.msls_root)
        else:
            print(args.msls_root, root_default)
            raise FileNotFoundError("Not found: {}".format(args.msls_root))

    args.seq_length = 1

    if args.cities == 'test' or args.cities in default_cities['test']:
        mode = 'test'
    else:
        mode = 'val'

    dataset = MSLS(args.msls_root, cities=args.cities, mode=mode, posDistThr=args.threshold,
                    task=args.task, seq_length=args.seq_length, subtask=args.subtask)

    dbImage = [database_key.replace(str(args.msls_root) + '/', '') for database_key in dataset.dbImages]
    qImage = [query_key.replace(str(args.msls_root) + '/', '') for query_key in dataset.qImages[dataset.qIdx]]

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

    save_db_struct('mapillary' + args.cities + '.mat', db)


if __name__ == "__main__":
    main()
