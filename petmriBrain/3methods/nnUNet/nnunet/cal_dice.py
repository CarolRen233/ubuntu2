import json
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='AneurysmSeg study evaluation')
parser.add_argument('-p', '--pred_path', type=str, required=True, default='',
                    help='pred path')

args = parser.parse_args()

def get_results(pred_path):
    with open(os.path.join(pred_path, "summary.json")) as f:
        summary = json.load(f)
    scores = []
    for i,val in summary['results']['mean'].items():
        print('idx: ', i, 'Dice:', val['Dice'])
        scores.append(val['Dice'])
    print('mean:', 'Dice:', np.mean(scores))


if __name__ == '__main__':
    get_results(args.pred_path)
