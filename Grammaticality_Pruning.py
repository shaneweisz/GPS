import argparse
from language_quality import extract_good_candidates_by_LQ
from utils import read_candidates
import wandb

def main(args):
    wandb.init(project="gps-pruning", entity="shaneweisz")

    print(f'Reading candidates from {args.candidates_fname}')
    candidates = read_candidates(args.candidates_fname)
    print(f'Before filtering by LQ, there are {len(candidates)} candidates.')

    candidates = extract_good_candidates_by_LQ(candidates, LQ_thres=0.52, num_of_generation=30000)
    print(f'After filtering by LQ, there are {len(candidates)} candidates.')

    print('Writing pruned candidates to file...')
    pruned_candidates_fname = args.candidates_fname.replace('.txt', '_pruned.txt')
    with open(pruned_candidates_fname, 'w') as f:
        for c in candidates:
            f.write(c + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--candidates_fname', type=str, help='candidates file name', required=True)
    args = parser.parse_args()
    assert args.candidates_fname.endswith('.txt')
    main(args)
