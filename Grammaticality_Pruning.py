import argparse
from language_quality import extract_good_candidates_by_LQ
from utils import read_candidates
import wandb

def main(args):
    wandb.init(project="gps-pruning", entity="shaneweisz")
    config = wandb.config
    config.dataset = args.dataset

    print('Load generated candidates from Module 1...')
    candidates = read_candidates('./data/' + args.dataset + '_candidates.txt')

    print('Extract good candidates by LQ...')
    candidates = extract_good_candidates_by_LQ(candidates, LQ_thres=0.52, num_of_generation=30000)
    print(f'After filtering by LQ, there are {len(candidates)} candidates.')

    print('Writing pruned candidates to file...')
    with open('./data/' + args.dataset + '_candidates_pruned.txt', 'w') as f:
        for c in candidates:
            f.write(c + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Main.py', description='choose dataset from reddit, gab, conan')
    parser.add_argument('--dataset', type=str, default='reddit', choices=['reddit', 'gab', 'conan'])
    args = parser.parse_args()
    main(args)
