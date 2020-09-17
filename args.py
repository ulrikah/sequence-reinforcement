import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Reinforcement learning for generative MIDI sequences')
    
    parser.add_argument('mode', choices=['train', 'test'], default='train', help='to test or train the system')
    parser.add_argument('-c', '--checkpoint', help="path to pretrained checkpoint", default=None)

    args = parser.parse_args()
    return args