from train import train
from agent import Agent
from env import MidiEnv

# System variables that should be set from CLI args
N_EPISODES = 100
MAX_STEPS = 50
BATCH_SIZE = 128
LR = 1e-2
GAMMA = 0.99
LOG = True

def main():
    env = MidiEnv()
    agent = Agent(env, learning_rate=LR, gamma=GAMMA)
    train(env, agent, n_episodes=N_EPISODES, max_steps=MAX_STEPS, batch_size=BATCH_SIZE, log=LOG)

if __name__ == "__main__":
    main()