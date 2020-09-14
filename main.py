from train import train
from agent import Agent
from env import MidiEnv

# System variables that should be set from CLI args
N_EPISODES = 10000
MAX_STEPS = 100
BATCH_SIZE = 128
LOG = True

def main():
    env = MidiEnv()
    agent = Agent(env)
    train(env, agent, n_episodes=N_EPISODES, max_steps=MAX_STEPS, batch_size=BATCH_SIZE, log=LOG)

if __name__ == "__main__":
    main()