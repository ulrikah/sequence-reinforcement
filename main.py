from train import train
from agent import Agent
from env import MidiEnv

N_EPISODES = 1000
MAX_STEPS = 400
BATCH_SIZE = 128

def main():
    env = MidiEnv()
    agent = Agent(env)
    train(env, agent, N_EPISODES, MAX_STEPS, BATCH_SIZE)

if __name__ == "__main__":
    main()
