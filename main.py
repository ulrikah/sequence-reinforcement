from train import train
from agent import Agent
from env import MidiEnv

'''

    inspired by RaveForce: 
    https://github.com/chaosprint/RaveForce/blob/master/Python/raveforce.py

'''

# System variables that should be set from CLI args
N_EPISODES = 1000
MAX_STEPS = 16
BATCH_SIZE = 256
LR = 1e-2
GAMMA = 0.99
LOG = True
SAVE_MODEL_TO = "checkpoints/"

def main():
    env = MidiEnv(n_bars=2)
    agent = Agent(env, learning_rate=LR, gamma=GAMMA)

    print("Optimizer", agent.optimizer)
    print("")

    train(
        env,
        agent,
        n_episodes=N_EPISODES,
        max_steps=MAX_STEPS,
        batch_size=BATCH_SIZE,
        log=LOG,
        save_model_to=SAVE_MODEL_TO
    )

if __name__ == "__main__":
    main()
