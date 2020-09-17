from train import train
from agent import Agent
from env import SimpleEnv
from metrics import AbsoluteDifference, NormalizedSum


'''
    inspired by RaveForce: 
    https://github.com/chaosprint/RaveForce/blob/master/Python/raveforce.py
'''

# System variables that should be set from CLI args
N_EPISODES = 20000
MAX_STEPS = 30
BATCH_SIZE = 2
LR = 1e-2
GAMMA = 0.7
EPS_DECAY = 10000
LOG = True
LOG_INTERVAL = 10
SAVE_MODEL_TO = "checkpoints/"
LOAD_MODEL_FROM = None

def main():
    env = SimpleEnv(n_bars = 2)
    agent = Agent(
        env,
        in_dim = len(env.get_state()),
        out_dim = env.action_space.n,
        learning_rate=LR,
        gamma=GAMMA)

    print("Optimizer", agent.optimizer)
    print("")
    print("Network", agent.model)
    print("")

    train(
        env,
        agent,
        n_episodes=N_EPISODES,
        max_steps=MAX_STEPS,
        batch_size=BATCH_SIZE,
        eps_decay=EPS_DECAY,
        log=LOG,
        log_interval=LOG_INTERVAL if LOG_INTERVAL <= N_EPISODES else N_EPISODES,
        save_model_to=SAVE_MODEL_TO,
        load_model_from=LOAD_MODEL_FROM
    )

if __name__ == "__main__":
    main()
