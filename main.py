from test import test
from train import train
from agent import Agent
from env import SimpleEnv
from args import parse_args

# System variables (that should be set from CLI args)
N_EPISODES = 1000 # total number of episode to train for
MAX_STEPS = 50 # number of steps per episode
BATCH_SIZE = 32
LR = 1e-2 # learning rate
GAMMA = 0.7 # discount rate for the Q-learning
EPS_DECAY = 10000 # decay for the epsilon-greedy selection
LOG = True
LOG_INTERVAL = 10
SAVE_MODEL_TO = "checkpoints/"
LOAD_MODEL_FROM = None

def main():
    args = parse_args()

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

    if args.mode == "train":
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
            load_model_from=args.checkpoint
        )

    elif args.mode == "test":
        test(env, agent, checkpoint_path=args.checkpoint)

if __name__ == "__main__":
    main()
