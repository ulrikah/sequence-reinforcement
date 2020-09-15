## Issues

- max rewards per now is 0.75 by the way it is made, which doesn't correspond to the ranges
    -> look into other ways of calculating reward which always yield a reward in range (0, 1)
- weight initialisation does have a substantial impact. what to do?

- epsilon: should explore/exploit be on a per episode or per step level?

- evaluating when I'm done seems a bit random

## To do

- argparse for n_episodes ++
- implement a logger that can log to file. use same naming convention for plots and logs
- save model checkpoints!
- define observation space and action space in a more gym style manner (inherit from gym.Space)
- metric could be a separate class / interface
    -> attributes: reward_range, done_range
    -> functions: calculate_reward(kick_seq, snare_seq)

- log exploration vs. exploitation ratio for each episode


## Questions


- What criteria should I use for done?

- rewards are in range (0.0, 1.0), but random sampling yields a reward of ~0.5

- First step in the snare sequence almost always ends up as close to 1. Any way to avoid that?

- NN spits out sequences between (0.0, 1.0) --> what is the correct