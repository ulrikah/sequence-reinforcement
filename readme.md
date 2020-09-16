## Issues

- max rewards per now is 0.75 by the way it is made, which doesn't correspond to the ranges
    -> look into other ways of calculating reward which always yield a reward in range (0, 1)

- epsilon: should explore/exploit be on a per episode or per step level?

## To do

- argparse for n_episodes ++
- implement a logger that can log to file. use same naming convention for plots and logs
- save model checkpoints!
- define observation space and action space in a more gym style manner (inherit from gym.Space)

- log exploration vs. exploitation ratio for each episode
- env.render() should actually render the MIDI

## Questions

- punish for keeping the same sequence as it is?

## Alternative way

Input to neural network is the concatenated state of the snare and the kick pattern
    -> e.g. 16 + 16 = 32 
    -> should be one-hot encoded
    => observation space is a vector of length 32

Output of the neural network is a number that is used to toggle the current step in the sequence
    -> for a sequence of 16 steps, we need 16 steps for each toggle + 1 additional step for the none_action

A step in the environment consists of:


Samplng a random action involves
    -> toggling ONE of the 16 musical steps at random