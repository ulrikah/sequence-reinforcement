## Issues

- looks like the training is very poor when the neural network comes in
- max rewards per now is 0.75 by the way it is made, which doesn't correspond to the ranges
    -> look into other ways of calculating reward which always yield reward in range (0, 1)

## To do

- argparse for n_episodes ++
- implement a logger that can log to file. use same naming convention for plots and logs
- save model checkpoints!