# Gomoku-AlphaZero

## Build

``` bash
python setup.py build_ext -i
rm mcts.c
rm board.c
```

## Train

``` bash
python train.py
```

## Play

``` bash
python play.py
```

## References

- [AlpahZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku/)
- [TensorFlow Models AlphaZero](https://github.com/tensorflow/models/blob/master/research/a3c_blogpost/a3c_cartpole.py)
