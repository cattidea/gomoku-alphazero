# Gomoku-AlphaZero

## Build

```bash
python setup.py build_ext -i
rm mcts.c
rm board.c
```

## Train

```bash
TF_CPP_MIN_LOG_LEVEL=3 python train.py \
    --width=8 \
    --height=8 \
    --lr=0.002
```

## Play

```bash
python play.py \
    --mode pve \
    --weights="./data/model-8x8#5.h5" \
    --width=8 \
    --height=8
```

## References

-  [AlpahZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku/)
