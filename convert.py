import argparse

import h5py
import numpy as np

from config import CHANNELS
from policy import PolicyValueModelResNet as PolicyValueModel


def convert_pretrained_weights(
    src_weights_file,
    dst_weights_file,
    src_width=8,
    dst_width=15,
    src_height=8,
    dst_height=15,
):
    model_src = PolicyValueModel(src_width, src_height)
    model_src.build(input_shape=(None, src_width, src_height, CHANNELS))
    model_src.load_weights(src_weights_file)

    model_dst = PolicyValueModel(dst_width, dst_height)
    model_dst.build(input_shape=(None, dst_width, dst_height, CHANNELS))

    assert len(model_src.cnn_layers) == len(model_dst.cnn_layers)
    for i in range(len(model_src.cnn_layers)):
        layer_src = model_src.cnn_layers[i]
        layer_dst = model_dst.cnn_layers[i]
        layer_dst.set_weights(layer_src.get_weights())

    model_dst.save_weights(dst_weights_file)


def convert_pretrained_buffer(
    src_buffer_file,
    dst_buffer_file,
    src_width=8,
    dst_width=15,
    src_height=8,
    dst_height=15,
):
    assert dst_height >= src_height
    assert dst_width >= src_width
    f_src = h5py.File(src_buffer_file, "r")
    f_dst = h5py.File(dst_buffer_file, "w")

    states_src = f_src["states"][...]
    mcts_probs_src = f_src["mcts_probs"][...]

    buffer_length = states_src.shape[0]
    start_width_idx = (dst_width - src_width) // 2
    start_height_idx = (dst_height - src_height) // 2

    states_dst = np.zeros(
        shape=(buffer_length, dst_width, dst_height, CHANNELS),
        dtype=states_src.dtype,
    )
    states_dst[
        :,
        start_width_idx : start_width_idx + src_width,
        start_height_idx : start_height_idx + src_height,
    ] = states_src[:]

    # 最后一根轴只能是全 1 或全 0
    states_dst[:, :, :, -1] = states_src[:, 0:1, 0:1, -1]

    mcts_probs_dst = np.zeros(
        shape=(buffer_length, dst_width * dst_width),
        dtype=mcts_probs_src.dtype,
    )
    mcts_probs_dst = mcts_probs_dst.reshape((buffer_length, dst_width, dst_width))
    mcts_probs_dst[
        :,
        start_width_idx : start_width_idx + src_width,
        start_height_idx : start_height_idx + src_height,
    ] = mcts_probs_src[:].reshape((buffer_length, src_width, src_width))
    mcts_probs_dst = mcts_probs_dst.reshape((buffer_length, dst_width * dst_width))

    f_dst["states"] = states_dst
    f_dst["mcts_probs"] = mcts_probs_dst
    f_dst["rewards"] = f_src["rewards"][...]

    f_src.close()
    f_dst.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gomoku AlphaZero Weights Converter")
    parser.add_argument("--src-weights", default="./data/model-8x8#5.h5", help="源预训练权重存储位置")
    parser.add_argument("--dst-weights", default="./data/model-15x15#5.h5", help="目标预训练权重存储位置")
    parser.add_argument("--src-buffer", default="./data/buffer-8x8#5.h5", help="源经验池存储位置")
    parser.add_argument("--dst-buffer", default="./data/buffer-15x15#5.h5", help="目标经验池存储位置")
    args = parser.parse_args()

    # 小棋盘预训练数据迁移到大棋盘
    convert_pretrained_weights(
        args.src_weights,
        args.dst_weights,
        src_width=8,
        dst_width=15,
        src_height=8,
        dst_height=15,
    )

    convert_pretrained_buffer(
        args.src_buffer,
        args.dst_buffer,
        src_width=8,
        dst_width=15,
        src_height=8,
        dst_height=15,
    )
