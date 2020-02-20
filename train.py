import multiprocessing
import os
import queue
import threading
import time
import argparse

import numpy as np
import tensorflow as tf

from actor import ActorCriticModel
from config import *
from game import Game

'''
based on: https://github.com/tensorflow/models/blob/master/research/a3c_blogpost/a3c_cartpole.py
'''

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description='Gomoku A3C')
parser.add_argument('--resume', action='store_true', help='恢复模型继续训练')
args = parser.parse_args()

class Memory:
    def __init__(self):
        self.color = 0
        self.states = []
        self.actions = []
        self.reward = 0
        self.cnt = 0

    def store(self, state, action):
        self.states.append(state)
        self.actions.append(action)
        self.cnt += 1

    def set_reward(self, reward):
        self.reward = reward

    def clear(self):
        self.color = 0
        self.states.clear()
        self.actions.clear()
        self.reward = 0
        self.cnt = 0


class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0
    save_lock = threading.Lock()

    def __init__(self,
                 global_model,
                 opt,
                 result_queue,
                 idx,
                 gamma=0.99,
                 ):
        super().__init__()
        self.result_queue = result_queue
        self.global_model = global_model
        self.opt = opt
        self.local_model = ActorCriticModel()
        self.local_model.build(input_shape=(None, width, height, 1))
        self.local_model.set_weights(self.global_model.get_weights())
        self.worker_idx = idx
        self.gamma = gamma
        self.game = Game()
        self.mean_loss = tf.keras.metrics.Mean(name='train_loss')

    def run(self):
        mems = [Memory(), Memory()]
        game = self.game
        while Worker.global_episode < MAX_EPISODE:
            episode = Worker.global_episode
            mems[0].clear()
            mems[1].clear()
            mems[0].color = BLACK
            mems[1].color = WHITE
            step = 0
            flag = True
            while flag:
                id = step % 2
                color = game.curr_value
                curr_state = color * game.board
                curr_state = np.expand_dims(curr_state, axis=-1)
                curr_state = np.expand_dims(curr_state, axis=0)

                logits, _ = self.local_model(
                    tf.convert_to_tensor(curr_state, dtype=tf.float32))
                mask = tf.reshape(curr_state == 0, (width*height, ))
                logits = tf.where(mask, logits, -np.inf)
                probs = tf.nn.softmax(logits)

                action = np.random.choice(height*width, p=probs.numpy()[0])
                x, y = action // width, action % width

                status = game.play(x, y)

                mems[id].store(curr_state, action)
                if status != NOTHING:
                    reward = status * color
                    mems[id].set_reward(reward)
                    mems[1-id].set_reward(-reward)
                    break
                step += 1

            for memory in mems:
                # Get discounted rewards
                reward_sum = 0.
                discounted_rewards = []
                rewards = [0 for _ in range(memory.cnt-1)]
                rewards.insert(0, memory.reward)    # reverse buffer r
                for reward in rewards:
                    reward_sum = reward + self.gamma * reward_sum
                    discounted_rewards.append(reward_sum)
                discounted_rewards.reverse()

                with tf.GradientTape() as tape:
                    logits, values = self.local_model(tf.convert_to_tensor(
                        np.array(memory.states).reshape(memory.cnt, width, height, 1), dtype=tf.float32))

                    # Get our advantages
                    advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                                                     dtype=tf.float32) - values

                    # Value loss
                    value_loss = advantage ** 2

                    # Calculate our policy loss
                    policy = tf.nn.softmax(logits)
                    entropy = tf.keras.losses.categorical_crossentropy(
                        y_true=policy, y_pred=logits, from_logits=True)
                    policy_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=memory.actions,
                                                                                  y_pred=logits, from_logits=True)
                    policy_loss *= tf.stop_gradient(advantage)
                    policy_loss -= 0.01 * entropy
                    total_loss = tf.reduce_mean(
                        (0.5 * value_loss + policy_loss))

                # Calculate local gradients
                grads = tape.gradient(
                    total_loss, self.local_model.trainable_weights)
                # Push local gradients to global model
                self.opt.apply_gradients(
                    zip(grads, self.global_model.trainable_weights))
                # Update local model with new weights
                self.local_model.set_weights(self.global_model.get_weights())

                self.mean_loss(total_loss)
                print('Episode: {:5d}, Worker: {:2d}, Color: {:5s}, Winner: {:5s}, Step: {:3d}, Loss: {}  '.format(
                    episode+1,
                    self.worker_idx,
                    COLOR[memory.color],
                    COLOR[memory.reward * memory.color],
                    memory.cnt,
                    total_loss.numpy()), end='\r')
                memory.clear()

            if (episode + 1) % 50 == 0:
                print('Episode: {:5d}, Loss: {}  '.format(
                    episode+1,
                    self.mean_loss.result()
                ))
                self.mean_loss.reset_states()
                with Worker.save_lock:
                    self.global_model.save_weights(MODEL_FILE)

            self.result_queue.put(total_loss.numpy())
            with Worker.save_lock:
                Worker.global_episode += 1
        self.result_queue.put(None)


def train():
    global_model = ActorCriticModel()
    learning_rate = 1e-4
    gamma = 0.99
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    global_model.build(input_shape=(None, width, height, 1))
    if args.resume:
        global_model.load_weights(MODEL_FILE)

    res_queue = queue.Queue()

    workers = [Worker(global_model,
                      optimizer,
                      res_queue,
                      i,
                      gamma=gamma) for i in range(multiprocessing.cpu_count())]

    for i, worker in enumerate(workers):
        print("Starting worker {}".format(i))
        worker.setDaemon(True)
        worker.start()

    while True:
        try:
            reward = res_queue.get()
            if reward is None:
                break
            time.sleep(0.1)
        except KeyboardInterrupt:
            break
    # [w.join() for w in workers]


if __name__ == '__main__':
    train()
