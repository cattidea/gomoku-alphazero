import copy
import numpy as np
cimport numpy as np
cimport cython

cdef double MAX_VALUE = 9999.


cdef class Node(object):
    cdef public object parent, children
    cdef public int n_visits
    cdef double _Q, _u, _P

    def __cinit__(self, parent, prior_p):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self._Q = 0.
        self._u = 0.
        self._P = prior_p

    cdef void expand(self, object action_props):
        """ 使用各个 action 以及对应的 probability 扩展下一层结点 """
        for action, prob in action_props:
            if action not in self.children:
                self.children[action] = Node(self, prob)

    cdef select(self, double c_puct):
        """ 获取值最大的 action 及相应结点 """
        cdef int max_action = -1
        cdef double max_value = -MAX_VALUE
        cdef Node node
        cdef double value
        for action in self.children:
            node = self.children[action]
            value = node.get_value(c_puct)
            if value > max_value:
                max_value = value
                max_action = action
        return max_action, self.children[max_action]

    cdef void update(self, double leaf_value):
        """ 根据叶子值更新 """
        self.n_visits += 1
        self._Q += 1.0*(leaf_value - self._Q) / self.n_visits

    cdef void update_recursive(self, double leaf_value):
        """ 递归地向上更新 """
        cdef Node parent = self.parent
        if parent is not None:
            parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    cdef double get_value(self, double c_puct):
        """ 获取当前值 """
        self._u = (c_puct * self._P * np.sqrt(self.parent.n_visits) / (1 + self.n_visits))
        return self._Q + self._u

    cdef bint is_leaf(self):
        """ 判断是否为叶子结点 """
        return self.children == {}

    cdef bint is_root(self):
        """ 判断是否为根结点 """
        return self.parent is None


cdef class MCTS(object):
    """ Monte Carlo Tree Search Algorithm """
    cdef Node _root
    cdef object _policy
    cdef double _c_puct
    cdef int _n_playout
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self._root = Node(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    cdef void _playout(self, object state):
        cdef object leaf_value
        cdef int player
        cdef Node node = self._root
        while not node.is_leaf():
            assert len(state.availables) == len(node.children)
            action, node = node.select(self._c_puct)
            state.move_to(loc=action)

        player = state.curr_player
        action_probs, leaf_value = self._policy(state)
        is_end, winner = state.game_end()
        if not is_end:
            node.expand(action_probs)
        if leaf_value is None:
            leaf_value = self._evaluate_rollout(state)
        elif is_end:
            leaf_value = winner * player
        node.update_recursive(-leaf_value)

    cdef int _evaluate_rollout(self, object state, limit=1000):
        """ 根据当前状态反复随机落子，获取当前局结束时的 Reward """
        cdef int player = state.curr_player
        for i in range(limit):
            is_end, winner = state.game_end()
            if is_end:
                break
            availables = state.availables
            probs = np.random.rand(availables.shape[0])
            max_action = availables[np.argmax(probs)]
            state.move_to(loc=max_action)
        else:
            print("warn: rollout reached move limit")
        return winner * player

    def get_move(self, state):
        """ 根据当前局势进行推演，根据结果选取访问次数最高的 action
        如果访问次数相同则用 value 排序 """
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        cdef int max_action = -1
        cdef Node node
        cdef int max_visits = 0
        cdef int n_visits
        cdef double max_value = -MAX_VALUE
        cdef double value
        for action in self._root.children:
            node = self._root.children[action]
            n_visits = node.n_visits
            if n_visits > max_visits:
                max_action = action
                max_visits = n_visits
                max_value = -MAX_VALUE
            elif n_visits == max_visits:
                value = node.get_value(self._c_puct)
                if value > max_value:
                    max_action = action
                    max_value = value
        return max_action

    def get_move_probs(self, state, temp=1e-3):
        """ 根据当前局势进行推演，根据结果获得 action 以及其对应的 probability """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        act_visits = [(act, node.n_visits)
                      for act, node in self._root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """ 根据 action 更新至子结点，如果 action 为 -1，则重置搜索树 """
        if last_move in self._root.children:
            self._root = self._root.children[last_move]
            self._root.parent = None
        else:
            self._root = Node(None, 1.0)

cdef softmax(np.ndarray x):
    cdef np.ndarray probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs
