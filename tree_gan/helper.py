import random
import time
from collections import namedtuple

import hyperopt
import torch
from hyperopt import pyll


class ReplayMemory(object):

    def __init__(self, capacity, field_names=('state', 'action', 'reward', 'next_state')):
        self._Transition = namedtuple('Transition', field_names)
        self._capacity = capacity
        self.memory = []
        self._position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self._capacity:
            self.memory.append(None)
        self.memory[self._position] = self._Transition(*args)
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()
        self._position = 0


def normalize_tensor(tensor, eps=1e-7):
    std = tensor.std()
    if torch.isnan(std):
        return tensor - tensor.mean()
    return (tensor - tensor.mean()) / (std + eps)


def td_lambda_returns(rewards, state_values, gamma, gae_lambda=0):
    gae = torch.tensor(0.0, device=rewards.device)
    delta = rewards + gamma * state_values[1:] - state_values[:-1]
    td_lambda_targets = delta
    if gae_lambda > 0:
        for t in reversed(range(rewards.size(0))):
            gae = delta[t] + gamma * gae_lambda * gae
            td_lambda_targets[t] = gae + state_values[t]
    return td_lambda_targets


class ModelEvaluator:
    def __init__(self, evaluation_func, prior_params, integer_param_names=None, indexed_param_values=None,
                 invert_loss=False):
        # evaluation_func(params) -> metric, model, other_metrics(?)
        if integer_param_names is None:
            integer_param_names = []
        if indexed_param_values is None:
            indexed_param_values = {}
        integer_param_names = set(integer_param_names)
        integer_param_names.update(indexed_param_values.keys())
        self.evaluation_func = evaluation_func
        self.prior_params = prior_params
        self.integer_param_names = integer_param_names
        self.indexed_param_values = indexed_param_values
        self.invert_loss = invert_loss
        self.best_model = None
        # Keep indexed parameters in best_params as indexed all the time
        self.best_params = None
        self.best_loss = None
        self.best_other_metrics = None
        self.iter_count = 0

    def state_dict(self):
        # Save best parameters with the prior parameters
        best_params = self.best_params.copy()
        best_params.update(self.prior_params)
        return dict(
            integer_param_names=self.integer_param_names,
            indexed_param_values=self.indexed_param_values,
            invert_loss=self.invert_loss,
            best_model=self.best_model,
            best_params=best_params,
            best_loss=self.best_loss,
            best_other_metrics=self.best_other_metrics
        )

    def load_state_dict(self, state_dict):
        for name, value in state_dict.items():
            if hasattr(self, name):
                setattr(self, name, value)

    def __call__(self, params):
        # 'loss' is either None or real number
        loss = params['loss']
        del params['loss']

        all_params = params.copy()
        all_params.update(self.prior_params)
        for p in self.integer_param_names:
            all_params[p] = int(all_params[p])
        # turn index value 'all_params[p]' into the actual intended value in 'self.indexed_param_values[p]'
        for p, vals in self.indexed_param_values.items():
            all_params[p] = vals[all_params[p]]

        # Check if this trial is pre-calculated
        if loss is not None:
            model = other_metrics = None
            eval_time = 0.0
        else:
            start = time.time()
            metric, model, other_metrics = self.evaluation_func(**all_params)
            eval_time = time.time() - start
            loss = -metric if self.invert_loss else metric

        if (self.best_loss is not None and loss < self.best_loss) or self.best_loss is None:
            self.best_model = model
            self.best_params = params
            self.best_loss = loss
            self.best_other_metrics = other_metrics
        self.iter_count += 1
        return {'status': hyperopt.STATUS_OK, 'loss': loss, 'params': params, 'eval_time': eval_time,
                'iter_count': self.iter_count}

    def reset(self):
        self.best_model = None
        self.best_params = None
        self.best_loss = None
        self.best_other_metrics = None
        self.iter_count = 0


class LowLevelModelEvaluator:
    def __init__(self, model_evaluator):
        self.model_evaluator = model_evaluator

    def __call__(self, expr, memo, ctrl):
        pyll_rval = pyll.rec_eval(
            expr,
            memo=memo,
            print_node_on_error=False)
        if 'loss' in ctrl.current_trial['misc']['vals']:
            loss = ctrl.current_trial['misc']['vals']['loss'][0]
        else:
            loss = None
        pyll_rval.update({'loss': loss})
        return self.model_evaluator(pyll_rval)
