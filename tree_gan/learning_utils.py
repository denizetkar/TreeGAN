import random
import time
from collections import namedtuple

import hyperopt
import numpy as np
import torch
from hyperopt import pyll
from torch import nn as nn
from torch.utils.data import DataLoader

from tree_gan import optim
from tree_gan import tree_discriminator
from tree_gan import tree_generator

PADDING_ACTION = -1
UNIVERSAL_ACTION_OFFSET = 1


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
            start = time.process_time()
            metric, model, other_metrics = self.evaluation_func(**all_params)
            eval_time = time.process_time() - start
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


def pre_train_generator(tree_generator, generator_optimizer, a_s_dataloader,
                        pre_train_epochs, pre_train_batch_size, device):
    ce_loss_func = nn.CrossEntropyLoss()
    for _ in range(pre_train_epochs):
        current_batch_size, loss = 0, 0.0
        for action_sequence, parent_action_sequence in a_s_dataloader:
            current_batch_size += 1
            initial_gen_state = tree_generator.rand_initial_state_func().to(device, non_blocking=True)
            action_sequence = action_sequence.squeeze().to(device, non_blocking=True)
            parent_action_sequence = parent_action_sequence.squeeze().to(device, non_blocking=True)
            log_probs, values = tree_generator.evaluate(initial_gen_state, action_sequence, parent_action_sequence)
            del values
            loss += ce_loss_func(log_probs, action_sequence) / pre_train_batch_size
            if current_batch_size == pre_train_batch_size:
                generator_optimizer.zero_grad()
                loss.backward()
                generator_optimizer.step()
                current_batch_size, loss = 0, 0.0
                del log_probs
        if current_batch_size != 0:
            generator_optimizer.zero_grad()
            loss.backward()
            generator_optimizer.step()
            del log_probs


def train_discriminator(tree_generator, tree_discriminator, discriminator_optimizer, a_s_dataloader, episode_timesteps,
                        discriminator_train_epochs, discriminator_train_batch_size, device):
    ce_loss_func = nn.CrossEntropyLoss()
    real_label = torch.tensor(1, device=device)
    fake_label = torch.tensor(0, device=device)
    for _ in range(discriminator_train_epochs):
        current_batch_size, loss = 0, 0.0
        for real_actions, real_parent_actions in a_s_dataloader:
            current_batch_size += 1
            # Calculate the loss for this real sequence
            # Action sequences arrive with batch dimension of size 1 from the data loader (squeeze it!):
            real_actions = real_actions.squeeze().to(device, non_blocking=True)
            real_parent_actions = real_parent_actions.squeeze().to(device, non_blocking=True)
            truth_log_probs = tree_discriminator(real_actions, real_parent_actions)
            loss += ce_loss_func(truth_log_probs,
                                 real_label.expand(len(truth_log_probs))) / discriminator_train_batch_size
            # 1 fake sequence for each real sequence
            with torch.no_grad():
                _, fake_actions, fake_parent_actions, _, _ = tree_generator(max_sequence_length=episode_timesteps)
            truth_log_probs = tree_discriminator(fake_actions, fake_parent_actions)
            loss += ce_loss_func(truth_log_probs,
                                 fake_label.expand(len(truth_log_probs))) / discriminator_train_batch_size
            if current_batch_size == discriminator_train_batch_size:
                discriminator_optimizer.zero_grad()
                loss.backward()
                discriminator_optimizer.step()
                current_batch_size, loss = 0, 0.0
                del truth_log_probs
        if current_batch_size != 0:
            discriminator_optimizer.zero_grad()
            loss.backward()
            discriminator_optimizer.step()
            del truth_log_probs


def train_generator(generator_optimizer, t_max, tree_generator, tree_generator_old, tree_discriminator, batch_timestep,
                    max_total_step, episode_timesteps, gamma, gae_lambda, eps_clip, buffer_timestep, lr_decay_order,
                    k_epochs):
    lr_scheduler = optim.CosineLogAnnealingLR(generator_optimizer, t_max, eta_min=0.0, log_order=lr_decay_order)

    episode_reward_list = []
    episode_memory = ReplayMemory(buffer_timestep + episode_timesteps, (
        'initial_gen_state', 'actions', 'parent_actions', 'log_probs', 'values', 'lt_rewards'))
    buffer_step = total_step = 0
    # Start training loop
    while True:
        episode_steps = min(episode_timesteps, buffer_timestep - buffer_step)
        with torch.no_grad():
            initial_gen_state, actions, parent_actions, log_probs, values = tree_generator_old(episode_steps)
            truth_log_probs = tree_discriminator(actions, parent_actions)
        rewards = torch.exp(truth_log_probs).select(dim=-1, index=int(True))
        # Add a penalty for long action sequences
        rewards = rewards - 0.4
        lt_rewards = td_lambda_returns(rewards, values, gamma, gae_lambda)
        values = values[:-1]
        buffer_step += actions.nelement()
        total_step += actions.nelement()
        episode_reward_list.append(rewards.mean().item())
        episode_memory.push(initial_gen_state, actions, parent_actions, log_probs, values, lt_rewards)
        del initial_gen_state, actions, parent_actions, log_probs, values, lt_rewards, truth_log_probs, rewards

        if buffer_step >= buffer_timestep:
            old_initial_gen_state_list, old_actions_list, old_parent_actions_list, old_log_probs_list, \
                old_values_list, old_lt_rewards_list = zip(*episode_memory.memory)
            episode_memory.clear()
            # Convert all list of torch.Tensor's into np.array of Tensor's to facilitate easy indexing
            old_initial_gen_state_list = np.array(old_initial_gen_state_list, dtype=torch.Tensor)
            old_actions_list = np.array(old_actions_list, dtype=torch.Tensor)
            old_parent_actions_list = np.array(old_parent_actions_list, dtype=torch.Tensor)
            old_log_probs_list = np.array(old_log_probs_list, dtype=torch.Tensor)
            old_values_list = np.array(old_values_list, dtype=torch.Tensor)
            old_lt_rewards_list = np.array(old_lt_rewards_list, dtype=torch.Tensor)

            # Optimize policy for K epochs:
            number_of_episodes = len(old_actions_list)
            shuffled_episode_indexes = list(range(number_of_episodes))
            for _ in range(k_epochs):
                random.shuffle(shuffled_episode_indexes)
                current_batch_steps = first_episode_in_batch = 0
                # Perform update for each batch with 'batch_timestep' steps
                for last_episode_in_batch, episode_index in enumerate(shuffled_episode_indexes):
                    current_batch_steps += old_actions_list[episode_index].nelement()
                    if current_batch_steps < batch_timestep and last_episode_in_batch < number_of_episodes - 1:
                        continue
                    episode_batch_indexes = shuffled_episode_indexes[first_episode_in_batch:(last_episode_in_batch + 1)]
                    policy_loss, entropy_loss, value_loss = tree_generator.ppo_losses(
                        old_initial_gen_state_list[episode_batch_indexes],
                        old_actions_list[episode_batch_indexes],
                        old_parent_actions_list[episode_batch_indexes],
                        old_log_probs_list[episode_batch_indexes],
                        old_values_list[episode_batch_indexes],
                        old_lt_rewards_list[episode_batch_indexes],
                        eps_clip=eps_clip)

                    if policy_loss is not None:
                        loss = policy_loss.mean() + 0.001 * entropy_loss.mean() + 0.5 * value_loss.mean()
                        # take gradient step
                        generator_optimizer.zero_grad()
                        loss.backward()
                        generator_optimizer.step()
                        del loss, policy_loss, entropy_loss, value_loss

                    first_episode_in_batch = last_episode_in_batch + 1
                    current_batch_steps = 0
            del old_initial_gen_state_list, old_actions_list, old_parent_actions_list, \
                old_log_probs_list, old_values_list, old_lt_rewards_list
            # Copy new weights into old policy:
            tree_generator_old.load_state_dict(tree_generator.state_dict())

            if total_step >= max_total_step:
                break
            buffer_step = 0
            lr_scheduler.step()

    return episode_reward_list


def tree_gan_evaluate(a_s_dataset, max_total_step, initial_episode_timesteps, final_episode_timesteps,
                      episode_timesteps_log_order, gamma, gae_lambda, eps_clip, lr, buffer_timestep, lr_decay_order,
                      k_epochs, buffer_to_batch_ratio, optimizer_betas, pre_train_epochs, pre_train_batch_size,
                      discriminator_train_epochs, discriminator_train_batch_size, gan_epochs, num_data_loader_workers=1,
                      device=torch.device('cpu'), random_seed=None, generator_ckp=None, discriminator_ckp=None,
                      generator_kwargs=None, discriminator_kwargs=None):
    generator_kwargs = {} if generator_kwargs is None else generator_kwargs
    discriminator_kwargs = {} if discriminator_kwargs is None else discriminator_kwargs
    batch_timestep = max(buffer_timestep // buffer_to_batch_ratio, 1)
    t_max = (max_total_step - 1) // buffer_timestep + 1
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    tree_gen = tree_generator.TreeGenerator(a_s_dataset.action_getter, **generator_kwargs).to(
        device, non_blocking=True)
    tree_gen_old = tree_generator.TreeGenerator(a_s_dataset.action_getter, **generator_kwargs).to(
        device, non_blocking=True)
    tree_dis = tree_discriminator.TreeDiscriminator(a_s_dataset.action_getter, **discriminator_kwargs).to(
        device, non_blocking=True)

    if generator_ckp is not None:
        tree_gen.load_state_dict(generator_ckp)
    if discriminator_ckp is not None:
        tree_dis.load_state_dict(discriminator_ckp)

    generator_optimizer = optim.Ranger(tree_gen.parameters(), lr=lr, betas=optimizer_betas)
    discriminator_optimizer = optim.Ranger(tree_dis.parameters(), lr=lr, betas=optimizer_betas)

    a_s_dataloader = DataLoader(a_s_dataset, batch_size=1, shuffle=True, num_workers=num_data_loader_workers,
                                pin_memory=('cpu' not in device.type))
    pre_train_generator(tree_gen, generator_optimizer, a_s_dataloader,
                        pre_train_epochs, pre_train_batch_size, device)

    tree_gen_old.load_state_dict(tree_gen.state_dict())

    if gan_epochs > 1:
        cos_log_scale = optim.CosineLogAnnealingScale(gan_epochs - 1, episode_timesteps_log_order)
        get_ep_tstep_scale = cos_log_scale.get_scale
    else:
        get_ep_tstep_scale = lambda epoch: 0
    episode_reward_lists = []
    # Run main TreeGAN loop
    for epoch in range(gan_epochs):
        episode_timesteps = int(final_episode_timesteps + (
                initial_episode_timesteps - final_episode_timesteps) * get_ep_tstep_scale(epoch))

        episode_reward_list = train_generator(generator_optimizer, t_max, tree_gen, tree_gen_old,
                                              tree_dis, batch_timestep, max_total_step, episode_timesteps,
                                              gamma, gae_lambda, eps_clip, buffer_timestep, lr_decay_order, k_epochs)
        episode_reward_lists.extend(episode_reward_list)

        train_discriminator(tree_gen, tree_dis, discriminator_optimizer, a_s_dataloader,
                            episode_timesteps, discriminator_train_epochs, discriminator_train_batch_size, device)

    # use every last bit of gradient information (if any left unused)
    generator_optimizer.finalize_steps()
    discriminator_optimizer.finalize_steps()

    # Evaluate the quality of action sequences produced by the generator (discriminator perplexity)
    episode_count, total_step, mean_reward = 0, 0, 0.0
    while total_step < final_episode_timesteps:
        with torch.no_grad():
            _, fake_actions, fake_parent_actions, _, _ = tree_gen(
                max_sequence_length=(final_episode_timesteps - total_step))
            truth_log_probs = tree_dis(fake_actions, fake_parent_actions)
        episode_count += 1
        total_step += fake_actions.nelement()
        rewards = torch.exp(truth_log_probs).select(dim=-1, index=int(True))
        mean_reward = mean_reward + (rewards.mean().item() - mean_reward) / episode_count

    return mean_reward, (tree_gen, tree_dis), episode_reward_lists
