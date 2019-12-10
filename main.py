import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import tree_gan
from tree_gan import ActionSequenceDataset, TreeGenerator, TreeDiscriminator

torch.set_default_dtype(torch.float32)


def pre_train_generator(tree_generator, generator_optimizer, a_s_dataloader,
                        pre_train_epochs, pre_train_batch_size, device):
    ce_loss_func = nn.CrossEntropyLoss()
    for _ in range(pre_train_epochs):
        current_batch_size, loss = 0, 0.0
        for action_sequence, parent_action_sequence in a_s_dataloader:
            current_batch_size += 1
            initial_gen_state = tree_generator.rand_initial_state_func().to(device)
            action_sequence = action_sequence.squeeze().to(device)
            parent_action_sequence = parent_action_sequence.squeeze().to(device)
            log_probs, _ = tree_generator.evaluate(initial_gen_state, action_sequence, parent_action_sequence)
            loss += ce_loss_func(log_probs, action_sequence) / pre_train_batch_size
            if current_batch_size == pre_train_batch_size:
                generator_optimizer.zero_grad()
                loss.backward()
                generator_optimizer.step()
                current_batch_size, loss = 0, 0.0
        if current_batch_size != 0:
            generator_optimizer.zero_grad()
            loss.backward()
            generator_optimizer.step()
        del loss


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
            real_actions = real_actions.squeeze().to(device)
            real_parent_actions = real_parent_actions.squeeze().to(device)
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
        if current_batch_size != 0:
            discriminator_optimizer.zero_grad()
            loss.backward()
            discriminator_optimizer.step()
        del loss


def train_generator(generator_optimizer, t_max, tree_generator, tree_generator_old, tree_discriminator, batch_timestep,
                    max_total_step, episode_timesteps, gamma, gae_lambda, eps_clip, buffer_timestep, lr_decay_order,
                    k_epochs):
    lr_scheduler = tree_gan.optim.CosineLogAnnealingLR(generator_optimizer, t_max, eta_min=0.0,
                                                       log_order=lr_decay_order)

    episode_reward_list = []
    episode_memory = tree_gan.learning_utils.ReplayMemory(buffer_timestep + episode_timesteps, (
        'initial_gen_state', 'actions', 'parent_actions', 'log_probs', 'values', 'lt_rewards'))
    buffer_step = total_step = 0
    # Start training loop
    while True:
        episode_steps = min(episode_timesteps, buffer_timestep - buffer_step)
        with torch.no_grad():
            initial_gen_state, actions, parent_actions, log_probs, values = tree_generator_old(episode_steps)
            truth_log_probs = tree_discriminator(actions, parent_actions)
        rewards = torch.exp(truth_log_probs).select(dim=-1, index=int(True))
        lt_rewards = tree_gan.learning_utils.td_lambda_returns(rewards, values, gamma, gae_lambda)
        values = values[:-1]
        buffer_step += actions.nelement()
        total_step += actions.nelement()
        episode_reward_list.append(rewards.mean().item())
        episode_memory.push(initial_gen_state, actions, parent_actions, log_probs, values, lt_rewards)

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

                    first_episode_in_batch = last_episode_in_batch + 1
                    current_batch_steps = 0
            # Copy new weights into old policy:
            tree_generator_old.load_state_dict(tree_generator.state_dict())

            if total_step >= max_total_step:
                break
            buffer_step = 0
            lr_scheduler.step()

    return episode_reward_list


def main():
    reduced_C_dir = os.path.join('data', 'reduced_C_lang')

    reduced_C_bnf_path = os.path.join(reduced_C_dir, 'reduced_C_lang.bnf')
    reduced_C_lark_path = os.path.join(reduced_C_dir, 'reduced_C_lang.lark')
    reduced_C_text_dir = os.path.join(reduced_C_dir, 'text_files')
    reduced_C_action_getter_path = os.path.join(reduced_C_dir, 'action_getter.pickle')
    reduced_C_action_sequences_dir = os.path.join(reduced_C_dir, 'action_sequence_files')

    # ------------------HYPER PARAMETERS---------------------
    # HYPER PARAMETERS WITH DEFAULT VALUES: (device, random_seed, initial_generator)
    max_total_step = 50000       # min number of steps to take during training
    episode_timesteps = 5000     # max time steps in one episode
    gamma = 0.99                 # discount factor
    gae_lambda = 0.95            # lambda value for td(lambda) returns
    eps_clip = 0.2               # clip parameter for PPO
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random_seed = 1234  # TODO: make default value None
    initial_generator = None
    initial_discriminator = None
    lr = 1e-4
    buffer_timestep = 10000
    lr_decay_order = 5
    k_epochs = 5
    buffer_to_batch_ratio = 5
    optimizer_betas = (0.5, 0.75)
    # PRE-TRAINING HYPER PARAMETERS
    pre_train_epochs = 6
    pre_train_batch_size = 3
    # DISCRIMINATOR TRAINING HYPER PARAMETERS
    discriminator_train_epochs = 3
    discriminator_train_batch_size = 4
    # -------------------------------------------------------
    batch_timestep = max(buffer_timestep // buffer_to_batch_ratio, 1)
    t_max = (max_total_step - 1) // buffer_timestep + 1
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    a_s_dataset = ActionSequenceDataset(reduced_C_bnf_path, reduced_C_lark_path, reduced_C_text_dir,
                                        reduced_C_action_getter_path, reduced_C_action_sequences_dir)
    tree_generator = TreeGenerator(a_s_dataset.action_getter).to(device)
    tree_generator_old = TreeGenerator(a_s_dataset.action_getter).to(device)
    tree_discriminator = TreeDiscriminator(a_s_dataset.action_getter).to(device)

    if isinstance(initial_generator, TreeGenerator):
        tree_generator.load_state_dict(initial_generator.state_dict())
    if isinstance(initial_discriminator, TreeDiscriminator):
        tree_discriminator.load_state_dict(initial_discriminator.state_dict())

    generator_optimizer = tree_gan.optim.Ranger(tree_generator.parameters(), lr=lr, betas=optimizer_betas)
    discriminator_optimizer = tree_gan.optim.Ranger(tree_discriminator.parameters(), lr=lr, betas=optimizer_betas)

    a_s_dataloader = DataLoader(a_s_dataset, batch_size=1, shuffle=True, num_workers=4,
                                pin_memory=torch.cuda.is_available())
    pre_train_generator(tree_generator, generator_optimizer, a_s_dataloader,
                        pre_train_epochs, pre_train_batch_size, device)

    tree_generator_old.load_state_dict(tree_generator.state_dict())

    train_discriminator(tree_generator, tree_discriminator, discriminator_optimizer, a_s_dataloader, episode_timesteps,
                        discriminator_train_epochs, discriminator_train_batch_size, device)

    episode_reward_list = train_generator(generator_optimizer, t_max, tree_generator, tree_generator_old,
                                          tree_discriminator, batch_timestep, max_total_step, episode_timesteps, gamma,
                                          gae_lambda, eps_clip, buffer_timestep, lr_decay_order, k_epochs)

    # use every last bit of gradient information (if any left unused)
    discriminator_optimizer.finalize_steps()
    generator_optimizer.finalize_steps()

    with torch.no_grad():
        # Generate an action sequence (equivalent to parse tree or text file)
        _, generated_actions, _, _, _ = tree_generator(max_sequence_length=episode_timesteps)

    action_sequence, _ = a_s_dataset[a_s_dataset.index('code.c')]
    print(a_s_dataset.action_getter.construct_text(action_sequence.numpy().tolist()))
    print('---------------------------------')
    print(a_s_dataset.action_getter.construct_text_partial(generated_actions.squeeze().cpu().numpy().tolist()))
    print('---------------------------------')


if __name__ == '__main__':
    START = time.process_time()
    main()
    print('ELAPSED TIME (sec): ' + str(time.process_time() - START))
    print('---------------------------------')
