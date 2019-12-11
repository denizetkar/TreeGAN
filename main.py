import os
import time

import torch

import tree_gan
from tree_gan.learning_utils import tree_gan_evaluate

torch.set_default_dtype(torch.float32)


def main():
    reduced_C_dir = os.path.join('data', 'reduced_C_lang')

    reduced_C_bnf_path = os.path.join(reduced_C_dir, 'reduced_C_lang.bnf')
    reduced_C_lark_path = os.path.join(reduced_C_dir, 'reduced_C_lang.lark')
    reduced_C_text_dir = os.path.join(reduced_C_dir, 'text_files')
    reduced_C_action_getter_path = os.path.join(reduced_C_dir, 'action_getter.pickle')
    reduced_C_action_sequences_dir = os.path.join(reduced_C_dir, 'action_sequence_files')

    # ------------------HYPER PARAMETERS---------------------
    # HYPER PARAMETERS WITH DEFAULT VALUES: (device, random_seed, initial_generator)
    a_s_dataset = tree_gan.ActionSequenceDataset(reduced_C_bnf_path, reduced_C_lark_path, reduced_C_text_dir,
                                                 reduced_C_action_getter_path, reduced_C_action_sequences_dir)
    num_data_loader_workers = 1
    max_total_step = 50000       # min number of steps to take during training
    initial_episode_timesteps = 5000  # initial max time steps in one episode
    final_episode_timesteps = 10000   # final max time steps in one episode (MUST NOT EXCEED 'buffer_timestep')
    episode_timesteps_log_order = 0
    gamma = 0.99                 # discount factor
    gae_lambda = 0.95            # lambda value for td(lambda) returns
    eps_clip = 0.2               # clip parameter for PPO
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random_seed = 1234
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
    pre_train_batch_size = 2
    # DISCRIMINATOR TRAINING HYPER PARAMETERS
    discriminator_train_epochs = 1
    discriminator_train_batch_size = 2
    # GAN TRAINING HYPER PARAMETERS
    gan_epochs = 1
    # -------------------------------------------------------

    mean_reward, (tree_generator, _), episode_reward_lists = tree_gan_evaluate(a_s_dataset, max_total_step,
                                                                               initial_episode_timesteps,
                                                                               final_episode_timesteps,
                                                                               episode_timesteps_log_order, gamma,
                                                                               gae_lambda, eps_clip, lr,
                                                                               buffer_timestep, lr_decay_order,
                                                                               k_epochs, buffer_to_batch_ratio,
                                                                               optimizer_betas, pre_train_epochs,
                                                                               pre_train_batch_size,
                                                                               discriminator_train_epochs,
                                                                               discriminator_train_batch_size,
                                                                               gan_epochs, num_data_loader_workers,
                                                                               device, random_seed, initial_generator,
                                                                               initial_discriminator)

    with torch.no_grad():
        # Generate an action sequence (equivalent to parse tree or text file)
        _, generated_actions, _, _, _ = tree_generator(max_sequence_length=final_episode_timesteps)

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
