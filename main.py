import os
import pickle
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

    reduced_C_lang_model_path = os.path.join('models', 'reduced_C_lang.model')

    # If exists, load the last checkpoints from the model file path
    if os.path.exists(reduced_C_lang_model_path):
        with open(reduced_C_lang_model_path, 'rb') as f:
            generator_ckp, discriminator_ckp = pickle.load(f)
    else:
        generator_ckp, discriminator_ckp = None, None

    # ------------------HYPER PARAMETERS---------------------
    all_params = dict(
        # HYPER PARAMETERS WITH DEFAULT VALUES: (device, random_seed, initial_generator)
        a_s_dataset=tree_gan.ActionSequenceDataset(reduced_C_bnf_path, reduced_C_lark_path, reduced_C_text_dir,
                                                   reduced_C_action_getter_path, reduced_C_action_sequences_dir),
        generator_ckp=generator_ckp,
        discriminator_ckp=discriminator_ckp,
        generator_kwargs={'action_embedding_size': 128},
        discriminator_kwargs={'action_embedding_size': 128},
        num_data_loader_workers=1,
        max_total_step=400000,  # min number of steps to take during training
        initial_episode_timesteps=5000,  # initial max time steps in one episode
        final_episode_timesteps=10000,  # final max time steps in one episode (MUST NOT EXCEED 'buffer_timestep')
        episode_timesteps_log_order=0,
        gamma=0.99,  # discount factor
        gae_lambda=0.95,  # lambda value for td(lambda) returns
        eps_clip=0.2,  # clip parameter for PPO
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

        random_seed=1234,
        lr=1e-4,
        buffer_timestep=20000,
        lr_decay_order=5,
        k_epochs=5,
        buffer_to_batch_ratio=2,
        optimizer_betas=(0.5, 0.75),
        # PRE-TRAINING HYPER PARAMETERS
        pre_train_epochs=6,
        pre_train_batch_size=2,
        # DISCRIMINATOR TRAINING HYPER PARAMETERS
        discriminator_train_epochs=1,
        discriminator_train_batch_size=2,
        # GAN TRAINING HYPER PARAMETERS
        gan_epochs=10
    )
    # -------------------------------------------------------

    mean_reward, (tree_gen, tree_dis), episode_reward_lists = tree_gan_evaluate(**all_params)

    # Save the current checkpoints to the model file path
    with open(reduced_C_lang_model_path, 'wb') as f:
        pickle.dump((tree_gen.state_dict(), tree_dis.state_dict()), f)

    with torch.no_grad():
        # Generate an action sequence (equivalent to parse tree or text file)
        _, generated_actions, _, _, _ = tree_gen(max_sequence_length=all_params['final_episode_timesteps'])

    a_s_dataset = all_params['a_s_dataset']
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
