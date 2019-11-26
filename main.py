import os
import time

import torch

from tree_gan import ActionSequenceDataset, TreeGenerator, TreeDiscriminator

START = time.process_time()
reduced_C_dir = os.path.join('data', 'reduced_C_lang')

reduced_C_bnf_path = os.path.join(reduced_C_dir, 'reduced_C_lang.bnf')
reduced_C_lark_path = os.path.join(reduced_C_dir, 'reduced_C_lang.lark')
reduced_C_text_dir = os.path.join(reduced_C_dir, 'text_files')
reduced_C_action_getter_path = os.path.join(reduced_C_dir, 'action_getter.pickle')
reduced_C_action_sequences_dir = os.path.join(reduced_C_dir, 'action_sequence_files')

a_s_dataset = ActionSequenceDataset(reduced_C_bnf_path, reduced_C_lark_path, reduced_C_text_dir,
                                    reduced_C_action_getter_path, reduced_C_action_sequences_dir)
action_sequence = a_s_dataset[a_s_dataset.index('code.c')]
print('ELAPSED TIME (sec): ' + str(time.process_time() - START))
print('---------------------------------')
print(a_s_dataset.action_getter.construct_text(action_sequence))
print('---------------------------------')
tree_generator = TreeGenerator(a_s_dataset.action_getter.rules_dict, a_s_dataset.action_getter.symbol_names,
                               a_s_dataset.action_getter.action_offsets)
tree_discriminator = TreeDiscriminator(a_s_dataset.action_getter.rules_dict, a_s_dataset.action_getter.symbol_names,
                                       a_s_dataset.action_getter.action_offsets)

with torch.no_grad():
    # Generate an action sequence (equivalent to parse tree or text file)
    generated_actions, generated_parent_actions = tree_generator(max_sequence_length=1000)
    truth_log_probs = tree_discriminator(generated_actions, generated_parent_actions)

print(a_s_dataset.action_getter.construct_text_partial(generated_actions.squeeze().numpy().tolist()))
print('---------------------------------')
pass
