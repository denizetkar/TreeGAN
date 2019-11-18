import os
import time

from tree_gan import ActionSequenceDataset

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
pass
