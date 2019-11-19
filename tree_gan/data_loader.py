import os
import pickle

from lark import Lark
from torch.utils.data import Dataset

from tree_gan.utils import Enumerator, CustomBNFParser, SimpleTreeActionGetter


class ActionSequenceDataset(Dataset):
    def __init__(self, bnf_path, lark_path, texts_dir, action_getter_path='', action_sequences_dir='', start=None,
                 lang_grammar_start='start'):
        self.texts_dir = texts_dir
        self.action_sequences_dir = action_sequences_dir
        self.start = start
        self.text_filenames = Enumerator(
            [dir_entry.name for dir_entry in os.scandir(texts_dir) if dir_entry.is_file()])
        # Get rule dictionary of the language
        # First check if action getter already exists, if not then parse the language grammar to create it
        if os.path.exists(action_getter_path):
            with open(action_getter_path, 'rb') as f:
                action_getter = pickle.load(f)
        else:
            my_bnf_parser = CustomBNFParser()
            _, rules_dict, symbol_names = my_bnf_parser.parse_file(bnf_path, start=lang_grammar_start)
            action_getter = SimpleTreeActionGetter(rules_dict, symbol_names)
            if action_getter_path:
                with open(action_getter_path, 'wb') as f:
                    pickle.dump(action_getter, f)
        self.action_getter = action_getter

        with open(lark_path) as f:
            self.parser = Lark(f, keep_all_tokens=True, start=lang_grammar_start)

    def index(self, text_filename):
        return self.text_filenames.index(text_filename)

    def __getitem__(self, index):
        # First check if action sequence of parse tree of the text file already exists, if not then calculate it
        text_filename = self.text_filenames[index]
        text_file_path = os.path.join(self.texts_dir, text_filename)
        text_action_sequence_path = os.path.join(self.action_sequences_dir, text_filename + '.pickle')
        if os.path.exists(text_action_sequence_path):
            with open(text_action_sequence_path, 'rb') as f:
                action_sequence = pickle.load(f)
        else:
            with open(text_file_path) as f:
                # Get parse tree of the text file written in the language defined by the given grammar
                text_tree = self.parser.parse(f.read(), start=self.start)
                id_tree = self.action_getter.lark_tree_to_id_tree(text_tree)
                # Get sequence of actions taken by each non-terminal symbol in 'prefix DFS left-to-right' order
                action_sequence = self.action_getter.collect_actions(id_tree)
            if self.action_sequences_dir:
                with open(text_action_sequence_path, 'wb') as f:
                    pickle.dump(action_sequence, f)

        return action_sequence

    def __len__(self):
        return len(self.text_filenames)
