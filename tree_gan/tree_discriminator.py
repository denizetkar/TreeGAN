import torch
import torch.nn as nn

from tree_gan.utils import NonTerminal


class TreeDiscriminator(nn.Module):
    def __init__(self, rules_dict, symbol_names, action_offsets, initial_state_func=None, start='start',
                 action_embedding_size=None, hidden_size=None, batch_first=False, rnn_cls=None, rnn_kwargs=None):
        super(TreeDiscriminator, self).__init__()
        self.rules_dict = rules_dict
        self.symbol_names = symbol_names
        self.action_offsets = action_offsets
        self.initial_state_func = initial_state_func
        self.start_id = symbol_names.index(NonTerminal(start))
        num_of_rules = sum(len(rules) for rules in rules_dict.values())
        if action_embedding_size is None:
            action_embedding_size = (num_of_rules - 1) // 4 + 1
        if hidden_size is None:
            hidden_size = action_embedding_size
        self.batch_first = batch_first
        if rnn_cls is None:
            rnn_cls = nn.GRU
            if rnn_kwargs is None:
                rnn_kwargs = dict(num_layers=2, dropout=0.1)
            if self.initial_state_func is None:
                num_layers = rnn_kwargs.get('num_layers', 1)
                num_directions = int(rnn_kwargs.get('bidirectional', False)) + 1

                def func():
                    return torch.zeros((num_layers * num_directions, 1, hidden_size))
                self.initial_state_func = func
        else:
            if rnn_kwargs is None:
                rnn_kwargs = dict()
            assert self.initial_state_func is not None, 'initial_state_func is not known!'
        rnn_kwargs['batch_first'] = batch_first
        input_size = action_embedding_size * 2
        self.rnn = rnn_cls(input_size, hidden_size, **rnn_kwargs)

        self.padding_action = -1
        self.universal_action_offset = 1

        self.action_embeddings = nn.Embedding(num_of_rules + self.universal_action_offset, action_embedding_size,
                                              padding_idx=self.padding_action + self.universal_action_offset)
        self.truth_layer = nn.Linear(hidden_size, 2)
        self.device = None

    def to(self, *args, **kwargs):
        if args and (isinstance(args[0], torch.device) or ('cuda' in args[0]) or ('cpu' in args[0])):
            self.device = args[0]
        elif kwargs and 'device' in kwargs:
            self.device = kwargs['device']

        return super(TreeDiscriminator, self).to(*args, **kwargs)

    def forward(self, actions, parent_actions):
        # TODO: take batch_size into account !!!!
        # input: [seq_len, 1] OR [1, seq_len]
        # output: [seq_len, 1, 2] OR [1, seq_len, 2]
        initial_state = self.initial_state_func().to(self.device)
        action_embeddings = self.action_embeddings(actions + self.universal_action_offset)
        parent_action_embeddings = self.action_embeddings(parent_actions + self.universal_action_offset)

        current_input = torch.cat([action_embeddings, parent_action_embeddings], dim=-1)
        out, _ = self.rnn(current_input, initial_state)

        return torch.log_softmax(self.truth_layer(out), dim=-1)
