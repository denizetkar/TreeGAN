import torch
import torch.nn as nn

from tree_gan import learning_utils
from tree_gan import parse_utils


class TreeDiscriminator(nn.Module):
    """
    Effectively the batch size of the actions and parent_actions inputs are both 1.
    """
    def __init__(self, action_getter, initial_state_func=None, start='start',
                 action_embedding_size=None, hidden_size=None, batch_first=False, rnn_cls=None, rnn_kwargs=None):
        super().__init__()
        self.rules_dict = action_getter.rules_dict
        self.symbol_names = action_getter.symbol_names
        self.action_offsets = action_getter.action_offsets
        self.initial_state_func = initial_state_func
        self.start_id = self.symbol_names.index(parse_utils.NonTerminal(start))
        num_of_rules = sum(len(rules) for rules in self.rules_dict.values())
        if action_embedding_size is None:
            action_embedding_size = (num_of_rules - 1) // 4 + 1
        if hidden_size is None:
            hidden_size = action_embedding_size
        self.batch_first = batch_first
        if rnn_cls is None:
            rnn_cls = nn.GRU
            if rnn_kwargs is None:
                rnn_kwargs = dict(num_layers=2, dropout=0.0)
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

        self.action_embeddings = nn.Embedding(num_of_rules + learning_utils.UNIVERSAL_ACTION_OFFSET,
                                              action_embedding_size,
                                              padding_idx=learning_utils.PADDING_ACTION +
                                                          learning_utils.UNIVERSAL_ACTION_OFFSET)
        num_directions = int(getattr(self.rnn, 'bidirectional', False)) + 1
        self.truth_layer = nn.Linear(num_directions * hidden_size, 2)
        self.device = None

    def cpu(self):
        res = super().cpu()
        self.device = 'cpu'
        return res

    def cuda(self, device=None):
        res = super().cuda(device=device)
        self.device = 'cuda' if device is None else device
        return res

    def to(self, *args, **kwargs):
        res = super().to(*args, **kwargs)

        if args and (isinstance(args[0], torch.device) or ('cuda' in args[0]) or ('cpu' in args[0])):
            self.device = args[0]
        elif kwargs and 'device' in kwargs:
            self.device = kwargs['device']

        return res

    def forward(self, actions, parent_actions):
        # input: (seq_len, )
        # output: (seq_len, 2)
        batch_dim_index = 1 - int(self.batch_first)
        actions = actions.unsqueeze(batch_dim_index)
        parent_actions = parent_actions.unsqueeze(batch_dim_index)
        initial_state = self.initial_state_func().to(self.device, non_blocking=True)
        action_embeddings = self.action_embeddings(actions + learning_utils.UNIVERSAL_ACTION_OFFSET)
        parent_action_embeddings = self.action_embeddings(parent_actions + learning_utils.UNIVERSAL_ACTION_OFFSET)

        current_input = torch.cat([action_embeddings, parent_action_embeddings], dim=-1)
        out, _ = self.rnn(current_input, initial_state)

        return torch.log_softmax(self.truth_layer(out), dim=-1).squeeze(batch_dim_index)
