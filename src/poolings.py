import torch
from torch.nn import functional
from torch.nn import Conv1d

from utils_model import get_last_hidden_state


class PoolingStrategy:
    @staticmethod
    def mean_pooling(hidden_states):
        last_hidden_state = get_last_hidden_state(hidden_states)
        return torch.mean(last_hidden_state, dim=0).cpu().detach().numpy()

    @staticmethod
    def max_pooling(hidden_states):
        last_hidden_state = get_last_hidden_state(hidden_states)
        return torch.max(last_hidden_state, dim=0).values.cpu().detach().numpy()

    @staticmethod
    def mean_max_pooling(hidden_states):
        last_hidden_state = get_last_hidden_state(hidden_states)

        mean_pooling_embeddings = torch.mean(last_hidden_state, dim=0)
        max_pooling_embeddings = torch.max(last_hidden_state, dim=0).values

        return torch.cat((mean_pooling_embeddings, max_pooling_embeddings),
                         dim=0).cpu().detach().numpy()  # TODO: check about dim 0

    @staticmethod
    def concatenate_pooling(hidden_states):
        last_four_layers = [hidden_states[:, i] for i in (-1, -2, -3, -4)]
        cat_hidden_states = torch.cat(last_four_layers, -1)
        cat_hidden_states = torch.reshape(torch.mean(cat_hidden_states, dim=0), (1, -1))

        return cat_hidden_states.cpu().detach().numpy()

    @staticmethod
    def last_four_sum_pooling(hidden_states):
        last_four_states = torch.sum(hidden_states[:, -4:], dim=1)
        return torch.mean(last_four_states, dim=0).cpu().detach().numpy()

    @staticmethod
    def last_two_sum_pooling(hidden_states):
        last_two_states = torch.sum(hidden_states[:, -2:], dim=1)
        return torch.mean(last_two_states, dim=0).cpu().detach().numpy()

    @staticmethod
    def conv_1d_pooling(hidden_states):
        last_hidden_state = get_last_hidden_state(hidden_states)
        last_hidden_state.unsqueeze_(0)

        last_hidden_state = last_hidden_state.cpu().detach()

        cnn1 = Conv1d(768, 256, kernel_size=2, padding=1)
        cnn2 = Conv1d(256, 1, kernel_size=2, padding=1)

        last_hidden_state = last_hidden_state.permute(0, 2, 1)
        cnn_embeddings = functional.relu(cnn1(last_hidden_state))

        return torch.mean(cnn2(cnn_embeddings)[0]).detach().numpy()