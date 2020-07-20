import os

import torch
import matplotlib.pyplot as plt
import numpy as np


class SparsePruning:
    """
    this class performs Sparse-pruning according to
    To prune, or not to prune: exploring the efficacy of pruning for model compression, Michael H. Zhu, Suyog Gupta
    https://arxiv.org/abs/1710.01878
    """
    def __init__(self, model, prune_layers, training_steps, epoch_per_step, t0=0, initial_sparsity=0, final_sparsity=75):
        """
        :param model: any Pytorch module
        :param prune_layers: list or set of prefixes of the layers to be pruned
        :param training_steps: total number of training steps
        :param epoch_per_step: number of training epochs for each step
        :param t0: number of pre-training steps before pruning start
        :param initial_sparsity: first sparsity ratio to apply(0-100)
        :param final_sparsity: target sparsity ratio to apply(0-100)
        """
        # get pruning layers
        self._to_prune = self._pruning_layers(model, prune_layers)
        self._model = model
        # set variables according to the paper's notation
        self._st = initial_sparsity if t0 == 0 else 0
        self._si = initial_sparsity
        self._sf = final_sparsity
        # self._delta = training_steps / num_iter
        self._training_steps = training_steps

        self._t = 0
        self._t0 = t0
        self._n = training_steps * epoch_per_step
        self._delta = 1 / epoch_per_step
        self._sparse_vec = []
        self._update_masks()

        plt.style.use('ggplot')

    @property
    def masks(self):
        return self._masks

    @property
    def curr_sparsity(self):
        return self._st

    def _update_masks(self):
        """
        set a mask for each of the pruned layers
        maskes are being updated at the beginning of each training step
        """
        masks = {}
        for layer_name, parameters in self._model.named_parameters():
            # if the layer shouldn't be pruned -> skip
            if not self._is_prune_layer(layer_name):
                continue
            # calculate how many parameters to prune form the layer according to s_t (st is a percentage - 0-100)
            st = int(parameters.view(-1).shape[0] * self._st / 100)
            # calculate masking bar according to the the weight's magnitude

            bar = parameters.abs().view(-1).topk(parameters.view(-1).shape[0] - st)[0].min()
            # set mask
            mask_positive_indices = torch.where(parameters.abs() >= bar)
            mask = torch.zeros(parameters.shape).to(self._model.device)
            mask[mask_positive_indices] = 1
            masks[layer_name] = mask
        self._masks = masks

    def update_sparsity(self):
        """
        get next sparsity value
        s_t = s_f - (s_i - s_f) * (1 - (t-t_0)/(n*delta))^3
        """
        self._t += 1
        if self._t == self._t0:
            self._st = self._si
        elif self._t > self._t0:
            self._st = self._sf + (self._si - self._sf)*((1 - ((self._t - self._t0)/(self._n * self._delta)))**3)
        self._sparse_vec.append((self._t, self._st))
        self._update_masks()

    def plot_sparsity(self, plot_dir="."):
        """
        plot Sparsity as a function of training steps
        """
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        # create a variable for the line so we can later update it
        plt.plot([x[0] for x in self._sparse_vec], [100 - x[1] for x in self._sparse_vec], '-o', alpha=0.8)
        # update plot label/title
        plt.ylim([-1, 110])
        plt.xlim([-1, self._training_steps + 1])
        plt.ylabel('NNZ')
        plt.xlabel('Training Step')
        plt.title('Non Zero Values')
        plt.savefig(os.path.join(plot_dir, "sparsity_graph.png"))
        plt.show()
        plt.pause(0.05)

    def _pruning_layers(self, model, prune_layers):
        """
        search for layers to be pruned according to input prefixes
        """
        layers = set()
        for prune_layer in prune_layers:
            for layer_name, parameters in model.named_parameters():
                if prune_layer in layer_name:
                    layers.add(layer_name)
        return layers

    def _is_prune_layer(self, layer_name):
        return layer_name in self._to_prune

    def __call__(self, module, module_in):
        """
        pruning function
        """
        # iterate models layers
        for layer_name, parameters in module.named_parameters():
            # if the layer shouldn't be pruned -> skip
            if not self._is_prune_layer(layer_name):
                continue
            # apply mask to layer's weights
            module.state_dict()[layer_name].copy_(torch.mul(parameters, self._masks[layer_name]))


if __name__ == '__main__':
    trainng_steps = 1000
    epochs_per_train_step = 10
    pruning = SparsePruning(None, set(), trainng_steps, epochs_per_train_step,
                            t0=20, initial_sparsity=0,
                            final_sparsity=95)
    for i in range(trainng_steps):
        pruning.update_sparsity()


