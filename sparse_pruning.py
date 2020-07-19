import torch


class SparsePruning:
    def __init__(self, model, prune_layers, training_steps, t0=0, initial_sparsity=0, final_sparsity=75, delta=2):
        self._to_prune = self._pruning_layers(model, prune_layers)
        self._st = initial_sparsity if t0 != 0 else 0
        self._si = initial_sparsity
        self._sf = final_sparsity
        self._delta = delta
        self._t = t0
        self._t0 = t0
        self._n = training_steps

    def update_sparcity(self):
        self._t += 1
        if self._t < self._t0:
            return
        self._st = self._sf + (self._si - self._sf)*((1 - ((self._t - self._t0)/(self._n * self._delta)))**3)

    def _pruning_layers(self, model, prune_layers):
        layers = set()
        for prune_layer in prune_layers:
            for layer_name, parameters in model.named_parameters():
                if prune_layer in layer_name:
                    layers.add(layer_name)
        return layers

    def _is_prune_layer(self, layer_name):
        return layer_name in self._to_prune

    def __call__(self, module, module_in):
        for layer_name, parameters in module.named_parameters():
            if not self._is_prune_layer(layer_name):
                continue
            bar = parameters.view(-1).topk(8)[0].min()
            mask = torch.where(parameters < bar)
            with torch.no_grad():
                parameters[mask] = 0


