import torch

from mcts_training_tools import set_logger, load_params, load_datasets, load_model, set_trainer, train
from sparse_pruning import SparsePruning
import json


def run():
    set_logger()
    user_params = json.loads(open("hyper_params.json", "rt", encoding="utf-8").read())
    model_args, data_args, training_args = load_params(user_params)
    train_dataset, eval_dataset, test_dataset = load_datasets(data_args, training_args)
    model = load_model(data_args, model_args)
    trainer = set_trainer(model, data_args, training_args, train_dataset, eval_dataset)
    pruning = register_pruning_hook(model, user_params)

    for i in range(int(user_params["total_training_steps"]/user_params["delta"])):
        train(trainer, model_args)
        eval(trainer, eval_dataset)
        pruning.update_sparcity()


def register_pruning_hook(model, user_params):
    pruning_layers = [
        "bert.embeddings",
        "bert.encoder"
    ]

    pruning = SparsePruning(model, pruning_layers, user_params['total_training_steps'],
                            t0=user_params['fine_tune_steps'], initial_sparsity=user_params['pruning_init_sparse'],
                            final_sparsity=user_params['pruning_final_sparse'], delta=user_params['delta'])
    model.register_forward_pre_hook(pruning)
    return pruning


if __name__ == '__main__':
    run()
