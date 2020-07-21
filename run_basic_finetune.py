import csv
import datetime
import os
import torch
from mrpc_training_tools import set_logger, load_params, load_datasets, load_model, set_trainer, train, evaluate, \
    check_sparsity
from sparse_pruning import SparsePruning
import json


def run(hyper_params="hyper_params.json"):
    # creation time for saving files
    creation_time = datetime.datetime.now().strftime("%Y-%B-%d %H-%M-%S")
    base = os.path.join("models", creation_time)
    os.makedirs(base, exist_ok=True)

    # load user params file
    user_params = json.loads(open(hyper_params, "rt", encoding="utf-8").read())

    # set up enviroment
    model_args, data_args, training_args = load_params(user_params)
    train_dataset, eval_dataset, test_dataset = load_datasets(data_args, model_args)
    model = load_model(data_args, model_args)
    trainer = set_trainer(model, data_args, training_args, train_dataset, eval_dataset)
    pruning = register_pruning_hook(model, user_params)
    measures = ["training_step", "sparsity", "eval_loss", "eval_acc", "eval_f1", "eval_acc_and_f1"]
    results = [measures]

    # prune -> train -> predict loop
    for i in range(user_params['total_training_steps']):
        train(trainer, model_args)
        eval_results = evaluate(trainer, eval_dataset)

        # print weights sparsity
        print(i, "="*50, "sparsity", pruning.curr_sparsity)
        check_sparsity(model)
        print("=" * 50)
        # show intermediate result
        for k, v in eval_results.items():
            print('{:<30}'.format(k), v)

        # save measures and model parameters
        results.append([i, pruning.curr_sparsity] + [eval_results[measure] for measure in measures[2:]])
        dump_model(model, i, base, pruning)

        # update pruning rate
        pruning.update_sparsity()

    # save measures scv and sparsity graph
    to_csv(results, os.path.join(base, "results_{}".format(creation_time) + ".csv"))
    pruning.plot_sparsity(base)


def dump_model(model, train_step, base, pruning):
    os.makedirs(os.path.join(base, "pt_state_dict"), exist_ok=True)
    os.makedirs(os.path.join(base, "ONNX"), exist_ok=True)

    # as pt
    torch.save({
        'sparsity': pruning.curr_sparsity,
        'pruning': pruning,
        'training_step': train_step,
        'model_state_dict': model.state_dict(),
    }, os.path.join(base, "pt_state_dict", "model_state_dict_" + str(train_step) + ".pt"))
    # as ONNX
    torch.onnx.export(model, model.dummy_inputs['input_ids'].to(model.device),
                      os.path.join(base, "ONNX", "bert_base_pruned_" + str(train_step) + ".onnx"))


def to_csv(table, fname):
    with open(fname, "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(table)


def register_pruning_hook(model, user_params):
    """
    initiate prunning object and hook it so it will activate before the forward begins
    """
    # layers to be pruned
    pruning_layers = [
        "bert.embeddings",
        "bert.encoder"
    ]
    # initiate pruning object + hook
    pruning = SparsePruning(model, pruning_layers, user_params['total_training_steps'],
                            user_params['epoch_per_training_step'], t0=user_params['fine_tune_steps'],
                            initial_sparsity=user_params['pruning_init_sparse'],
                            final_sparsity=user_params['pruning_final_sparse'])
    model.register_forward_pre_hook(pruning)
    return pruning


if __name__ == '__main__':
    run()
