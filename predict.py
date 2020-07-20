import json
import torch
from mrpc_training_tools import load_params, load_datasets, load_model, evaluate, set_trainer, check_sparsity

if __name__ == '__main__':
    path_to_model = "models/2020-July-20 15-42-07/pt_state_dict/model_state_dict_9.pt"
    # load user params file
    user_params = json.loads(open("hyper_params.json", "rt", encoding="utf-8").read())

    # set up enviroment
    model_args, data_args, training_args = load_params(user_params)
    train_dataset, eval_dataset, test_dataset = load_datasets(data_args, model_args)
    model = load_model(data_args, model_args)
    trainer = set_trainer(model, data_args, training_args, train_dataset, eval_dataset)

    checkpoint = torch.load(path_to_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    check_sparsity(model)

    eval_results = evaluate(trainer, eval_dataset, training_args)
    print(eval_results)
