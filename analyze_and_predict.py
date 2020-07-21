import json
import torch
from IPython.display import Image
from IPython.display import display
import pandas as pd
from mrpc_training_tools import load_params, load_datasets, load_model, evaluate, set_trainer, check_sparsity, predict
import onnx
import os


def load_pretrained_model(path_to_model, data_args, model_args):
    model = load_model(data_args, model_args)
    checkpoint = torch.load(path_to_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def load_enviroment(path_to_model, params_file="hyper_params.json"):
    # load user params file
    user_params = json.loads(open(params_file, "rt", encoding="utf-8").read())

    # set up enviroment
    model_args, data_args, training_args = load_params(user_params)
    train_dataset, eval_dataset, test_dataset = load_datasets(data_args, model_args)
    model = load_pretrained_model(path_to_model, data_args, model_args)
    trainer = set_trainer(model, data_args, training_args, train_dataset, eval_dataset)

    try:
        image, res_table = None, None
        base_dir = path_to_model.replace("/", os.sep).rsplit(os.sep, 2)[0]
        for filename in os.listdir(base_dir):
            if filename == "sparsity_graph.png":
                image = os.path.join(base_dir, "sparsity_graph.png")

            elif "results_" in filename:
                res_table = os.path.join(base_dir, filename)

        if res_table is not None:
            df = pd.read_csv(res_table)
            display(df)
        if image is not None:
            display(Image(filename=image))
    except:
        pass

    return model, trainer, train_dataset, eval_dataset, test_dataset


def analayze_and_predict(model, trainer, train_dataset, eval_dataset):
    train_results = evaluate(trainer, train_dataset)
    dev_results = evaluate(trainer, eval_dataset)

    print("=" * 10, "Sparsity", "=" * 10)
    check_sparsity(model)
    print("=" * 30)

    for res_type, res in zip(["Train", "Dev"], [train_results, dev_results]):
        print("=" * 10, "Results for", res_type, "="*10)
        for k, v in res.items():
            print('{:<30}'.format(k), v)
        print("=" * 20)


def export_predictions(trainer, dataset, out_file):
    predict(trainer, dataset, out_file)


if __name__ == '__main__':
    PARAMS = "hyper_params.json"
    MODEL = "models/2020-July-20 20-06-46/pt_state_dict/model_state_dict_5.pt"

    model, trainer, train_dataset, eval_dataset, test_dataset = load_enviroment(MODEL, params_file=PARAMS)
