import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)
import json


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


def set_logger():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    # Set seed
    set_seed(260290)


def load_params(user_params):
    """
    generate model, data and training arg-opjects according to user hyper-parameters
    """
    # params parser
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # override default params
    default_file = os.path.join(__file__.replace("/", os.sep).rsplit(os.sep, 1)[0], "source_attrib.json")
    default_params = json.loads(open(default_file, "rt", encoding="utf-8").read())
    for k, v in user_params.items():
        default_params[k] = v
    default_params["num_train_epochs"] = user_params["epoch_per_training_step"]

    # create alternative json and parse it
    json.dump(default_params, open("temp_params.json", "wt"))
    model_args, data_args, training_args = parser.parse_json_file(json_file="temp_params.json")
    os.remove("temp_params.json")

    training_args.do_eval = False
    return model_args, data_args, training_args


def load_datasets(data_args, model_args):
    # set tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=None,
    )
    # Get datasets
    train_dataset = GlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir)
    eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
    test_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)

    return train_dataset, eval_dataset, test_dataset


def load_model(data_args, model_args):
    # load configuration
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=glue_tasks_num_labels[data_args.task_name],
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    # ============================= DEBUG use only one layer ==============================================
    # config.num_hidden_layers = 1
    # config.num_attention_heads = 1
    # load model according to configuration
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    return model


def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        return glue_compute_metrics(task_name, preds, p.label_ids)
    return compute_metrics_fn


def set_trainer(model, data_args, training_args, train_dataset, eval_dataset):
    # set training object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
    )
    return trainer


def train(trainer, model_args):
    trainer.train(
        model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
    )


def evaluate(trainer, eval_dataset):
    trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
    eval_result = trainer.evaluate(eval_dataset=eval_dataset)
    return eval_result


def predict(trainer, test_dataset, output_test_file):
    predictions = np.argmax(trainer.predict(test_dataset=test_dataset).predictions, axis=1)
    if trainer.is_world_master():
        with open(output_test_file, "w") as writer:
            writer.write("index\tprediction\n")
            for index, item in enumerate(predictions):
                item = test_dataset.get_labels()[item]
                writer.write("%d\t%s\n" % (index, item))


def model_params_count(model):
    """
    count model params by layers
    """
    params = list(model.named_parameters())
    count = {}
    # loop all layers
    for p in params:
        # get name and number of parameters (scaled by M)
        layer_name = ".".join(p[0].split(".")[:2])
        param_count = p[1].view(-1).size()[0]
        count[layer_name] = count.get(layer_name, 0) + param_count
    for k, v in count.items():
        count[k] /= 1000000

    # print and return
    for k, v in count.items():
        print('{:<30}'.format(k), v)
    return count


def check_sparsity(model):
    for layer_name, parameters in model.named_parameters():
        sparse_count = torch.where(parameters == 0)[0].shape[0]
        param_count = parameters.view(-1).shape[0]
        print('{:<50}'.format(layer_name), round(sparse_count/param_count, 2))


if __name__ == '__main__':
    set_logger()
    user_params_ = json.loads(open("hyper_params.json", "rt", encoding="utf-8").read())
    model_args_, data_args_, training_args_ = load_params(user_params_)
    train_dataset_, eval_dataset_, test_dataset_ = load_datasets(data_args_, model_args_)
    model_ = load_model(data_args_, model_args_)
    model_params_count(model_)
    e = 0
