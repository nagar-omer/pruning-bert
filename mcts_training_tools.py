import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import numpy as np
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
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    default_file = os.path.join(__file__.replace("/", os.sep).rsplit(os.sep, 1)[0], "default_config.json")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    default_params = json.loads(open(default_file, "rt", encoding="utf-8").read())
    # override default params
    for k, v in user_params.items():
        default_params[k] = v

    default_params["num_train_epochs"] = user_params["delta"]
    json.dump(default_params, open("temp_params.json", "wt"))
    model_args, data_args, training_args = parser.parse_json_file(json_file="temp_params.json")
    os.remove("temp_params.json")
    logger.info("Training/evaluation parameters %s", training_args)
    return model_args, data_args, training_args


def load_datasets(data_args, training_args):
    tokenizer = AutoTokenizer.from_pretrained(
        'bert-base-uncased',
        cache_dir=None,
    )
    # Get datasets
    train_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, cache_dir=None)
    )
    eval_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    test_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
        if training_args.do_predict
        else None
    )
    return train_dataset, eval_dataset, test_dataset


def load_model(data_args, model_args):
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=glue_tasks_num_labels[data_args.task_name],
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

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
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    # if trainer.is_world_master():
    #     tokenizer.save_pretrained(training_args.output_dir)


def eval(trainer, eval_dataset):
    trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
    eval_result = trainer.evaluate(eval_dataset=eval_dataset)

    output_eval_file = os.path.join(
        training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
    )
    if trainer.is_world_master():
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
            for key, value in eval_result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

    return eval_result


def predict(trainer, test_dataset):
    predictions = np.argmax(trainer.predict(test_dataset=test_dataset).predictions, axis=1)

    output_test_file = os.path.join(
        training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
    )
    if trainer.is_world_master():
        with open(output_test_file, "w") as writer:
            logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
            writer.write("index\tprediction\n")
            for index, item in enumerate(predictions):
                item = test_dataset.get_labels()[item]
                writer.write("%d\t%s\n" % (index, item))


def model_params_count(model):
    params = list(model.named_parameters())
    count = {}
    for p in params:
        layer_name = ".".join(p[0].split(".")[:2])
        param_count = (p[1].size()[0]) if len(p[1].size()) == 1 else (p[1].size()[0] * p[1].size()[1])
        count[layer_name] = count.get(layer_name, 0) + param_count
    for k, v in count.items():
        count[k] /= 1000000

    for k, v in count.items():
        print('{:<30}'.format(k), v)
    return count


if __name__ == '__main__':
    set_logger()
    user_params = json.loads(open("hyper_params.json", "rt", encoding="utf-8").read())
    model_args, data_args, training_args = load_params(user_params)
    train_dataset, eval_dataset, test_dataset = load_datasets(data_args, training_args)
    model = load_model(data_args, model_args)
    model_params_count(model)
    e = 0
