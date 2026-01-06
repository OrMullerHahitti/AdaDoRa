#!/usr/bin/env python
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
# Script for regular LoRA/DoRA finetuning (no adaptive rank allocation).
"""Finetuning models for sequence classification on GLUE with LoRA/DoRA."""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


@dataclass
class DataTrainingArguments:
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(default=128)
    overwrite_cache: bool = field(default=False)
    pad_to_max_length: bool = field(default=True)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys:
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)


@dataclass
class LoraArguments:
    """Arguments for LoRA/DoRA configuration."""
    use_dora: bool = field(default=False, metadata={"help": "Enable DoRA (Weight-Decomposed Low-Rank Adaptation)"})
    r: int = field(default=8, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha scaling factor"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout rate"})
    target_modules: Optional[str] = field(
        default="query,value",
        metadata={"help": "Comma-separated target modules (e.g., 'query,value')"},
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LoraArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, lora_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    method_name = "DoRA" if lora_args.use_dora else "LoRA"
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"{method_name} parameters: r={lora_args.r}, alpha={lora_args.lora_alpha}, use_dora={lora_args.use_dora}")

    set_seed(training_args.seed)

    # load dataset
    raw_datasets = load_dataset("nyu-mll/glue", data_args.task_name, cache_dir=model_args.cache_dir)

    # labels
    is_regression = data_args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    # load model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # setup LoRA/DoRA config
    target_modules = lora_args.target_modules.split(",") if lora_args.target_modules else ["query", "value"]

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_args.r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        use_dora=lora_args.use_dora,
    )

    logger.info(f"PEFT Config: {peft_config}")

    # apply PEFT
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # preprocessing
    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    padding = "max_length" if data_args.pad_to_max_length else False
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        return tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples:
        train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))

    eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    if data_args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))

    # metrics
    metric = evaluate.load("glue", data_args.task_name, cache_dir=model_args.cache_dir)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # data collator
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("Done!")


if __name__ == "__main__":
    main()
