import os
import random
from typing import List, Tuple, Dict, Union, Optional

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from pytorch_metric_learning import miners, losses
from collections import defaultdict

from dataset import load_data, load_cfg

dpath = os.path.dirname(os.path.abspath(__file__))

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class PairedCollator:
    def __init__(self, tokenizer, max_length: int = 40, mode: str = "paired") -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.valid_modes = {"paired", "single"}
        self.set_mode(mode)
    
    def __call__(
        self, batch: List[Union[Tuple[str, str, int], Tuple[str, int]]]
    ) -> Dict[str, torch.Tensor]:
        if self.mode == "paired":
            return self._tokenize_paired(batch)
        else:  # mode == "single" (validated in set_mode)
            return self._tokenize_single(batch)
    
    def _tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    
    def _tokenize_paired(
        self, batch: List[Tuple[str, str, int]]
    ) -> Dict[str, torch.Tensor]:
        query1, query2, query_id = zip(*batch)
        encodings1 = self._tokenize(list(query1))
        encodings2 = self._tokenize(list(query2))
        
        return {
            "input_ids1": encodings1["input_ids"],
            "attention_mask1": encodings1["attention_mask"],
            "input_ids2": encodings2["input_ids"],
            "attention_mask2": encodings2["attention_mask"],
            "labels": torch.tensor(query_id),
            "return_loss": True,
        }
    
    def _tokenize_single(self, batch: List[Tuple[str, int]]) -> Dict[str, torch.Tensor]:
        query1, query_id = zip(*batch)
        encodings = self._tokenize(list(query1))
        
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(query_id),
        }
    
    def set_mode(self, mode: str) -> None:
        if mode not in self.valid_modes:
            raise ValueError(f"Mode must be one of {self.valid_modes}")
        self.mode = mode

class ContrastiveTrainer(Trainer):
    def __init__(
        self,
        model,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        miner_margin: float = 0.2,
        type_of_triplets: str = "all",
        ms_loss_alpha: float = 2,
        ms_loss_beta: float = 50,
        ms_loss_base: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )
        
        # initialize miner and loss function
        self.miner = miners.TripletMarginMiner(
            margin=miner_margin, type_of_triplets=type_of_triplets
        )
        self.loss_fn = losses.MultiSimilarityLoss(
            alpha=ms_loss_alpha, beta=ms_loss_beta, base=ms_loss_base
        )
    
    def get_embeddings(self, input_ids, attention_mask):
        #get embeddings for each query
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return F.normalize(cls_embeddings, p=2, dim=1)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        cls_embeddings1 = self.get_embeddings(
            inputs["input_ids1"], inputs["attention_mask1"]
        )
        cls_embeddings2 = self.get_embeddings(
            inputs["input_ids2"], inputs["attention_mask2"]
        )
        
        labels = inputs["labels"]
        
        # Concatenate embeddings and labels
        query_embed = torch.cat([cls_embeddings1, cls_embeddings2], dim=0)
        all_labels = torch.cat([labels, labels], dim=0)
        
        # Hard mining of negatives for loss computation
        hard_samples = self.miner(query_embed, all_labels)
        loss = self.loss_fn(query_embed, all_labels, hard_samples)
        
        if return_outputs:
            return loss, (query_embed, all_labels)
        
        return loss

    # Сustom prediction step for paired input
    def prediction_step(
        self, model, inputs, prediction_loss_only=True, ignore_keys=None
    ):
        inputs = self._prepare_inputs(inputs)
        return_loss = inputs.get("return_loss", self.can_return_loss)

        if return_loss is None:
            return_loss = self.can_return_loss

        with torch.no_grad():
            loss = None
            with self.compute_loss_context_manager():
                if prediction_loss_only:
                    loss = self.compute_loss(model, inputs, return_outputs=False)
                    return (loss, None, None)
                else:
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True
                    )

        return (loss, outputs[0], outputs[1])

def train(
    path_queries: str,
    path_kb: str,
    output_folder: Optional[str] = None,
    model_path: str = "GerMedBERT/medbert-512",
    learning_rate: float = 2e-5,
    batch_size: int = 32,
    epochs: int = 10,
    miner_margin: float = 0.2,
    type_of_triplets: str = "all",
    loss_alpha: float = 2,
    loss_beta: float = 50,
    loss_base: float = 0.5,
    split_dev_test: float = 0.1,
    split_train_eval: float = 0.1,
    seed: Optional[int] = None,
):
    if seed:
        set_all_seeds(seed)

    if output_folder is None:
        output_folder = dpath
    output_path = os.path.join(output_folder)

    checkpoint_dir = os.path.join(output_path, "results")
    logging_dir = os.path.join(output_path, "logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)

    # Load and split datasets
    train_data, test_data = load_data(
        path_queries, path_kb, split_dev_test, split_train_eval
    )
    train_ds, eval_ds = train_data["train"], train_data["eval"]
    train_collator = PairedCollator(tokenizer=tokenizer, mode="paired")

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir=logging_dir,
        logging_strategy="epoch",
        learning_rate=learning_rate,
        metric_for_best_model="loss",
    )

    model = AutoModel.from_pretrained(model_path).to(device)

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        miner_margin=miner_margin,
        type_of_triplets=type_of_triplets,
        ms_loss_alpha=loss_alpha,
        ms_loss_beta=loss_beta,
        ms_loss_base=loss_base,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=train_collator,
    )
    # Training
    train_res = trainer.train()
    metrics = train_res.metrics
    trainer.save_model(os.path.join(checkpoint_dir, "checkpoint-best"))
    tokenizer.save_pretrained(os.path.join(checkpoint_dir, "checkpoint-best"))
    metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(eval_ds)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help="Config file path.", type=str)
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    train(**cfg)