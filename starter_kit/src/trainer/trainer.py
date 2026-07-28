from typing import Union, Dict, Tuple, Optional, List
from tqdm import tqdm
import optuna
import os
import random
import csv
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import transformers
from transformers.modeling_outputs import SequenceClassifierOutput
from src.trainer.helpers.data_reader import read_transform_dico, label_encoder
from src.trainer.helpers.model import Model
from src.trainer.helpers.dataset_generator import NLIDataset
from src.trainer.helpers.evaluation import evaluate

class Trainer:
    def __init__(self, params: Dict[str, Union[str, int, float, bool, optuna.trial.Trial, None]]) -> None:
        """
        params: dictionary containing the following parameters:

            - seed: int, random seed for reproducibility.
            - max_length: int, maximum length to use by one of the truncation/padding parameters.
            - do_train: bool, whether to train the model.
            - train_path (optional): str, path to the training dataset.
            - do_dev: bool, whether to evaluate the model on the development set.
            - dev_path (optional): str, path to the development dataset.
            - do_test: bool, whether to evaluate the model on the test set.
            - test_path (optional): str, path to the test dataset.

            - model_path: str, a path to a directory containing model weights saved using save_pretrained() method or a model id of a pretrained model hosted inside a model repo on huggingface.co.
            - tokenizer_path: str, a path to a directory containing vocabulary files required by the tokenizer saved using the save_pretrained() method or a the model id of a predefined tokenizer hosted inside a model repo on huggingface.co.
                - num_labels (derived from experiment): int, number of labels for classification.

            - dropout (optional): float, dropout rate for the classifier layer.
            - warmup_pcrt (optional): float, percentage of training steps to use for linear learning rate warmup.
            - lr (optional): float, learning rate for the optimizer.
            - batch_size (optional): int, batch size for training and evaluation.
            - wd (optional): float, weight decay for the optimizer.
            - num_epochs (optional): int, number of training epochs.

            - optuna (optional): optuna.trial.Trial object, for hyperparameter optimization.
            - save_path (optional): str, path to save the trained model.
            - epoch_to_stop (optional): int, Epoch number to stop training in case intermediate saving is desired.
            - results_dir: str, directory where timestamped prediction CSV files will be written.
            - reference_path (optional): str, hidden reference CSV for official DiCo-NLI scoring.

        """
        self.params = params
        self.params.setdefault("optuna", None)
        self.params.setdefault("results_dir", "results")
        self.params.setdefault("reference_path", None)
        print(self.params)

        # Set seeds for reproducibility
        self.g: torch.Generator = self.set_seed()

        # Cuda configuration
        self.device: torch.device = self.configure_cuda()

        # Load dico datasets
        if params["do_train"]:
            self.ids_train, self.X_train, self.y_train = self.load_dataset(path=str(params["train_path"]))
        if params["do_dev"]:
            self.ids_dev, self.X_dev, self.y_dev = self.load_dataset(path=str(params["dev_path"]))
        if params["do_test"]:
            self.ids_test, self.X_test, self.y_test = self.load_dataset(path=str(params["test_path"]))

        self.labels = label_encoder.classes_

        # Either training, development, or test data (or some combination) may be provided
        self.X_train: Optional[np.ndarray]
        self.y_train: Optional[np.ndarray]
        self.X_dev: Optional[np.ndarray]
        self.y_dev: Optional[np.ndarray]
        self.X_test: Optional[np.ndarray]
        self.y_test: Optional[np.ndarray]
        self.labels: np.ndarray

        # Initialize model
        self.params["num_labels"] = len(self.labels)
        self.model = Model(params=self.params)
        self.model.to(device=self.device)

        # Generate Datasets and Dataloaders
        if params["do_train"]:
            self.train_dataset = self.model.tokenize(data=self.X_train)
            self.train_dataset = NLIDataset(X=self.train_dataset, labels=self.y_train, ids=self.ids_train, device=self.device)
            self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=int(params["batch_size"]), collate_fn=self.train_dataset.collate_fn, sampler=RandomSampler(self.train_dataset, generator=self.g), num_workers=0)
            if params["do_dev"]:
                self.dev_dataset = self.model.tokenize(data=self.X_dev)
                self.dev_dataset = NLIDataset(X=self.dev_dataset, labels=self.y_dev, ids=self.ids_dev, device=self.device)
                self.dev_dataloader = DataLoader(dataset=self.dev_dataset, batch_size=int(params["batch_size"]), collate_fn=self.dev_dataset.collate_fn, sampler=SequentialSampler(self.dev_dataset), num_workers=0)
        if params["do_test"]:
            self.test_dataset = self.model.tokenize(data=self.X_test)
            self.test_dataset = NLIDataset(X=self.test_dataset, labels=self.y_test, ids=self.ids_test, device=self.device)
            self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=int(params["batch_size"]), collate_fn=self.test_dataset.collate_fn, sampler=SequentialSampler(self.test_dataset), num_workers=0)

        # Either training, development, or test data (or some combination) may be provided
        self.train_dataset: Optional[NLIDataset]
        self.train_dataloader: Optional[DataLoader]
        self.dev_dataset: Optional[NLIDataset]
        self.dev_dataloader: Optional[DataLoader]
        self.test_dataset: Optional[NLIDataset]
        self.test_dataloader: Optional[DataLoader]

        if params["do_train"]:
            self.train_loop()

        if params["do_test"]:
            self.evaluation_loop(partition="test", print_results=True)
        
    def set_seed(self) -> torch.Generator:
        # https://docs.pytorch.org/docs/stable/notes/randomness.html
        # https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/trainer_utils.py#L61
        seed = int(self.params["seed"])
        transformers.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
        os.environ["HCCL_DETERMINISTIC"] = "1"
        os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        g = torch.Generator()
        g.manual_seed(seed)
        return g

    def configure_cuda(self) -> torch.device:
        cuda: bool = torch.cuda.is_available()
        device: torch.device = torch.device("cuda" if cuda else "cpu")
        print("\nRunning on device: {}".format(device))
        if cuda:
            print("CUDA device count: {}".format(torch.cuda.device_count()))
            print("CUDA current device: {}".format(torch.cuda.current_device()))
            print("CUDA device 0: {}".format(torch.cuda.device(0)))
            print("CUDA device 0 name: {}".format(torch.cuda.get_device_name(0)))
            print()
        return device

    def load_dataset(self, path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        ids, X, y = read_transform_dico(path=path)
        ids: np.ndarray
        X: np.ndarray
        y: Optional[np.ndarray]

        print(f"Dataset loaded. Path: {path}")
        print(f"Shape of X: {X.shape}")
        if y is not None:
            print(f"Shape of y: {y.shape}")
        else:
            print("No labels provided in the dataset.")
        print()

        return ids, X, y
    
    def train_loop(self) -> None:
        lr = float(self.params["lr"])
        weight_decay = float(self.params["wd"])
        num_epochs = int(self.params["num_epochs"])
        warmup_pct = float(self.params["warmup_pcrt"])

        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-6,
        )

        train_data_len: int = len(self.train_dataloader)
        total_training_steps: int = train_data_len * num_epochs
        warmup_steps = int(total_training_steps * warmup_pct)

        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

        print("\nStarting training loop:")

        for epoch in range(num_epochs):
            print(f"\n>>> Epoch {epoch + 1}/{num_epochs}")
            self.model.train()
            epoch_loss = 0.0

            for batch in tqdm(self.train_dataloader):
                optimizer.zero_grad()
                outputs: SequenceClassifierOutput = self.model.forward(batch=batch, device=self.device)
                loss: torch.Tensor = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / train_data_len
            print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

            if self.params["do_dev"]:
                results:Dict[str, Union[float, np.ndarray]] = self.evaluation_loop(partition="dev", print_results=True)
                if self.params["optuna"] is not None:
                    
                    # Set another metric to report if needed 

                    self.params["optuna"].report(value=results["f1_weighted"], step=(epoch + 1))
                    if self.params["optuna"].should_prune():
                        raise optuna.TrialPruned()
            
            if self.params["epoch_to_stop"] is not None and (epoch + 1) == int(self.params["epoch_to_stop"]):
                print(f"Stopping at epoch {epoch + 1} as specified.")
                break
            
        if self.params["save_path"] is not None:
            print(f"Saving model in path: {self.params['save_path']}")
            self.model.save(path=str(self.params["save_path"]))

    def evaluation_loop(self, partition: str, print_results: bool) -> Optional[Dict[str, Union[float, np.ndarray]]]:
        self.model.eval()
        if partition == "test":
            dataloader = self.test_dataloader
        elif partition == "dev":
            dataloader = self.dev_dataloader
        else:
            raise ValueError(f"Unknown partition: {partition}")
        dataloader: DataLoader

        preds: List[int] = []
        labels: Optional[List[int]] = []
        ids: List[str] = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                outputs: SequenceClassifierOutput = self.model.forward(batch=batch, device=self.device)
                logits: torch.Tensor = outputs.logits
                batch_preds:List[int] = torch.argmax(input=logits, dim=-1).cpu().numpy().tolist()
                preds.extend(batch_preds)

                ids.extend(batch["ids"])

                if "labels" in batch:
                    batch_labels:List[int] = batch["labels"].cpu().numpy().tolist()
                    labels.extend(batch_labels)

        if len(labels) != 0:
            results = evaluate(y_true=labels, y_pred=preds, num_labels=len(self.labels))
            if print_results:
                decoded_preds: np.ndarray = label_encoder.inverse_transform(y=np.array(preds).ravel())
                decoded_labels: np.ndarray = label_encoder.inverse_transform(y=np.array(labels).ravel())
                if self.params["optuna"] is None:
                    prediction_path = self.write_predictions(ids=ids, labels=decoded_preds, partition=partition)
                    self.score_official_predictions(prediction_path=prediction_path, partition=partition)

                print(f"\nEvaluation results on {partition} set:")
                print(
                    pd.DataFrame(
                        data=results["matrix"],
                        index=[f"true_{name}" for name in list(self.labels)],
                        columns=[f"pred_{name}" for name in list(self.labels)],
                    ).to_string()
                ) 
                print(f"Accuracy: {results['accuracy']:.4f}")
                print(f"F1-macro: {results['f1_macro']:.4f}")
                print(f"F1-weighted: {results['f1_weighted']:.4f}")
                print()

                sentences:List[str] = []
                for batch in dataloader:
                    input_ids:List[List[int]] = batch["input_ids"].cpu().numpy().tolist()
                    decoded_sentences:List[str]= self.model.decode(token_ids=input_ids)
                    sentences.extend(decoded_sentences)

                if partition == "test" or partition == "dev":
                    for id, sent, pred, true in zip(ids, sentences, decoded_preds, decoded_labels):
                        print(f"ID: {id} | Sentence: {sent} | Predicted: {pred} | True: {true}")         
        else:
            print(f"No labels found in the {partition} set. Skipping evaluation metrics.")
            results = None   

            if print_results:
                sentences:List[str] = []
                for batch in dataloader:
                    input_ids:List[List[int]] = batch["input_ids"].cpu().numpy().tolist()
                    decoded_sentences:List[str]= self.model.decode(token_ids=input_ids)
                    sentences.extend(decoded_sentences)
                
                preds = label_encoder.inverse_transform(y=np.array(preds).ravel())
                if self.params["optuna"] is None:
                    prediction_path = self.write_predictions(ids=ids, labels=preds, partition=partition)
                    self.score_official_predictions(prediction_path=prediction_path, partition=partition)

                for id, sent, pred in zip(ids, sentences, preds):
                    print(f"ID: {id} | Sentence: {sent} | Predicted: {pred}")
        return results

    def write_predictions(self, ids: List[str], labels: np.ndarray, partition: str) -> Path:
        results_dir = Path(str(self.params["results_dir"]))
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = results_dir / f"predictions_{partition}_{timestamp}.csv"

        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["instance_id", "label"], lineterminator="\n")
            writer.writeheader()
            for instance_id, label in zip(ids, labels):
                writer.writerow({"instance_id": instance_id, "label": str(label)})

        print(f"Predictions written to: {output_path}")
        return output_path

    def score_official_predictions(self, prediction_path: Path, partition: str) -> None:
        reference_path = self.params["reference_path"]
        if reference_path is None:
            return

        project_root = Path(__file__).resolve().parents[3]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from evaluation_functions import ScoringError, score_files

        output_dir = Path(str(self.params["results_dir"])) / f"{prediction_path.stem}_official_scores"
        try:
            report = score_files(
                reference_path=reference_path,
                prediction_path=prediction_path,
                output_dir=output_dir,
            )
        except ScoringError as exc:
            print(f"Official DiCo-NLI scoring skipped: {exc}")
            return

        payload = report.to_dict()
        print(f"\nOfficial DiCo-NLI metrics on {partition} set:")
        print(f"Weighted F1: {payload['weighted_f1']:.4f}")
        print(f"SoftCons: {payload['soft_cons']:.4f}")
        print(f"HardCons: {payload['hard_cons']:.4f}")
        print(f"Official score files written to: {output_dir}")
