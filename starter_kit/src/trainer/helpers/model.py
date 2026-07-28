from typing import Union, Dict, Any, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import numpy as np


class Model:
    def __init__(self, params: Dict[str, Union[str, int, float, bool]]) -> None:

        self.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=str(params["model_path"])
        )

        if params["dropout"] is not None:
            # https://huggingface.co/docs/transformers/model_doc/modernbert#transformers.ModernBertConfig.classifier_dropout
            # https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Config.classifier_dropout
            # https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartConfig.classifier_dropout
            # https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaConfig.classifier_dropout
            # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertConfig.classifier_dropout
            if hasattr(self.config, "classifier_dropout"):
                self.config.classifier_dropout = float(params["dropout"])
            # https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L1031
            elif hasattr(self.config, "pooler_dropout") and not hasattr(self.config, "cls_dropout"):
                self.config.cls_dropout = float(params["dropout"])
                self.config.problem_type = "single_label_classification"
            # https://huggingface.co/docs/transformers/model_doc/albert#transformers.AlbertConfig.classifier_dropout_prob
            elif hasattr(self.config, "classifier_dropout_prob"):
                self.config.classifier_dropout_prob = float(params["dropout"])
            else:
                # https://discuss.huggingface.co/t/classifier-dropout-for-decodermodel-forsequenceclassification-classes/113989
                # https://huggingface.co/docs/transformers/model_doc/gpt_neox#transformers.GPTNeoXConfig.classifier_dropout
                # GPT2, OPT
                print("Warning: classifier_dropout attribute not found in AutoConfig. Dropout will make no effect.")

        self.config.num_labels = int(params["num_labels"])

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=str(params["tokenizer_path"])
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=str(params["model_path"]),
            config=self.config,
        )

        # For pythia
        if not self.tokenizer.pad_token:
            print("No padding token detected. Expanding tokenizer and setting padding token to [PAD].")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.emb_size: int = self.model.config.hidden_size
        # self.model.config.max_position_embeddings
        self.max_length: int = int(params["max_length"])

    def forward(self, batch: Dict[str, torch.Tensor], device: torch.device) -> SequenceClassifierOutput:
        inputs: Dict[str, torch.Tensor] = {
            "input_ids": batch["input_ids"].to(device=device),
            "attention_mask": batch["attention_mask"].to(device=device),
        }
        
        if "token_type_ids" in batch:
            inputs["token_type_ids"] = batch["token_type_ids"].to(device=device)

        # Check if labels are provided in the batch
        if "labels" in batch:
            inputs["labels"] = batch["labels"].to(device=device)

        return self.model(**inputs)

    def tokenize(self, data: np.ndarray) -> dict[str, Any]:
        texts_first, texts_second = zip(*data)
        texts_first: np.ndarray
        texts_second: np.ndarray

        encs: BatchEncoding = self.tokenizer(
            list(texts_first),
            list(texts_second),
            padding="max_length",
            truncation="only_second",
            return_tensors="pt",
            max_length=self.max_length,
        )

        return encs.data
    
    def to(self, device: torch.device) -> None:
        self.model.to(device=device)

    def parameters(self) -> torch.nn.Parameter:
        return self.model.parameters()

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def decode(self, token_ids: List[List[int]]) -> List[str]:
        decoded_outputs:List[str] = self.tokenizer.batch_decode(sequences=token_ids, skip_special_tokens=False)
        return [text.replace(self.tokenizer.pad_token, "") for text in decoded_outputs]

    def save(self, path: str) -> None:
        self.model.save_pretrained(save_directory=path)
        self.tokenizer.save_pretrained(save_directory=path)
