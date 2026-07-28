from torch.utils.data import Dataset
import numpy as np
import torch
from typing import List, Dict, Optional, Any

class NLIDataset(Dataset):
    def __init__(
        self, 
        X: Dict[str, np.ndarray], 
        ids: np.ndarray,
        device: torch.device, 
        labels: Optional[np.ndarray] = None 
    ) -> None:
        super().__init__()
        self.device: torch.device = device
        self.X: Dict[str, np.ndarray] = X
        self.ids: np.ndarray = ids
        
        self.labels: Optional[np.ndarray] = None
        if labels is not None:
            self.labels = labels.reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.X["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "id": str(self.ids[idx]),  # Cast to string to ensure native Python type
            "input_ids": torch.clone(input=self.X["input_ids"][idx]),
            "attention_mask": torch.clone(input=self.X["attention_mask"][idx]),
        }

        # Add labels only if they were provided
        if self.labels is not None:
            item["labels"] = torch.from_numpy(self.labels[idx])

        # Add token_type_ids if available
        if "token_type_ids" in self.X:
            item["token_type_ids"] = torch.clone(input=self.X["token_type_ids"][idx])
            
        return item

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        collated: Dict[str, Any] = {
            "ids": [item["id"] for item in batch],
            "input_ids": torch.stack(tensors=[item["input_ids"] for item in batch]).to(device=self.device),
            "attention_mask": torch.stack(tensors=[item["attention_mask"] for item in batch]).to(device=self.device),
        }
        
        if "labels" in batch[0]:
            collated["labels"] = torch.stack(tensors=[item["labels"] for item in batch]).to(device=self.device)

        if "token_type_ids" in batch[0]:
            collated["token_type_ids"] = torch.stack(tensors=[item["token_type_ids"] for item in batch]).to(device=self.device)

        return collated