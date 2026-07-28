import pandas as pd
import numpy as np
import re
from typing import Tuple, Optional
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
label_encoder.fit(["BACKWARD_ENTAILMENT", "FORWARD_ENTAILMENT", "EQUIVALENCE", "NEGATIVE_OTHER"])

def remove_extra_spaces(sentence: str) -> str:
    return re.sub(pattern=r"\s+", repl=" ", string=sentence).strip()

def read_transform_dico(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:             
    """
    Dataset format (comma-separated):
    - Required columns: instance_id, text1, text2
    - Optional column: label
    """

    # 1. Read by column name so participant and hidden-gold CSVs both work.
    df: pd.DataFrame = pd.read_csv(filepath_or_buffer=path, sep=",", header=0)
    required_columns = ["instance_id", "text1", "text2"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing_columns)}")

    has_label = "label" in df.columns
    columns = required_columns + (["label"] if has_label else [])
    df = df[columns].rename(
        columns={
            "instance_id": "INSTANCE_ID",
            "text1": "FIRST_TEXT",
            "text2": "SECOND_TEXT",
            "label": "GOLD_NLI_LABELS",
        }
    )

    # 2. Clean the string columns to remove extra spaces
    for col in df.columns:
        df[col] = df[col].astype(str).apply(func=remove_extra_spaces)

    # 3. Handle if-else to encode labels or return None
    if has_label:
        transformed_labels = label_encoder.transform(y=df["GOLD_NLI_LABELS"])
    else:
        transformed_labels = None
    
    # 4. Return instance IDs, text pairs, and transformed labels (or None)
    return (
        df["INSTANCE_ID"].to_numpy(),
        df[["FIRST_TEXT", "SECOND_TEXT"]].to_numpy(), 
        transformed_labels
    )
