from collections import Counter

import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def get_weighted_sampler_and_loss_weights(csv_file: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(csv_file)
    labels = df.label.tolist()
    class_counts = Counter(labels)
    print(class_counts)
    num_samples = len(labels)

    sampler_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [sampler_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # loss function weights
    loss_weights = {
        cls: num_samples / (len(class_counts) * count)
        for cls, count in class_counts.items()
    }
    class_order = sorted(class_counts.keys())  # sorted label order
    sorted_loss_weights = [loss_weights[cls] for cls in class_order]
    class_weights_tensor = torch.tensor(sorted_loss_weights, dtype=torch.float).to(
        device
    )

    return sampler, class_weights_tensor, class_order
