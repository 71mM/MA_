import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

def plot_history(history: Dict[str, List[float]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    if len(history.get("test_loss", [])) > 0:
        plt.plot(history["test_loss"], label="test_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"))
    plt.close()

    if "val_f1" in history and not all(np.isnan(np.array(history["val_f1"]))):
        plt.figure()
        plt.plot(history["val_f1"], label="val_f1")
        if len(history.get("test_f1", [])) > 0:
            plt.plot(history["test_f1"], label="test_f1")
        plt.xlabel("epoch")
        plt.ylabel("f1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "f1_curves.png"))
        plt.close()

    if "val_auc" in history and not all(np.isnan(np.array(history["val_auc"]))):
        plt.figure()
        plt.plot(history["val_auc"], label="val_auc")
        if len(history.get("test_auc", [])) > 0:
            plt.plot(history["test_auc"], label="test_auc")
        plt.xlabel("epoch")
        plt.ylabel("auc")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "auc_curves.png"))
        plt.close()
