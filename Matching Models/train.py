import os
import glob
import deepmatcher as dm

from Evaluation.eval_while_training import (
    evaluate_metrics_and_errors,
    plot_learningcurve_bce,
    plot_metrics_subplots,
)

train_ab, validation_ab, test_ab = dm.data.process(
    path=r"../data/abt-buy_textual/",
    train='train_full.csv',
    validation='validation_full.csv',
    test='test_full.csv',
    cache=False
)

train_ag, validation_ag, test_ag = dm.data.process(
    path=r"../data/Amazon-Google_structured/",
    train='train_full.csv',
    validation='validation_full.csv',
    test='test_full.csv',
    cache=False
)

train_ia, validation_ia, test_ia = dm.data.process(
    path=r"../data/Itunes-Amazon/",
    train='train_full.csv',
    validation='validation_full.csv',
    test='test_full.csv',
    cache=False
)

train_wm, validation_wm, test_wm = dm.data.process(
    path=r"../data/Walmart-Amazon_dirty/",
    train='train_full.csv',
    validation='validation_full.csv',
    test='test_full.csv',
    cache=False
)

datasets = {
    "ab": (train_ab, validation_ab, test_ab),
    "ag": (train_ag, validation_ag, test_ag),
    "ia": (train_ia, validation_ia, test_ia),
    "wm": (train_wm, validation_wm, test_wm),
}


models = ["sif", "attention", "hybrid", "rnn"]

save_dir = r"Matching Models"
os.makedirs(save_dir, exist_ok=True)

num_of_epochs = 15

for ending, (train, validation, test) in datasets.items():
    for modeltype in models:
        model = dm.MatchingModel(attr_summarizer=modeltype)

        base_name = f"Model_{modeltype}_{ending}"
        best_path = os.path.join(save_dir, f"{base_name}_best.pth")

        epoch_prefix = os.path.join(save_dir, base_name)

        model.run_train(
            train,
            validation,
            epochs=num_of_epochs,
            best_save_path=best_path,
            save_every_prefix=epoch_prefix,
            save_every_freq=1,
        )


        epoch_files = sorted(
            glob.glob(os.path.join(save_dir, f"{base_name}_ep*.pth")),
            key=lambda p: int(os.path.splitext(p)[0].split("_ep")[-1])
        )

        epochs = []
        # Jetzt inkl. Fehler:
        train_hist = {
            "accuracy": [], "precision": [], "recall": [], "f1": [],
            "error_rate": [], "bce": []
        }
        test_hist = {
            "accuracy": [], "precision": [], "recall": [], "f1": [],
            "error_rate": [], "bce": []
        }

        for pth in epoch_files:
            ep = int(os.path.splitext(pth)[0].split("_ep")[-1])
            epochs.append(ep)

            model.load_state(pth)

            tr = evaluate_metrics_and_errors(model, train)
            te = evaluate_metrics_and_errors(model, test)

            for k in train_hist.keys():
                train_hist[k].append(tr[k])
                test_hist[k].append(te[k])

        out1 = os.path.join(save_dir+"\\Vis\\", f"{base_name}__learningcurve.png")
        plot_learningcurve_bce(
            epochs,
            train_hist["bce"],
            test_hist["bce"],
            out1,
            title=f"{base_name} – Lernkurve"
        )

        # Diagramm 2: Alle Metriken zusammen (Train & Test) + Fehler optional integriert
        # -> wir plotten hier Accuracy/Precision/Recall/F1 + zusätzlich error_rate & bce
        out2 = os.path.join(save_dir+"\\Vis\\", f"{base_name}__metrics.png")
        plot_metrics_subplots(
            epochs,
            train_hist,
            test_hist,
            out2,
            title=f"{base_name} – Metrics (Train & Test)"
        )

        print(
            f"[OK] Gespeichert:\n"
            f"  {out1}\n"
            f"  {out2}\n"
        )
