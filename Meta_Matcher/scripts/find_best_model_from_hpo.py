#because I did not log whih trial was the one that performed best i now need this script

import os
import json

base_path = "runs_hpo"

data_sets = ["abt-buy_textual", "amazon-google_structured","dblp_scholar_exp_data",  "itunes-amazon", "walmart-amazon_dirty"]

for dataset in data_sets:
    data_path = os.path.join(base_path, dataset)
    data_path = os.path.join(data_path,"minilm")
    with open(os.path.join(data_path,"best_params.json"), "r") as f:
        data = json.load(f)

    best_f1 = data["best_user_attrs"]["best_val_metrics"]["f1"]

    print(f"{dataset} → Best F1: {best_f1}")
    Found = False
    highest_found_f1 = None
    for i in range (0,169):
        if i < 10:
            trial_name = f"trial_000{i}"
        else:
            trial_name = f"trial_00{i}"
        if Found is True:
            break

        trail_path = os.path.join(data_path,trial_name)
        history_path = os.path.join(trail_path,"history.json")
        if os.path.exists(history_path):
            with open(os.path.join(history_path), "r") as f:
                trial_data = json.load(f)

            f1_history = trial_data["val_f1"]
            if highest_found_f1 is None or max(f1_history) > highest_found_f1:
                highest_found_f1 = max(f1_history)
                max_trial_name = trial_name
                max_index  = f1_history.index(max(f1_history))
                related_max_test_performance = trial_data["test_f1"][max_index]
            if best_f1 in f1_history:
                best_trial_name = trial_name
                best_index = f1_history.index(best_f1)
                related_best_test_performance = trial_data["test_f1"][best_index]
                print(f"the best F1_score for {dataset} is in {best_trial_name} it relates to an f1 score of {related_best_test_performance}")
                Found = True


    print(f"the highest f1 score found was {highest_found_f1} in {max_trial_name} it relates to an f1 score of {related_max_test_performance}")
    print("------------------------------------------------------------------")