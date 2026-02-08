import os
import numpy as np
import pandas as pd


def build_full_split(
    pairs_path: str,
    tableA_path: str,
    tableB_path: str,
    out_path: str,
    *,
    a_id_col: str = "id",
    b_id_col: str = "id",
    left_key: str = "ltable_id",
    right_key: str = "rtable_id",
    label_col: str = "label",
    left_prefix: str = "left_",
    right_prefix: str = "right_",
):

    pairs = pd.read_csv(pairs_path)
    A = pd.read_csv(tableA_path)
    B = pd.read_csv(tableB_path)


    required = {left_key, right_key}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"{pairs_path} missing columns: {missing}")

    if label_col not in pairs.columns:
        print(f"Hinweis: '{label_col}' nicht gefunden in {pairs_path}. Ich mache weiter ohne label.")
    if a_id_col not in A.columns:
        raise ValueError(f"{tableA_path} has no id column '{a_id_col}'")
    if b_id_col not in B.columns:
        raise ValueError(f"{tableB_path} has no id column '{b_id_col}'")

    A_pref = A.rename(columns={c: f"{left_prefix}{c}" for c in A.columns if c != a_id_col}).copy()
    B_pref = B.rename(columns={c: f"{right_prefix}{c}" for c in B.columns if c != b_id_col}).copy()

    A_pref = A_pref.rename(columns={a_id_col: left_key})
    B_pref = B_pref.rename(columns={b_id_col: right_key})


    out = pairs.merge(A_pref, on=left_key, how="left").merge(B_pref, on=right_key, how="left")

    out.insert(0, "id", np.arange(len(out)))

    left_cols = [c for c in out.columns if c.startswith(left_prefix)]
    right_cols = [c for c in out.columns if c.startswith(right_prefix)]

    missing_left = int(out[left_cols].isna().all(axis=1).sum()) if left_cols else 0
    missing_right = int(out[right_cols].isna().all(axis=1).sum()) if right_cols else 0

    if missing_left or missing_right:
        print(f"Join-Warnung: missing_left_rows={missing_left}, missing_right_rows={missing_right}")
    out.drop(columns=["ltable_id", "rtable_id"], inplace=True)
    out.to_csv(out_path, index=False)
    print(f"geschrieben: {out_path}  (rows={len(out)}, cols={out.shape[1]})")


if __name__ == "__main__":
    directories = ["abt-buy_textual", "Amazon-Google_structured", "Itunes-Amazon", "Walmart-Amazon_dirty"]
    for directory in directories:
        BASE = r"../data/" + directory + "/raw"

        tableB = os.path.join(BASE, "tableB.csv")
        tableA = os.path.join(BASE, "tableA.csv")

        build_full_split(
            pairs_path=os.path.join(BASE, "train.csv"),
            tableA_path=tableA,
            tableB_path=tableB,
            out_path=os.path.join(r"../data/" + directory, "train_full.csv"),
        )

        build_full_split(
            pairs_path=os.path.join(BASE, "valid.csv"),
            tableA_path=tableA,
            tableB_path=tableB,
            out_path=os.path.join(r"../data/" + directory, "validation_full.csv"),
        )

        build_full_split(
            pairs_path=os.path.join(BASE, "test.csv"),
            tableA_path=tableA,
            tableB_path=tableB,
            out_path=os.path.join(r"../data/" + directory, "test_full.csv"),
        )
