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
) -> pd.DataFrame:
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

    # drop original pair id cols
    if left_key in out.columns and right_key in out.columns:
        out.drop(columns=[left_key, right_key], inplace=True)

    out.to_csv(out_path, index=False)
    print(f"geschrieben: {out_path}  (rows={len(out)}, cols={out.shape[1]})")
    return out


def split_train_base_meta(
    train_full_df: pd.DataFrame,
    base_out_path: str,
    meta_out_path: str,
    *,
    base_frac: float = 0.40,
    label_col: str = "label",
    seed: int = 42,
) -> None:
    if not (0.0 < base_frac < 1.0):
        raise ValueError("base_frac must be between 0 and 1 (exclusive).")

    df = train_full_df.copy()
    n = len(df)
    if n == 0:
        raise ValueError("train_full_df is empty; cannot split.")

    rng = np.random.default_rng(seed)

    # Stratified split if label exists and has at least 2 classes
    if label_col in df.columns and df[label_col].notna().any():
        # group-wise sample to preserve class distribution
        base_idx = []
        for _, g in df.groupby(label_col, dropna=False):
            k = int(round(len(g) * base_frac))
            if k > 0:
                base_idx.extend(rng.choice(g.index.to_numpy(), size=k, replace=False).tolist())
        base_idx = np.array(base_idx, dtype=int)
    else:
        k = int(round(n * base_frac))
        base_idx = rng.choice(df.index.to_numpy(), size=k, replace=False)

    base_df = df.loc[base_idx].copy()
    meta_df = df.drop(index=base_idx).copy()

    # optional: reset ids inside each split (comment out if you want to keep original ids)
    base_df = base_df.reset_index(drop=True)
    meta_df = meta_df.reset_index(drop=True)

    base_df.to_csv(base_out_path, index=False)
    meta_df.to_csv(meta_out_path, index=False)

    print(f"geschrieben: {base_out_path} (rows={len(base_df)})")
    print(f"geschrieben: {meta_out_path} (rows={len(meta_df)})")


if __name__ == "__main__":
    directories = ["dblp_scholar_exp_data","abt-buy_textual", "Amazon-Google_structured", "Itunes-Amazon", "Walmart-Amazon_dirty"]
    for directory in directories:
        BASE = os.path.join("..", "data", directory, "raw")

        tableB = os.path.join(BASE, "tableB.csv")
        tableA = os.path.join(BASE, "tableA.csv")

        out_dir = os.path.join("..", "data", directory)

        # --- TRAIN (full + base/meta split) ---
        train_full_path = os.path.join(out_dir, "train_full.csv")
        train_full_df = build_full_split(
            pairs_path=os.path.join(BASE, "train.csv"),
            tableA_path=tableA,
            tableB_path=tableB,
            out_path=train_full_path,
        )

        split_train_base_meta(
            train_full_df=train_full_df,
            base_out_path=os.path.join(out_dir, "train_base.csv"),
            meta_out_path=os.path.join(out_dir, "train_meta.csv"),
            base_frac=0.40,
            label_col="label",
            seed=42,
        )

        # --- VALID ---
        build_full_split(
            pairs_path=os.path.join(BASE, "valid.csv"),
            tableA_path=tableA,
            tableB_path=tableB,
            out_path=os.path.join(out_dir, "validation_full.csv"),
        )

        # --- TEST ---
        build_full_split(
            pairs_path=os.path.join(BASE, "test.csv"),
            tableA_path=tableA,
            tableB_path=tableB,
            out_path=os.path.join(out_dir, "test_full.csv"),
        )
