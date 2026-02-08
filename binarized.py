import pandas as pd

# CSV laden
name = "big_scores_"
data = ["train", "test", "validation"]
directory = [["abt-buy_textual","ab"], ["Amazon-Google_structured","ag"], ["Itunes-Amazon", "ia"], ["Walmart-Amazon_dirty", "wm"]]
columns = ["sif", "rnn", "attention", "hybrid"]
for d in data:
        for direc in directory:

            df = pd.read_csv(f"data/{direc[0]}/{name}{d}_{direc[1]}.csv")

            for col in columns:
                df[col] = (df[col] > 0.5).astype(int)


            df.to_csv(f"data/{direc[0]}/{name}{d}_{direc[1]}_binarized.csv", index=False)