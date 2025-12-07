import pandas as pd

master_path = "data/master_sentences.csv"
out_path = "data/textguard_optionC.csv"

df = pd.read_csv(master_path)

df["optionC_main_category"] = ""          # one of 7 classes
df["optionC_secondary_category"] = ""     # optional
df["risk_flag"] = ""                      # 0/1
df["explicit_policy_reference"] = ""      # 0/1

df.to_csv(out_path, index=False)
print("Saved", out_path, "with", len(df), "rows")
