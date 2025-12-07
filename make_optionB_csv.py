import pandas as pd

master_path = "data/master_sentences.csv"
out_path = "data/textguard_optionB.csv"

df = pd.read_csv(master_path)

# add empty label columns for manual / weak labeling
df["optionB_main_theme"] = ""          # one of 10 classes
df["optionB_secondary_theme"] = ""     # optional
df["risk_flag"] = ""                   # 0/1
df["optionB_risk_type"] = ""           # usually from same set as main_theme
df["explicit_policy_reference"] = ""   # 0/1

df.to_csv(out_path, index=False)
print("Saved", out_path, "with", len(df), "rows")
