from datasets import Dataset, DatasetDict
import json
import pandas as pd

# build
df = pd.read_csv("self_actualization_prompts.csv")
ds = Dataset.from_pandas(df)


# publish
DatasetDict({"train": ds}).push_to_hub("cosmicoptima/introspection-prompts", private=False)