from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer


dataset = load_dataset("imdb")
print('====================')
print(dataset["train"][0])
print('====================')
print(dataset["train"][1])
print('====================')
print(dataset["train"][2])
