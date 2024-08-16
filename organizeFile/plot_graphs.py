from tensorboard import notebook
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, pipeline
from datasets import load_dataset
log_dir = "models/checkpoints-q2-trial2/runs"
notebook.start("--logdir {} --port 4000".format(log_dir))