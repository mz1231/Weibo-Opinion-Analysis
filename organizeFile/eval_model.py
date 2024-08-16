from tensorboard import notebook
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from accelerate import Accelerator

# Initialize the accelerator
accelerator = Accelerator()

print("Attempting to load tokenizer and model")
saved_model = "models/llama3-q2-trial1"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(saved_model, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(saved_model, torch_dtype=torch.bfloat16, local_files_only=True)

print("Loaded successfully")
# Move model to the appropriate device(s)
model = accelerator.prepare(model)

print("Loading dataset")
test_data = load_dataset('json', data_files='datasets/test-dataset-q2.json', split='train[90%:]')
print("loaded successfully")

# Prepare the pipeline with accelerator
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    truncation=True,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

def post_process_answer(generated_text):
    answer_part = generated_text.split("Answer:")[-1].strip()
    if "yes" in answer_part.lower():
        return "Yes"
    elif "no" in answer_part.lower():
        return "No"
    else:
        return None

true_positive = 0
false_positive = 0
false_negative = 0
true_negative = 0
unknowns = 0
for i in range(len(test_data)):
    prompt = test_data[i]['text']
    print(f"This is the prompt: {prompt}")

    result = pipe(prompt)
    generated_text = result[0]['generated_text']
    answer = post_process_answer(generated_text)
    label = test_data[i]['label']
    # print(f"Prompt:{i}, LLM Answer: {generated_text}, Label: {label}\n")
    print(f"Prompt:{i},  LLM Answer: {answer}, Label: {label}\n")
    
    if answer == "Yes":
        if answer == label:
            true_positive += 1
        else:
            false_positive += 1
    elif answer == "No":
        if answer == label:
            true_negative += 1
        else:
            false_negative += 1
    else:
        unknowns += 1


# Calculate precision, recall, and accuracy
answered = (len(test_data) - unknowns) / len(test_data) if len(test_data) > 0 else 0
precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
accuracy = (true_positive + true_negative) / len(test_data) if len(test_data) > 0 else 0

print(f"True Positives (correct Yes): {true_positive}")
print(f"False Positives (wrong Yes): {false_positive}")
print(f"True Negatives (correct No): {true_negative}")
print(f"False Negatives (wrong No): {false_negative}")
print(f"Unknowns (neither Yes or No): {unknowns}")
print(f"Questions Answered: {answered:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Accuracy: {accuracy:.2f}")
