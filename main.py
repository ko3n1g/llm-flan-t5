import os
import time

import matplotlib.pyplot as plt
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Disable parallelism and avoid the warning message
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define model IDs
model_ids = ["google/flan-t5-large", "google/flan-t5-xl"]

base_prompt = "For the following sentence, explain in your own words what this sentence is about: \n\n"

# Define prompts and types
prompts = ["Apple iPhone XS 64Gb"]

types = [
    "Knowledge Retrieval",
    "Knowledge Retrieval",
    "Knowledge Retrieval",
    "Logical Reasoning",
    "Cause and Effect",
    "Analogical Reasoning",
    "Inductive Reasoning",
    "Deductive Reasoning",
    "Counterfactual Reasoning",
    "In Context",
]

# Create empty lists to store generation times, model load times, tokenizer load times, and pipeline load times
xl_generation_times = []
large_generation_times = []

xl_model_load_times = []
large_model_load_times = []

xl_tokenizer_load_times = []
large_tokenizer_load_times = []

xl_pipeline_load_times = []
large_pipeline_load_times = []

prompt_types = []

for model_id in model_ids:
    # Load tokenizer
    tokenizer_start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer_end_time = time.time()

    # Load model
    model_start_time = time.time()
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model_end_time = time.time()

    # Load pipeline
    pipe_start_time = time.time()
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    pipe_end_time = time.time()

    # Store loading times
    if model_id == "google/flan-t5-large":
        large_model_load_times.append(model_end_time - model_start_time)
        large_tokenizer_load_times.append(
            tokenizer_end_time - tokenizer_start_time
        )
        large_pipeline_load_times.append(pipe_end_time - pipe_start_time)
    elif model_id == "google/flan-t5-xl":
        xl_model_load_times.append(model_end_time - model_start_time)
        xl_tokenizer_load_times.append(
            tokenizer_end_time - tokenizer_start_time
        )
        xl_pipeline_load_times.append(pipe_end_time - pipe_start_time)

    # Print model results
    print()
    print(f"Results for model: {model_id}")
    print("=" * 30)

    # Loop thru prompt list, measure the time to the generate answers, print prompt, answer, time, type
    for i, prompt in enumerate(prompts):
        prompt = f"{base_prompt} + '{prompt}'"
        start_time = time.time()
        answer = local_llm(prompt)
        end_time = time.time()
        print(f"Prompt: {prompt}")
        print(f"Answer: {answer}")
        print(f"Generation Time: {end_time - start_time:.5f} seconds")
        print(f"Type: {types[i]}")
        print()

        # store prompt types and time measures to generate ansswers by prompt types
        prompt_types.append(types[i])  # Store the prompt type

        if model_id == "google/flan-t5-large":
            large_generation_times.append(end_time - start_time)
        elif model_id == "google/flan-t5-xl":
            xl_generation_times.append(end_time - start_time)

    # print loading times
    print(f"Loading times for model {model_id}")
    print(
        "Tokenizer Loading Time:",
        f"{tokenizer_end_time - tokenizer_start_time:.5f}",
        "seconds",
    )
    print(
        "Model Loading Time:",
        f"{model_end_time - model_start_time:.5f}",
        "seconds",
    )
    print(
        "Pipeline Loading Time:",
        f"{pipe_end_time - pipe_start_time:.5f}",
        "seconds\n\n",
    )


# Plot model load times
model_load_times = [sum(xl_model_load_times), sum(large_model_load_times)]
model_labels = ["XL Model", "Large Model"]

plt.figure(figsize=(18, 6))
plt.subplot(131)
plt.bar(model_labels, model_load_times, color=["blue", "orange"])
plt.ylabel("Load Time (seconds)")
plt.xlabel("Model")
plt.title("Model Load Time Comparison")

# Plot tokenizer load times
tokenizer_load_times = [
    sum(xl_tokenizer_load_times),
    sum(large_tokenizer_load_times),
]

plt.subplot(132)
plt.bar(model_labels, tokenizer_load_times, color=["blue", "orange"])
plt.ylabel("Load Time (seconds)")
plt.xlabel("Model")
plt.title("Tokenizer Load Time Comparison")

# Plot pipeline load times
pipeline_load_times = [
    sum(xl_pipeline_load_times),
    sum(large_pipeline_load_times),
]
plt.subplot(133)
plt.bar(model_labels, pipeline_load_times, color=["blue", "orange"])
plt.ylabel("Load Time (seconds)")
plt.xlabel("Model")
plt.title("Pipeline Load Time Comparison")

# Plot generation times
plt.figure(figsize=(9, 6))
plt.barh(
    range(len(types)),
    xl_generation_times,
    height=0.4,
    align="center",
    color="blue",
    label="XL Model",
)
plt.barh(
    [x + 0.4 for x in range(len(types))],
    large_generation_times,
    height=0.4,
    align="center",
    color="orange",
    alpha=0.5,
    label="Large Model",
)
plt.yticks(range(len(types)), types)
plt.ylabel("Type")
plt.xlabel("Generation Time (seconds)")
plt.title("Generation Time Comparison")
plt.legend()

plt.tight_layout()
plt.show()
