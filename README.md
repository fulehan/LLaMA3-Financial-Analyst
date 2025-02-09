# LLaMa3 Financial Analyst

![LLM Financial Analyst](https://img.shields.io/badge/LLM-Llama3-orange) ![RAG Pipeline](https://img.shields.io/badge/Architecture-RAG-blue) ![SEC Data](https://img.shields.io/badge/Data-10--K-green)

A financial analysis AI system by fine-tuning Llama-3-8B with LoRA on 10-K Q&A data, paired with a RAG pipeline that retrieves real-time SEC filings. The system processes Sections 1A (Risk Factors) and 7 (MD&A) using open-source embeddings (BAAI/bge-large-en) to provide contextually accurate answers to complex financial queries. Combines parameter-efficient LLM training with semantic search for scalable enterprise analysis.


## Overview
This project combines:
1. Fine-tuned a LLaMa-3 model **meta-llama/Meta-Llama-3-8B-Instruct** with **LoRA adapters** on 10-K Q&A data using Unsloth AI framework.
2. An **SEC data pipeline** for retrieving 10-K filings in real-time using the SEC API.
3. Local embeddings with **BAAI/bge-large-en-v1.5** model for semantic search to provide contextually accurate answers to complex financial queries.
4. In-memory vector storage for contextual retrieval of 10-K filings.
5. A **RAG pipeline** to inject context into the LLM's inference process.

The system answers financial questions using relevant context from SEC 10-K reports (specifically Sections 1A and 7).

## Features
- 🦙 Parameter-efficient fine-tuning with LoRA adapters
- 📈 SEC API integration for real-time 10-K retrieval
- 🔍 Semantic search using open-source embeddings
- 💡 End-to-end RAG pipeline implementation


## Installation

1. Clone repository:
```bash
git clone https://github.com/mirabdullahyaser/LLaMA3-Financial-Analyst.git
cd llm-financial-analyst
```

2. Install dependencies:
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes
pip install sec_api
pip install -U langchain
pip install -U langchain-community
pip install -U sentence-transformers
pip install -U faiss-gpu-cu12
```

3. Setup environment variables:
```python
from google.colab import userdata

# HuggingFace token, required for accessing gated models (like LLaMa 3 8B Instruct)
hf_token = userdata.get("HUGGINGFACEHUB_API_KEY")
# SEC-API Key
sec_api_key = userdata.get("SEC_API_KEY")
```

## Usage

### Part 1: Fine Tuning

1. Initializing the LLaMa-3 model ****meta-llama/Meta-Llama-3-8B-Instruct**. We will be using the built in GPU on Colab to do all our fine tuning needs, using the [Unsloth Library](https://github.com/unslothai/unsloth). Much of the below code is augmented from [Unsloth Documentation!](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=AEEcJ4qfC7Lp)


```python    
# Load the model and tokenizer from the pre-trained FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    # Specify the pre-trained model to use
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct",
    # Specifies the maximum number of tokens (words or subwords) that the model can process in a single forward pass
    max_seq_length = 2048,
    # Data type for the model. None means auto-detection based on hardware, Float16 for specific hardware like Tesla T4
    dtype = None,
    # Enable 4-bit quantization, By quantizing the weights of the model to 4 bits instead of the usual 16 or 32 bits, the memory required to store these weights is significantly reduced. This allows larger models to be run on hardware with limited memory resources.
    load_in_4bit = True,
    # Access token for gated models, required for authentication to use models like Meta-Llama-2-7b-hf
    token = hf_token,
)
```

2. Adding LoRA adapters to the model for parameter-efficient fine-tuning. LoRA, or Low-Rank Adaptation, is a technique used in machine learning to fine-tune large models more efficiently. It works by adding a small, additional set of parameters to the existing model instead of retraining all the parameters from scratch. This makes the fine-tuning process faster and less resource-intensive. Essentially, LoRA helps tailor a pre-trained model to specific tasks or datasets without requiring extensive computational power or memory.

```python
# Apply LoRA (Low-Rank Adaptation) adapters to the model for parameter-efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    # Rank of the adaptation matrix. Higher values can capture more complex patterns. Suggested values: 8, 16, 32, 64, 128
    r = 16,
    # Specify the model layers to which LoRA adapters should be applied
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    # Scaling factor for LoRA. Controls the weight of the adaptation. Typically a small positive integer
    lora_alpha = 16,
    # Dropout rate for LoRA. A value of 0 means no dropout, which is optimized for performance
    lora_dropout = 0,
    # Bias handling in LoRA. Setting to "none" is optimized for performance, but other options can be used
    bias = "none",
    # Enables gradient checkpointing to save memory during training. "unsloth" is optimized for very long contexts
    use_gradient_checkpointing = "unsloth",
    # Seed for random number generation to ensure reproducibility of results
    random_state = 3407,
)
```

3. Prepare the dataset for fine-tuning. We will be using a Hugging Face dataset of Financial Q&A over form 10ks, provided by user [Virat Singh](https://github.com/virattt) here https://huggingface.co/datasets/virattt/llama-3-8b-financialQA

The following code below formats the entries into the prompt defined first for training, being careful to add in special tokens. In this case our End of Sentence token is <|eot_id|>. More LLaMa 3 special tokens [here](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/)

```python
# Defining the expected prompt
ft_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Below is a user question, paired with retrieved context. Write a response that appropriately answers the question,
include specific details in your response. <|eot_id|>

<|start_header_id|>user<|end_header_id|>

### Question:
{}

### Context:
{}

<|eot_id|>

### Response: <|start_header_id|>assistant<|end_header_id|>
{}"""

# Grabbing end of sentence special token
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

# Function for formatting above prompt with information from Financial QA dataset
def formatting_prompts_func(examples):
    questions = examples["question"]
    contexts       = examples["context"]
    responses      = examples["answer"]
    texts = []
    for question, context, response in zip(questions, contexts, responses):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = ft_prompt.format(question, context, response) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

dataset = load_dataset("virattt/llama-3-8b-financialQA", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)
```

4. Defining the Trainer Arguments for fine-tuning. We will be setting up and using HuggingFace Transformer Reinforcement Learning (TRL)'s [Supervised Fine-Tuning Trainer](https://huggingface.co/docs/trl/sft_trainer)

**Supervised fine-tuning** is a process in machine learning where a pre-trained model is further trained on a specific dataset with labeled examples. During this process, the model learns to make predictions or classifications based on the labeled data, improving its performance on the specific task at hand. This technique leverages the general knowledge the model has already acquired during its initial training phase and adapts it to perform well on a more targeted set of examples. Supervised fine-tuning is commonly used to customize models for specific applications, such as sentiment analysis, object recognition, or language translation, by using task-specific annotated data.

```python
trainer = SFTTrainer(
    # The model to be fine-tuned
    model = model,
    # The tokenizer associated with the model
    tokenizer = tokenizer,
    # The dataset used for training
    train_dataset = dataset,
    # The field in the dataset containing the text data
    dataset_text_field = "text",
    # Maximum sequence length for the training data
    max_seq_length = 2048,
    # Number of processes to use for data loading
    dataset_num_proc = 2,
    # Whether to use sequence packing, which can speed up training for short sequences
    packing = False,
    args = TrainingArguments(
        # Batch size per device during training
        per_device_train_batch_size = 2,
        # Number of gradient accumulation steps to perform before updating the model parameters
        gradient_accumulation_steps = 4,
        # Number of warmup steps for learning rate scheduler
        warmup_steps = 5,
        # Total number of training steps
        max_steps = 60,
        # Number of training epochs, can use this instead of max_steps, for this notebook its ~900 steps given the dataset
        # num_train_epochs = 1,
        # Learning rate for the optimizer
        learning_rate = 2e-4,
        # Use 16-bit floating point precision for training if bfloat16 is not supported
        fp16 = not is_bfloat16_supported(),
        # Use bfloat16 precision for training if supported
        bf16 = is_bfloat16_supported(),
        # Number of steps between logging events
        logging_steps = 1,
        # Optimizer to use (in this case, AdamW with 8-bit precision)
        optim = "adamw_8bit",
        # Weight decay to apply to the model parameters
        weight_decay = 0.01,
        # Type of learning rate scheduler to use
        lr_scheduler_type = "linear",
        # Seed for random number generation to ensure reproducibility
        seed = 3407,
        # Directory to save the output models and logs
        output_dir = "outputs",
    ),
)
```

5. Training the model

```python
trainer_stats = trainer.train()
```

### Part 2: RAG Pipeline


## Results

Example financial Q&A:

```bash
User Query: What region contributes most to international sales?
LLaMa3 Agent: Europe

User Query: What are significant announcements of products during fiscal year 2023?
LLaMa3 Agent: During fiscal year 2024, the Company announced the following significant products: MacBook Pro 14-in.

User Query: What are significant announcements of products during fiscal year 2023?
LLaMa3 Agent: Significant product announcements during fiscal year 2023 included the following: MacBook Pro 14-in.

```

## References

- [Dataset](https://huggingface.co/datasets/virattt/llama-3-8b-financialQA)
- [LLaMa 3](https://huggingface.co/docs/transformers/model_doc/llama3)
- [Unsloth AI](https://github.com/unslothai/unsloth)
- [Supervised fine-tuning](https://huggingface.co/docs/trl/sft_trainer)
- [LLaMa 3 special tokens](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/)
- [Beijing Academy of Artificial Intelligence's - Large English Embedding Model](https://huggingface.co/BAAI/bge-large-en-v1.5)

## Contributing

Contributions welcome! Please open an issue first to discuss proposed changes.

## Contact
For questions or suggestions, please contact Mir Abdullah Yaser via [GitHub Issues](https://github.com/mirabdullahyaser/LLaMA3-Financial-Analyst/issues) or [mirabdullahyaser@example.com](mailto:mirabdullahyaser@example.com).