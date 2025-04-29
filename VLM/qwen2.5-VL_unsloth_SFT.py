from unsloth import FastVisionModel  # FastLanguageModel for LLMs
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import torch
from datasets import load_dataset

dataset = load_dataset("unsloth/LaTeX_OCR", split="train")

instruction = "Write the LaTeX representation for this image."


def convert_to_conversation(sample):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["text"]}]},
    ]
    return {"messages": conversation}


converted_dataset = [convert_to_conversation(sample) for sample in dataset]

model_name = "unsloth/Qwen2.5-VL-3B-Instruct"
model, tokenizer = FastVisionModel.from_pretrained(
    model_name,
)
model = FastVisionModel.get_peft_model(
    model,
    r=128,  # The larger, the higher the accuracy, but might overfit
    lora_alpha=256,  # Recommended alpha == r at least
    lora_dropout=0.05,
    bias="none",
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
        "qkv",
        "proj",
    ],
)

FastVisionModel.for_training(model)  # Enable for training!
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(
        model, tokenizer
    ),  # train_on_responses_only=True, instruction_part="user", response_part="assistant"), # Must use!
    train_dataset=converted_dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=50,
        learning_rate=2e-4,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=5,
        optim="adamw_bnb_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # For Weights and Biases
        gradient_checkpointing=True,
        # You MUST put the below items for vision finetuning:
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,
        max_seq_length=2048,
    ),
)

# Print initial GPU memory usage
if torch.cuda.is_available():
    print(
        f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )
    print(
        f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB"
    )

trainer_stats = trainer.train()
print(trainer_stats)

# Print final GPU memory usage
if torch.cuda.is_available():
    print(
        f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )
    print(f"Final GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(
        f"Max GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )
