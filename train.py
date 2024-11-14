from transformers import AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
from data_formatting import get_tokenizer_and_data_collator_and_prompt_formatting
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import BitsAndBytesConfig
import os
import wandb



@hydra.main(config_path="conf", config_name="config_pythia.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    save_path = str(HydraConfig.get().runtime.output_dir)
    run = wandb.init(
        project="llm-memorization", id=cfg.wandb_id, resume="allow"
    )

    dataset = load_dataset(cfg.dataset.name, split="train")
    train_set = None
    eval_set = None
    if cfg.train.evaluate_split:
        train_test = dataset.train_test_split(test_size=0.05, seed=1122)
        train_set = train_test["train"]
        eval_set = train_test["test"]
    else:
        train_set = dataset
    # model = AutoModelForCausalLM.from_pretrained(cfg.model.name)

    tokenizer, collator, formatting_func = (
        get_tokenizer_and_data_collator_and_prompt_formatting(
            cfg.model.name, cfg.model.tokenizer
        )
    )
    target_modules= None if cfg.model.lora.target_modules is None else OmegaConf.to_object(cfg.model.lora.target_modules)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules 
    )
    # Add target modules here and in config
    # check this out too
    # peft_model.enable_input_require_grads()

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    training_arguments = TrainingArguments(
        **OmegaConf.to_object(cfg.training_arguments), use_cpu=False, output_dir=save_path
    )
    model_init_kwargs = {
        "quantization_config": quantization_config,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "use_cache": False,
        "trust_remote_code": True,
    }
    trainer = SFTTrainer(
        cfg.model.name,
        tokenizer=tokenizer,
        train_dataset=train_set,
        eval_dataset=eval_set,
        max_seq_length=cfg.train.seq_length,
        formatting_func=formatting_func,
        data_collator=collator,
        peft_config=peft_config,
        args=training_arguments,
        model_init_kwargs=model_init_kwargs,
    )
    if cfg.resume:
        trainer.train(resume_from_checkpoint=cfg.checkpoint_path)
    else:
        trainer.train()

    print(f"I am saving the model to {save_path}/last")
    trainer.save_model(f"{save_path}/last")


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "llm-memorization"
    main()
