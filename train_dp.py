from transformers import AutoModelForCausalLM, TrainingArguments, AdamW
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
from data_formatting import get_tokenizer_and_data_collator_and_prompt_formatting
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import BitsAndBytesConfig,get_constant_schedule
import os
from opacus import PrivacyEngine

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir

    
    privacy_engine = PrivacyEngine()
    
    
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

    tokenizer, collator, formatting_func = get_tokenizer_and_data_collator_and_prompt_formatting(cfg.model.name, cfg.model.tokenizer)
    peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=cfg.model.lora.target_modules
    )
    # Add target modules here and in config
    # check this out too
    # peft_model.enable_input_require_grads()
    
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    training_arguments = TrainingArguments(
        **cfg.training_arguments,
        use_cpu=False,
        output_dir=save_path
    )
    model_init_kwargs = {
        "quantization_config":quantization_config,
        "torch_dtype":torch.bfloat16,
        "low_cpu_mem_usage":True,
        "use_cache":False,
        "trust_remote_code":True
        }
    
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name,**model_init_kwargs)
    model.train()
    # optimizer = AdamW(model.parameters())
    # sche = get_constant_schedule(optimizer)
    trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        train_dataset=train_set,
        eval_dataset=eval_set,
        max_seq_length=cfg.train.seq_length,
        formatting_func=formatting_func,
        data_collator=collator,
        peft_config=peft_config,
        args=training_arguments,
        # optimizers=[optimizer,sche], 
        # model_init_kwargs=model_init_kwargs
        
    )
    # print(trainer.train_dataloader)
    model, trainer.optimizer, trainer.train_dataloader = privacy_engine.make_private(
        module=model,
        optimizer=trainer.optimizer,
        data_loader=trainer.train_dataloader,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
        poisson_sampling = False
    )
    # trainer.model

    trainer.train()
    trainer.save_model(f"{save_path}/last")
    
if __name__ == "__main__":
    os.environ["WANDB_PROJECT"]= "LLM_memorization"
    main()