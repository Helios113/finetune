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
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset
import flwr as fl
from client import gen_client_fn, set_parameters, get_evaluate_fn
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir
    

        # Generate data partitions using Flower Datasets
    # Each client will make use of a different data partition
    partitioner = IidPartitioner(num_partitions=cfg.flower.num_clients)
    fds = FederatedDataset(
        dataset=cfg.dataset.name,
        partitioners={"train": partitioner}
    )

    # For client
    # train_set = None
    # eval_set = None
    # if cfg.train.evaluate_split:
    #     train_test = dataset.train_test_split(test_size=0.05, seed=1122)
    #     train_set = train_test["train"]
    #     eval_set = train_test["test"]
    # else:
    #     train_set = dataset
    # model = AutoModelForCausalLM.from_pretrained(cfg.model.name)

    tokenizer, collator, formatting_func = (
        get_tokenizer_and_data_collator_and_prompt_formatting(
            cfg.model.name, cfg.model.tokenizer
        )
    )
    
    
    
    save_path = "./my_fl_model"
    # Make prototype client
    client = fl.client.ClientApp(
        client_fn=gen_client_fn(
            fds,
            tokenizer,
            formatting_func,
            collator,
            cfg, # pass model config
            save_path,
        )
    )
    
    
    # Make FedAvg Strategy.
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=cfg.flower.num_clients, # total clients
        fraction_fit=cfg.flower.fraction_fit, # ratio of clients to sample
        fraction_evaluate=0.0, # No federated evaluation
        # A (optional) function used to configure a "fit()" round
        on_fit_config_fn=get_on_fit_config(),
        # A (optional) function to aggregate metrics sent by clients
        fit_metrics_aggregation_fn=fit_weighted_average,
        # A (optional) function to execute on the server after each round. 
        # In this example the function only saves the global model.
        evaluate_fn=get_evaluate_fn( 
            cfg,
            cfg.train.save_every_round,
            cfg.flower.num_rounds,
            save_path
        ),
    )
    num_rounds = cfg.flower.num_rounds
    server = fl.server.ServerApp(
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


    # Run code showing oh FL finetuning works
    client_resources = dict(cfg.flower.client_resources)
    fl.simulation.run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=cfg.flower.num_clients,
        backend_config={"client_resources": client_resources}
    )
    
   
    # Add target modules here and in config
    # check this out too
    # peft_model.enable_input_require_grads()

    # quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    # training_arguments = TrainingArguments(
    #     **cfg.training_arguments, use_cpu=False, output_dir=save_path
    # )
    # model_init_kwargs = {
    #     "quantization_config": quantization_config,
    #     "torch_dtype": torch.bfloat16,
    #     "low_cpu_mem_usage": True,
    #     "use_cache": False,
    #     "trust_remote_code": True,
    # }
    # trainer = SFTTrainer(
    #     cfg.model.name,
    #     tokenizer=tokenizer,
    #     train_dataset=train_set,
    #     eval_dataset=eval_set,
    #     max_seq_length=cfg.train.seq_length,
    #     formatting_func=formatting_func,
    #     data_collator=collator,
    #     peft_config=peft_config,
    #     args=training_arguments,
    #     model_init_kwargs=model_init_kwargs,
    # )
    # if cfg.resume:
    #     trainer.train(resume_from_checkpoint=cfg.checkpoint_path)
    # else:
    #     trainer.train()

    # trainer.save_model(f"{save_path}/last")

def get_on_fit_config():
    def fit_config_fn(server_round: int):
        fit_config = {"current_round": server_round}
        return fit_config

    return fit_config_fn

def fit_weighted_average(metrics):
    """Aggregation function for (federated) evaluation metrics."""
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"train_loss": sum(losses) / sum(examples)}


    
if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "LLM_memorization"
    main()
