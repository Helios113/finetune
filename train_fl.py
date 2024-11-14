from data_formatting import get_tokenizer_and_data_collator_and_prompt_formatting
from omegaconf import OmegaConf
import os
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset
import flwr as fl
from client import gen_client_fn, get_evaluate_fn
import copy
import datasets
import wandb
from utils import RandomOrgClientManager

import argparse
parser = argparse.ArgumentParser(description="Train a model with the given config.")    
parser.add_argument(
    '--config_name', 
    type=str, 
    default='fl_config_pythia.yaml', 
    help='The name of the config file to use (default: fl_config_pythia.yaml)'
)
args = parser.parse_args()

def fit_config_fn(server_round: int):
    fit_config = {"current_round": server_round}
    return fit_config

def fit_weighted_average(metrics):
    """Aggregation function for (federated) evaluation metrics."""
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"train_loss": sum(losses) / sum(examples)}


os.environ["WANDB_PROJECT"] = "llm-memorization"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

save_path = "conf"
cfg = OmegaConf.load(save_path + "/" + args.config_name)
tokenizer, collator, formatting_func = (
    get_tokenizer_and_data_collator_and_prompt_formatting(
        cfg.model.name, cfg.model.tokenizer
    )
)
# Generate data partitions using Flower Datasets
# Each client will make use of a different data partition
partitioner = IidPartitioner(num_partitions=cfg.flower.num_clients)
# fds = FederatedDataset(
#     dataset=medalpaca/medical_meadow_medical_flashcards, partitioners={"train": partitioner}, data_files="/nfs-share/pa511/llm_memorisation/datasets/medical_dataset/deduplicated_medical_meadow_flashcards.json"
# )

fds = FederatedDataset(
    dataset=cfg.dataset.name, partitioners={"train": partitioner}
    , data_files=cfg.dataset.files # TODO: there is no such argument in my Federated Dataset class
)
cfg.run_id = wandb.util.generate_id()

train_sets = []
eval_sets = []
cen_eval_set = None
print(fds)

for i in range(cfg.flower.num_clients):
    # print(i)
    ds = fds.load_partition(i, "train")
    train_ds = ds.train_test_split(test_size=0.05, seed=1111)
    train_set = train_ds["train"]
    eval_set = train_ds["test"]
    train_sets.append(copy.deepcopy(train_set))
    eval_sets.append(copy.deepcopy(eval_set))
cen_eval_set = datasets.concatenate_datasets(eval_sets)



save_path = "./my_model_medical_centralized"
# Make prototype client
client = fl.client.ClientApp(
    client_fn=gen_client_fn(
        (eval_sets, train_sets),
        tokenizer,
        formatting_func,
        collator,
        cfg,  # pass model config
        save_path
    )
)

# Make FedAvg Strategy.
strategy = fl.server.strategy.FedAvg(
    min_available_clients=cfg.flower.num_clients,  # total clients
    fraction_fit=cfg.flower.fraction_fit,  # ratio of clients to sample
    fraction_evaluate=0.0,
    on_fit_config_fn=fit_config_fn,
    evaluate_fn=get_evaluate_fn(
        cfg,
        cfg.train.save_every_round,
        cfg.flower.num_rounds,
        save_path,
        tokenizer,
        cen_eval_set,
        formatting_func,
        collator
    ),
)
client_manager = RandomOrgClientManager("2e4c64c1-80ab-42b9-8924-9576140f571e")

num_rounds = cfg.flower.num_rounds

server = fl.server.ServerApp(
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy, 
    client_manager=client_manager
)

def main():
    client_resources = dict(cfg.flower.client_resources)
    fl.simulation.run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=cfg.flower.num_clients,
        backend_config={"client_resources": client_resources},
    )

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()

