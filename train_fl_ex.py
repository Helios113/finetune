import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import flwr as fl
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from datasets import load_dataset
from flwr.client.mod import fixedclipping_mod
from flwr.server.strategy import (
    DifferentialPrivacyClientSideFixedClipping
)

# Generate data partitions using Flower Datasets
# Each client will make use of a different data partition
partitioner = IidPartitioner(num_partitions=cfg.flower.num_clients)
fds = FederatedDataset(
    dataset=cfg.dataset.name,
    partitioners={"train": partitioner}
)



(
tokenizer,
data_collator,
formatting_prompts_func,
) = get_tokenizer_and_data_collator_and_propt_formatting(
    cfg.model.name, 
    cfg.train.padding_side,
)

# 
# 
# 


save_path = "./my_fl_model"
client = fl.client.ClientApp(
    client_fn=gen_client_fn(
        fds,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        cfg.model, # pass model config
        cfg.train, # pass train config
        save_path,
    ),
    mods=[fixedclipping_mod] # We pass a "mod" to enable client-side DP
)


# Instantiate strategy according to config.
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
        cfg.model,
        cfg.train.save_every_round,
        cfg.flower.num_rounds,
        save_path
    ),
)

# Make FedAvg DP-ready with a wrapper class
sampled_clients = cfg.flower.num_clients*strategy.fraction_fit
strategy = DifferentialPrivacyClientSideFixedClipping(
    strategy, 
    noise_multiplier=cfg.flower.dp.noise_mult,
    clipping_norm=cfg.flower.dp.clip_norm, 
    num_sampled_clients=sampled_clients
)

# ServerApp definition takes a Flower strategy
# and a config specifying the number of rounds.
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


# You can launch the full FL finetuning 
# using a larger Mistral-7B by running
# the same code as above but loading config 
# `federated_full.yaml`at the top of this notebook:
#
# cfg = get_config("federated_full")
#
# !! If you decide to do that, it's recommended running 
# the finetuning on a machine with one (or more) GPUs.
# Precisely these steps were used to train the FL model
# you'll make use of through fireworks.ai in the next section.