from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.utils import prepare_model_for_kbit_training
import math
import torch
import flwr as fl
from omegaconf import DictConfig, OmegaConf
from typing import List
from flwr.common.typing import NDArrays, Scalar
from typing import Dict, Tuple
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
)
from collections import OrderedDict
from trl import SFTTrainer
import logging
import wandb
logger = logging.getLogger("ray.serve")
from transformers import TrainerCallback, TrainerControl, TrainerState
import os

class CustomSFTTrainer(SFTTrainer):
    def train(self, resume_from_checkpoint=None, trial=None, **kwargs):
        # Initialize necessary variables
        self.global_step = 0
        self.epoch = 0
        tr_loss = torch.tensor(0.0).to(self.args.device)
        grad_norm = None
        ignore_keys_for_eval = getattr(self.model, "config", None)

        # Log initial loss before training
        _ = self._maybe_log_save_evaluate(
            tr_loss, grad_norm, self.model, trial, self.epoch, ignore_keys_for_eval
        )

        # Proceed with regular training
        return super().train(resume_from_checkpoint, trial, **kwargs)


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        cfg: DictConfig,
        dataset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        save_path,
        cid,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.logger = logging.getLogger("ray.serve")
        self.training_arguments = TrainingArguments(
            **OmegaConf.to_object(cfg.training_arguments),
            use_cpu=False,
            output_dir=save_path,
        )
        self.tokenizer = tokenizer
        self.formatting_func = formatting_prompts_func
        self.data_collator = data_collator
        self.save_path = save_path
        self.model = get_model(cfg)
        self.dataset = dataset
        self.cid = cid

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""

        state_dict = get_peft_model_state_dict(self.model)
        return [val.cpu().numpy() for _, val in state_dict.items()]

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        self.logger.error(f"Client id: {self.cid}")

        run = wandb.init(
            project="test-fl-memorization",
            id=f"{self.cfg.run_id}_{self.cfg.wandb_client_id}_{self.cid}",
            group=f"{self.cfg.run_id}_{self.cfg.group_id}",
            reinit=True,
            resume="allow",
        )
        wandb.define_metric("iter")
        # set all other train/ metrics to use this step
        wandb.define_metric("train/*", step_metric="iter")
        wandb.define_metric("eval/*", step_metric="iter")

        """Implement distributed fit function for a given client."""
        set_parameters(self.model, parameters)

        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.cfg.flower.num_rounds,
            self.cfg.train.learning_rate_max,
            self.cfg.train.learning_rate_min,
        )

        self.training_arguments.learning_rate = new_lr
        self.training_arguments.output_dir = self.save_path

        train_set = self.dataset[1]
        eval_set = self.dataset[0]

        # Add target modules here and in config
        # check this out too
        # peft_model.enable_input_require_grads()
        training_arguments = TrainingArguments(
            **OmegaConf.to_object(self.cfg.training_arguments),
            use_cpu=False,
            output_dir=self.save_path,
        )

        trainer = CustomSFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_set,
            eval_dataset=eval_set,
            max_seq_length=self.cfg.train.seq_length,
            formatting_func=self.formatting_func,
            data_collator=self.data_collator,
            args=training_arguments,
            callbacks=[
                CustomEvalCallback(
                    run,
                    step_offset=(
                        (int(config["current_round"]) - 1)
                        * int(self.cfg.training_arguments.max_steps)
                    ),
                )
            ],
        )
        trainer.evaluate()
        results = None
        if self.cfg.resume:
            results = trainer.train(resume_from_checkpoint=self.cfg.checkpoint_path)
        else:
            results = trainer.train()

        # trainer.save_model(f"{self.save_path}/last")

        metrics = {}
        # if self.cfg.train.evaluate_split:
        #     eval_res = trainer.evaluate()
        #     metrics["eval_loss"] = eval_res["eval_loss"]

        metrics = {**metrics, "train_loss": results.training_loss}

        return (
            self.get_parameters({}),
            len(self.dataset),
            metrics,
        )


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def gen_client_fn(
    fds,
    tokenizer,
    formatting_prompts_func,
    data_collator,
    cfg: DictConfig,
    save_path: str,
    # unique_id,
):  # pylint: disable=too-many-arguments
    """Generate the client function that creates the Flower Clients."""

    def client_fn(cid: str) -> fl.client.Client:
        """Create a Flower client representing a single organization."""

        # Let's get the partition corresponding to the i-th client
        client_dataset = (fds[0][int(cid)], fds[1][int(cid)])
        return FlowerClient(
            cfg,
            client_dataset,
            tokenizer,
            formatting_prompts_func,
            data_collator,
            save_path,
            cid,
        ).to_client()

    return client_fn


def set_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)


def get_evaluate_fn(
    cfg,
    save_every_round,
    total_round,
    save_path,
    tokenizer,
    eval_dataset,
    formatting_func,
    data_collator,
):
    """Return an evaluation function for saving global model."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):
        run = wandb.init(
            project="test-fl-memorization",
            id=f"{cfg.run_id}_{cfg.wandb_server_id}",
            group=f"{cfg.run_id}_{cfg.group_id}",
            reinit=True,
            resume="allow",
        )
        wandb.define_metric("iter")
        # set all other train/ metrics to use this step
        wandb.define_metric("train/*", step_metric="iter")
        wandb.define_metric("eval/*", step_metric="iter")
        logger.error("In eval function")
        logger.error(f"Config: {config}")
        logger.error(f"CFG: {cfg}")

        
        model = get_model(cfg)
        set_parameters(model, parameters)
        logger.error("params set")

        training_arguments = TrainingArguments(
            **OmegaConf.to_object(cfg.training_arguments),
            use_cpu=False,
            output_dir=save_path,
        )

        trainer = CustomSFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=eval_dataset,
            eval_dataset=eval_dataset,
            max_seq_length=cfg.train.seq_length,
            formatting_func=formatting_func,
            data_collator=data_collator,
            args=training_arguments,
            callbacks=[
                CustomEvalCallback(
                    run,
                    step_offset=(server_round) * int(cfg.training_arguments.max_steps),
                )
            ],
        )

        trainer.evaluate()

        model.save_pretrained(f"{save_path}/peft_{server_round}")

        return 0.0, {}
        return 0.0, {"accuracy": 0.1}

    return evaluate


def get_model(cfg: DictConfig):
    """Load model with appropriate quantization config and
    other optimizations."""

    use_cuda = torch.cuda.is_available()
    quantization_config = None
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    # instantiate model
    model_init_kwargs = {
        "quantization_config": quantization_config,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "use_cache": False,
        "trust_remote_code": True,
    }

    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **model_init_kwargs)

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.model.gradient_checkpointing
    )
    tm = None
    if cfg.model.lora.target_modules:
        tm = OmegaConf.to_object(cfg.model.lora.target_modules)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=tm,
    )

    peft_model = get_peft_model(model, peft_config)
    if not (use_cuda):
        peft_model.enable_input_require_grads()

    if cfg.model.gradient_checkpointing:
        model.config.use_cache = False

    return peft_model


class CustomEvalCallback(TrainerCallback):
    def __init__(self, wandb_run, step_offset):
        self.step_offset = step_offset
        self.wandb_run = wandb_run
        self.first_iteration_done = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.wandb_run is None:
            return

        # Offset the step
        adjusted_step = state.global_step + self.step_offset

        # Log metrics to wandb
        log_dict = {}
        for key, value in metrics.items():
            log_dict[f"eval/{key}"] = value
        log_dict["iter"] = adjusted_step
        self.wandb_run.log(log_dict)

    def on_log(self, args, state, control, logs: Dict[str, float] = None, **kwargs):
        if logs is None or self.wandb_run is None or (state.global_step % 10!=0 and state.global_step!=1):
            return

        # Offset the step
        adjusted_step = state.global_step + self.step_offset
        # Log metrics to wandb
        log_dict = {}
        for key, value in logs.items():
            log_dict[f"train/{key}"] = value
        log_dict["iter"] = adjusted_step
        self.wandb_run.log(log_dict)

    # def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     if not self.first_iteration_done:
    #         # Get the loss from the last logged step
    #         print("HERE!:",state)
    #         if state.log_history:
    #             last_log = state.log_history[-1]
    #             first_iteration_loss = last_log['loss']
                
    #             # Offset the step
    #             adjusted_step = state.global_step + self.step_offset
    #             # Log metrics to wandb
    #             log_dict = {}
    #             for key, value in logs.items():
    #                 log_dict[f"train/{key}"] = value
    #             log_dict["iter"] = adjusted_step
    #             self.wandb_run.log(log_dict)
    #         else:
    #             print("No log history available")

    #         self.first_iteration_done = True

    # def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     # Reset the flag at the beginning of training
    #     self.first_iteration_done = False