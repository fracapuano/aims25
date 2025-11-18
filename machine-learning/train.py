import jax
import wandb
from network import MiniMLP
from flow_matching import MiniFlowMLP
from dataset import MiniEEG2MEGDataloader
from optimization import MiniOptimizer
from tqdm import tqdm
from utils import ModelParameters


def train_epoch(
    model: MiniFlowMLP,
    params: ModelParameters,
    dataloader: MiniEEG2MEGDataloader,
    optimizer: MiniOptimizer,
    global_state: dict,
    key: jax.random.PRNGKey = jax.random.PRNGKey(42),
) -> ModelParameters:
    """Runs a single optimization epoch, conditioned on the state of the optimizer."""
    total_steps = len(dataloader.dataset) // dataloader.batch_size
    progress_bar = tqdm(total=total_steps, desc="Training")

    training_step = global_state["training_step"]
    optimizer_state = global_state["optimizer_state"]

    for batch in dataloader:
        x = batch["meg"]  # extracting the meg part only and training a FM model on that
        loss_function = lambda params: model.forward(params=params, key=key, x=x)[0]
        wandb.log({"train/loss": loss_function(params)})

        grad_loss_function = jax.grad(loss_function)
        grads = grad_loss_function(params)

        # Perform a (dummy) update step using gradients
        optimizer_state = optimizer.update(
            params=params, grad=grads, step=training_step, **optimizer.prepare(optimizer_state)
        )
        params = optimizer_state["params"]

        training_step += 1
        progress_bar.update(1)

    global_state["training_step"] = training_step
    global_state["optimizer_state"] = optimizer_state
    global_state["params"] = params

    progress_bar.close()

    return global_state

def eval_epoch(
        model: MiniFlowMLP,
        params: ModelParameters,
        dataloader: MiniEEG2MEGDataloader,
        key: jax.random.PRNGKey = jax.random.PRNGKey(42)
    ):
    # total = total number of iterations you expect
    total_steps = len(dataloader.dataset) // dataloader.batch_size
    progress_bar = tqdm(total=total_steps, desc="Running Eval")

    for batch in dataloader:
        x = batch["meg"]  # extracting the meg part only and training a FM model on that
        loss_function = lambda params: model.forward(params=params, key=key, x=x)[0]
        wandb.log({"eval/loss": loss_function(params)})
        
        progress_bar.update(1)

    progress_bar.close()


def main():
    DIM_IN = 400
    EMBED_DIM = 128
    DIM_OUT = 400

    time_embedding_mlp_config = dict(
        in_features=1,  # time index is a scalar float
        out_features=EMBED_DIM
    )

    mlp_config = dict(
        in_features = DIM_IN + EMBED_DIM,  # stacking input dimensions and embeddings
        out_features = DIM_OUT
    )

    flow_config = dict(
        n_steps=10
    )

    training_config = {
        "time_embedding_config": time_embedding_mlp_config,
        "flow_mlp_config": time_embedding_mlp_config,
        "flow_config": flow_config
    }

    repo_id = "fracapuano/brainformer-e-small"
    train_dataloader = MiniEEG2MEGDataloader(dataset_id=repo_id)
    eval_dataloader = MiniEEG2MEGDataloader(dataset_id=repo_id, split="test")

    time_embedding_mlp = MiniMLP(**time_embedding_mlp_config)
    mlp = MiniMLP(**mlp_config)

    # wrapping the flow matching model around the two MLPs
    flow_mlp = MiniFlowMLP(mlp=mlp, mlp_time=time_embedding_mlp, **flow_config)

    key = jax.random.PRNGKey(42)
    key, train_key, eval_key = jax.random.split(key, 3)

    params = flow_mlp.init_params(key=key)

    n_epochs = 5  # minimal training
    eval_every = 3

    optimizer_config = dict(
        name = "muon",
        total_steps = (n_epochs - 1) * len(train_dataloader),
        warmup_steps = len(train_dataloader),
        kwargs={"beta": 0.99, "gamma": 0.9, "lambda_wd": 1e-3},
        peak_lr=5e-2,
    )

    training_config["optimizer_config"] = optimizer_config
    wandb.init(project="eeg2meg", config=training_config)

    optimizer = MiniOptimizer(**optimizer_config)

    training_step = 0
    global_state = {
        "params": params,
        "training_step": training_step,
        "optimizer_state": optimizer.initialize_state(params),
    }

    for epoch in range(n_epochs):
        global_state = train_epoch(flow_mlp, global_state["params"], train_dataloader, optimizer, global_state, train_key)

        if epoch % eval_every == 0:
            eval_epoch(flow_mlp, global_state["params"], eval_dataloader, eval_key)
        
        # reshuffles the dataloader
        seed = train_dataloader.seed + jax.random.randint(key=key, shape=(1,), minval=0, maxval=100).item()
        train_dataloader.refresh_iter(seed)
        key, _ = jax.random.split(key)

        training_step += 1

if __name__ == "__main__":
    main()