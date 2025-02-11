import jax
import wandb
import hydra
import omegaconf
import jax.numpy as jnp
from evaluation import eval_model
from data.canonical_mcmc import Canonical_Sampler
import os
import sys
import optax
import time
import pickle
from dataloader import DataLoader


@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_devices
    #### jax flags ###
    for cfg_name, cfg_value in cfg.jax_config.items():
        jax.config.update(cfg_name, cfg_value)
    try:
        wandb_key = open("./wandb.key", "r").read()
        wandb.login(key=wandb_key)
        run = wandb.init(project=cfg.wandb_project_name)
    except:
        print("Weights and biases key not found or not valid. Will be logging locally.")
        run = wandb.init(project=cfg.wandb_project_name, mode="offline")
    wandb.run.log_code("..")
    wandb.config.update(
        omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    print("devices: ", *jax.devices())

    target_system = hydra.utils.instantiate(cfg.target_system)

    run.tags = run.tags + (f"{target_system.num_dim}D",)
    for N in cfg.eval_N_list + cfg.train_N_list:
        assert N in cfg.data_to_generate
    for i, N in enumerate(cfg.data_to_generate):
        sampler = Canonical_Sampler(target_system, N=N)
        sampler.sample(key=jax.random.PRNGKey(cfg.PRNGKey), dx=cfg.sampling_dx[i])

    print(80 * "-")
    print("Preparing data, this might take a few minutes...")
    eval_dataloaders = []
    for i, N in enumerate(cfg.eval_N_list):
        with open(target_system.data_path + f"_N={N}", "rb") as pickle_file:
            x = pickle.load(pickle_file)
        eval_dataloaders.append(DataLoader(x, batch_size=cfg.eval_batch_size[i]))
    train_x = []
    train_n = []
    for i, N in enumerate(cfg.train_N_list):
        with open(target_system.data_path + f"_N={N}", "rb") as pickle_file:
            x = pickle.load(pickle_file)
        padding_shape = (len(x), max(cfg.train_N_list) - N, x.shape[-1])
        x = jnp.concatenate((x, jnp.zeros(padding_shape)), 1)
        train_x.append(x)
        train_n.append(jnp.full((len(x), 1), N))

    train_x = jnp.concatenate(train_x)
    train_n = jnp.concatenate(train_n)
    train_loader = DataLoader(train_x, train_n, batch_size=cfg.batch_size)

    print("Done.")
    print(80 * "-")

    ddpm = hydra.utils.instantiate(cfg.model)
    ddpm.num_features = target_system.num_dim
    ddpm.E_model = hydra.utils.instantiate(cfg.E_model)
    ddpm.init_params(
        key=jax.random.PRNGKey(cfg.PRNGKey + 1), maxN=max(cfg.train_N_list)
    )

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(ddpm.params))
    print(f"num params: {num_params/1000:.1f}K")

    ## train
    optim = hydra.utils.instantiate(cfg.optim)
    opt_state = optim.init(ddpm.params)
    key = jax.random.PRNGKey(cfg.PRNGKey + 2)
    params = ddpm.params

    target_name = cfg.target_system._target_.split(".")[-1]
    if "model_name" in cfg.keys():
        target_name += f'.{cfg["model_name"]}'
    ckpt_path = f"{target_name}.ckpt"

    @jax.jit
    def update_step(key, params, batch, opt_state):
        loss_and_grad_fn = jax.value_and_grad(ddpm.loss_fn)
        loss, grad = loss_and_grad_fn(params, batch, key)
        updates, opt_state = optim.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    def eval_step():

        ## save params
        file = open(f"{ckpt_path}", "wb")
        pickle.dump({"params": params, "opt_state": opt_state, "cfg": cfg}, file)
        file.close()
        wandb.save(f"{ckpt_path}", policy="now")

        ## logs
        ddpm.params = params
        print("\nSampling... ", end=" ")
        for i, loader in enumerate(eval_dataloaders):
            print(f"{cfg.eval_N_list[i]}", end=" ")
            sys.stdout.flush()
            eval_model(loader, ddpm, target_system, cfg.eval_num_batches[i])
        print("Done.")

    start = time.time()
    for s in range(1, cfg.num_train_steps + 1):

        x_batch, n_batch = train_loader.next()
        key = jax.random.split(key, 2)[0]
        loss, params, opt_state = update_step(
            key, params, (x_batch, n_batch), opt_state
        )
        wandb.log({"loss": loss})
        print(f"training progress:  {s/cfg.num_train_steps:.3f}", end="    ")
        print(f"time: {time.time()-start:.2f}s", end="    ")
        print(f"loss: {loss:.4f}", end="    \r")
        if s % cfg.eval_every_n_steps == 0:
            eval_step()

    print(80 * "=")
    file = open(f"{ckpt_path}", "wb")
    pickle.dump({"params": params, "opt_state": opt_state, "cfg": cfg}, file)
    file.close()
    wandb.save(f"{ckpt_path}", policy="now")
    wandb.save(
        f"eval_logs_{target_name}",
        policy="now",
    )
    run.finish()


if __name__ == "__main__":
    main()
