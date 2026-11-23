"""Train PPO or SAC with the HW4 environment, losses, and environment-step budget."""
from importlib import import_module
from accelerator import SACUpdates, host_policy, prepare, refresh, transfer
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUN = Path.cwd()
HW = HERE.parents[1] / "hw4_reinforcement_learning"
DEMOS = RUN / "demonstrations.npz"
BUDGET = {"ppo": 1024000, "sac": 1024500}
sys.path.insert(0, str(HW))


def write_json(path, obj):
    with path.open("x") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write("\n")


def fingerprint(module):
    digest = hashlib.sha256()
    for name, param in module.state_dict().items():
        digest.update(name.encode())
        digest.update(param.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def setup(method, condition):
    config = json.loads((HERE / f"{method}-config.json").read_text())
    if method == "ppo":
        from scripts import train_ppo as trainer
        official = trainer.PPO_PARAMETERS
        assert BUDGET[method] == official["total_iterations"] * official["n_steps"]
        config["total_iterations"], remainder = divmod(BUDGET[method], config["n_steps"])
        assert not remainder
    else:
        from scripts import train_sac as trainer
        official = trainer.SAC_PARAMETERS
        assert BUDGET[method] == official["learning_start_steps"] + (official["total_iterations"] - 1)*official["train_freq"]
        first_update = (config["learning_start_steps"] + config["train_freq"] - 1) // config["train_freq"]
        config["total_iterations"] = BUDGET[method] // config["train_freq"] - first_update + 1
    assert not config.get("resume_from")
    trainer.ROOT_DIR = RUN / f"{condition}-{method}"
    return trainer, config


def build(method, trainer, config, env, device="cpu"):
    cls = trainer.PPOAgent if method == "ppo" else trainer.SACAgent
    kwargs = {k: v for k, v in config.items() if k in inspect.signature(cls).parameters}
    return cls(obs_dim=env.state_dim, act_dim=env.action_dim, device=torch.device(device), **kwargs)


def clone(method):
    from torch.nn import functional as F
    tr, cfg = setup(method, "bc")
    torch.manual_seed(3517)
    env = tr.SO100RLEnv(HW / "assets/mujoco/so100_pos_ctrl.xml")
    try:
        actor = build(method, tr, cfg, env, "cuda").actor
    finally:
        env.close()
    with np.load(DEMOS) as ds:
        obs = torch.from_numpy(ds["states"]).cuda()
        commands = torch.from_numpy(ds["commands"]).cuda()
        seeds = ds["seeds"].tolist()
    assert len(seeds) == len(set(seeds)) == 2048
    assert set(seeds).isdisjoint(range(107400,107503)) and set(seeds).isdisjoint(range(118000,118100))
    n, steps, dim = obs.shape
    assert (steps,dim)==(30,19) and commands.shape==(n,6)
    mean, scale = obs.flatten(0,1).mean(0), obs.flatten(0,1).std(0).clamp_min(.05)
    obs = (obs - mean) / scale
    optim = torch.optim.AdamW(actor.parameters(), lr=.001, weight_decay=.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=80,eta_min=.00005)
    # Separate sampling generator gives both actors the same BC mini-batches.
    rng = torch.Generator().manual_seed(3517)
    for epoch in range(80):
        for _ in range(112):
            episodes = torch.randint(n,(512,),generator=rng)
            times = torch.randint(steps,(512,),generator=rng)
            times[:256] = torch.randint(6,(256,),generator=rng)
            loss = F.mse_loss(actor.act_inference(obs[episodes.cuda(),times.cuda()]),commands[episodes.cuda()])
            optim.zero_grad()
            loss.backward()
            optim.step()
        sched.step()
    layer = (actor.mu_net if method == "ppo" else actor.net)[0]
    with torch.no_grad():
        prediction=actor.act_inference(obs[:2])
        layer.weight.div_(scale)
        layer.bias.sub_(layer.weight @ mean)
        torch.testing.assert_close(prediction,actor.act_inference(obs[:2]*scale+mean),atol=3e-5,rtol=3e-5)
    # Keep the RL actor's initial log standard deviation when loading the BC weights.
    mean_state = {k:v for k,v in actor.state_dict().items() if not k.startswith("log_std")}
    torch.save(mean_state,tr.ROOT_DIR/"bc-mean.pt")
    write_json(tr.ROOT_DIR/"bc.json",dict(seed=3517,episodes=n,steps_per_demo=steps,epochs=80,batches_per_epoch=112,
                                          batch_size=512,first_six_per_batch=256,final_batch_mse=loss.item(),
                                          data_sha256=hashlib.sha256(DEMOS.read_bytes()).hexdigest()))


def train(method, condition):
    tr, cfg = setup(method, condition)
    folder = tr.ROOT_DIR
    folder.mkdir()
    (folder/"assets").symlink_to(HW/"assets",target_is_directory=True)
    write_json(folder/"settings.json",dict(method=method,condition=condition,config=cfg,environment_steps=BUDGET[method],
                                          torch=torch.__version__,device="cuda",gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,selection_steps=[163840,409600,655360,819200,BUDGET[method]],
                                          dev_seeds=[118000,118099],test_seeds=[107400,107502]))
    if condition=="bc":
        clone(method)
    agent_class = tr.PPOAgent if method=="ppo" else tr.SACAgent
    def initialize(**kwargs):
        agent = agent_class(**kwargs)
        assert agent.device.type == "cuda" and next(agent.actor.parameters()).is_cuda and next(agent.critic.parameters()).is_cuda
        before = fingerprint(agent.actor)
        if condition=="bc":
            mean_state=torch.load(folder/"bc-mean.pt",map_location="cpu",weights_only=True)
            missing,unexpected=agent.actor.load_state_dict(mean_state,strict=False)
            assert missing and all(k.startswith("log_std") for k in missing) and not unexpected
        write_json(folder/"initialization.json",dict(raw_actor=before,training_actor=fingerprint(agent.actor),
                   critic=fingerprint(agent.critic),fresh_rl_optimizers=True))
        agent.save(folder/"step-0.pt")
        return agent
    envs=[]
    class CountedEnv(tr.SO100RLEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.interactions=0
            envs.append(self)
        def step(self, action):
            result=super().step(action)
            self.interactions+=1
            return result
    course = import_module(f"ethz_starter.hw4_reinforcement_learning.scripts.train_{method}")
    scope = dict(vars(course))
    scope.update(vars(tr))
    scope.update(SO100RLEnv=CountedEnv, host_policy=host_policy, refresh=refresh, transfer=transfer, SACUpdates=SACUpdates)
    scope[method.upper()+"Agent"] = initialize
    scope[method.upper()+"_PARAMETERS"] = cfg
    source = inspect.getsource(course.main)
    if method == "sac":
        # The course budget includes 116 environment steps after the last SAC update.
        assert source.count("done=done,") == 1
        source = source.replace("done=done,", "done=terminated,")
        old = "while it < total_iterations:"
        assert source.count(old)==1
        source=source.replace(old, f"while step < {BUDGET[method]}:")
    source = prepare(source, method)
    (folder/"trainer-main.py.txt").write_text(source)
    exec(compile(source,str(tr.__file__),"exec"),scope)
    scope["main"]()
    assert len(envs)==2 and envs[0].interactions==BUDGET[method]
    write_json(folder/"completed.json",dict(training_steps=envs[0].interactions,monitoring_steps=envs[1].interactions,
                                            update_cycles=cfg["total_iterations"]))


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("method",choices=["ppo","sac"])
    parser.add_argument("condition",choices=["rl","bc"])
    args=parser.parse_args()
    assert torch.cuda.is_available() and torch.__version__=="2.11.0+cu128"
    torch.distributions.Distribution.set_default_validate_args(False)
    torch.set_num_threads(1)
    train(args.method,args.condition)
