"""Re-evaluate the frozen DQN on the archived CartPole seeds."""
import sys
from pathlib import Path
import torch

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[1] / "hw4_reinforcement_learning"))
from envs.cartpole_wrapper import CartPoleWrapper
from exercises.ex2_dqn import DQN
from exercises.ex2_dqn_config import DQN_PARAMETERS as cfg

def evaluate(agent,env,seeds):
    rows=[]
    for seed in seeds:
        state,_=env.env.reset(seed=seed)
        total=0
        for _ in range(500):
            state,reward,done,_=env.step(agent.predict_action(state))
            total+=reward
            if done: break
        rows.append(total)
    return rows

if __name__ == "__main__":
    env = CartPoleWrapper(seed=42)
    try:
        agent = DQN(env.state_dim,cfg["hidden_dim"],env.action_dim,.00015,cfg["gamma"],0.,250,torch.device("cpu"))
        agent.load(here / "dqn.pth")
        scores = evaluate(agent,env,range(107100,107200))
        print(f"CartPole: {sum(r >= 475 for r in scores)}/100")
    finally:
        env.close()
