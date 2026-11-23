"""Deploy the finite-horizon maximum-reach controller on the same stochastic draws."""
import json
import os
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(os.environ["R"])/"hw4_reinforcement_learning"))
from envs.grid_world import CliffWalkingEnv
results=[]
for slip in (0.,.01,.2):
    env=CliffWalkingEnv(slip_chance=slip)
    p=np.zeros((env.n_states,env.n_actions,env.n_states))
    for s,actions in enumerate(env.P):
        for a,transitions in enumerate(actions):
            for prob,nxt,_,_ in transitions:p[s,a,nxt]+=prob
    v=np.zeros(env.n_states)
    v[env.goal_state]=1
    policies=[]
    for t in range(100):
        q=p@v
        policies.append(q.argmax(1))
        v=q.max(1)
    wins=[]
    for seed in range(107000,107100):
        rng=np.random.default_rng(seed)
        s=env.start_state
        for action_by_state in reversed(policies):
            transitions=env.P[s][action_by_state[s]]
            _,s,_,done=transitions[rng.choice(len(transitions),p=[t[0] for t in transitions])]
            if done:break
        wins.append(s==env.goal_state)
    results.append(dict(slip=slip,upper_bound=v[env.start_state],wins=wins,success=sum(wins)))
with Path("matched-optimal.json").open("x") as out:
    json.dump(results,out,indent=2)
print([(r["slip"],r["success"]) for r in results])
