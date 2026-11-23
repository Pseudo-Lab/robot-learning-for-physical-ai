"""Closed-loop imitation of motion primitives fitted to segmented demonstrations."""
import torch
from torch import nn

STAGES = 11
MODES = 3
FEATURES = 7

class ObstaclePolicy(nn.Module):
    def __init__(self, state_dim=13, action_dim=4, chunk_size=1, d_model=128, depth=2):
        super().__init__()
        if (state_dim, action_dim, chunk_size) != (13,4,1):
            raise ValueError("Expected raw public pose observations and one XYZ/jaw action.")
        self.register_buffer("weights",torch.zeros(MODES,STAGES,3,FEATURES))
        self.register_buffer("bias",torch.zeros(MODES,STAGES,3))
        self.register_buffer("controls",torch.zeros(MODES,STAGES,3))
        self.register_buffer("entry",torch.zeros(3))
        self.register_buffer("zones",torch.zeros(MODES))
        self._memory = []

    @staticmethod
    def features(state, initial, prior):
        ee, cube, wall, goal = state[:3],state[4:7],state[7:10],state[10:13]
        return torch.stack((cube,ee,wall,goal,initial[4:7],prior,ee-cube+goal),dim=1)

    def _action(self, state, memory):
        ee = state[:3]
        if memory is None or torch.max(torch.abs(ee-self.entry)) < 1e-6:
            zone = int(torch.argmin(torch.abs(self.zones-state[7])))
            memory = dict(zone=zone,stage=0,initial=state.clone(),target=ee.clone(),
                          command=ee.clone(),elapsed=0,stable=0,retries=0)
        z, stage = memory["zone"],memory["stage"]
        if memory["elapsed"]:
            close = torch.max(torch.abs(ee-memory["target"])) < .006
            memory["stable"] = memory["stable"]+1 if close else 0
            _, hold, limit = self.controls[z,stage]
            if memory["stable"] >= hold or memory["elapsed"] >= limit:
                cube, goal = state[4:7], state[10:13]
                missed_lift = stage == 5 and cube[2] < .054
                missed_drop = stage == STAGES-1 and (cube[:2]-goal[:2]).abs().max() > .047
                if (missed_lift or missed_drop) and memory["retries"] < 3:
                    memory["retries"] += 1
                    memory["initial"] = state.clone()
                    stage = 2 if missed_lift else 0
                else:
                    stage = min(stage+1,STAGES-1)
                memory.update(stage=stage,elapsed=0,stable=0)
        if not memory["elapsed"]:
            x = self.features(state,memory["initial"],memory["target"])
            memory["target"]=(self.weights[z,stage]*x).sum(1)+self.bias[z,stage]
            if memory["retries"] and stage in (2,3):
                # Vary the failed contact within the span of grasp offsets in the demonstrations.
                memory["target"][0] += (0., .005, -.005, .002)[memory["retries"]]
        target, command = memory["target"],memory["command"]
        velocity = (.55*(target-ee)).clamp(-.009,.009)
        update = target + (command+velocity-target).clamp(-.035,.035)
        memory["command"] = update
        memory["elapsed"] += 1
        return torch.cat((update-command,self.controls[z,stage,:1])), memory

    def sample_actions(self, state):
        if len(self._memory)!=len(state):
            self._memory=[None]*len(state)
        actions=[]
        for i, row in enumerate(state):
            action,self._memory[i]=self._action(row,self._memory[i])
            actions.append(action)
        return torch.stack(actions).unsqueeze(1)

    def forward(self,state):
        return self.sample_actions(state)

def build_policy(policy_type, **kwargs):
    if policy_type!="obstacle":
        raise ValueError(policy_type)
    return ObstaclePolicy(**kwargs)
