"""Exercise HW2 algorithms on separately seeded trajectories and targets."""
import argparse
import json
import os
import sys
from pathlib import Path
import mujoco
import numpy as np
r=Path(os.environ["R"])
sys.path.insert(0,str(r)); hw=r/"hw2_robot_control_mdps"
sys.path.insert(0,str(hw))
from hw2_robot_control_mdps.exercises.ex1 import build_keypoints,ik_track
from hw2_robot_control_mdps.exercises.ex2 import generate_quintic_spline_waypoints,pid_control
from env.so100_tracking_env import SO100TrackEnv
assets=hw/"so101_gym/assets"

def path_metrics(keypoints, tips):
    """Measure the tip travel and its largest 3-D distance to the closed path."""
    tips = np.asarray(tips)
    segments = np.roll(keypoints, -1, axis=0) - keypoints
    offsets = tips[:, None, :] - keypoints
    fractions = np.sum(offsets * segments, axis=2) / np.sum(segments**2, axis=1)
    residuals = offsets - np.clip(fractions, 0, 1)[:, :, None] * segments
    nearest_distance = np.linalg.norm(residuals, axis=2).min(axis=1)
    return {
        "tip_travel_m": float(np.linalg.norm(np.diff(tips, axis=0), axis=1).sum()),
        "max_path_deviation_m": float(nearest_distance.max()),
    }


def control(which, max_width, pid_gains=()):
    xml="so100_torque_ctrl.xml" if which==2 else "so100_pos_ctrl.xml"
    model=mujoco.MjModel.from_xml_path(str(assets/xml))
    rows=[]
    for seed in range(107200,107300):
        width=np.random.default_rng(seed).uniform(.14,max_width)
        points=build_keypoints(width=width)
        plan=mujoco.MjData(model)
        plan.qpos[:]=model.jnt_range.mean(axis=1)
        plan.qpos[-1]=0
        route=np.vstack([generate_quintic_spline_waypoints(a,b,5) for a,b in zip(points,np.roll(points,-1,axis=0))]) if which==2 else points
        commands=[]; errors=[];limits=True
        for target in route:
            q=ik_track(model,plan,"ee_site",target)
            plan.qpos[:]=q;mujoco.mj_forward(model,plan)
            limits=bool(limits and np.all(q>=model.jnt_range[:,0]-1e-9) and np.all(q<=model.jnt_range[:,1]+1e-9))
            errors.append(float(np.linalg.norm(plan.site("ee_site").xpos-target)))
            commands.append(q)
        row=dict(seed=seed,width=width,targets=len(route),max_ik_error_m=max(errors),joint_limits=limits)
        if which==1:
            row["success"]=limits and max(errors)<.005
        else:
            sim=mujoco.MjData(model);sim.qpos[:]=commands[0];mujoco.mj_forward(model,sim)
            reached=0;steps=0
            tips = []
            for target,q in zip(route,commands):
                history=[]
                for _ in range(3000):
                    sim.mocap_pos[0]=target
                    history.append(q-sim.qpos)
                    if len(history)>10: history.pop(0)
                    sim.ctrl[:]=pid_control(np.asarray(history), model.opt.timestep, *pid_gains)
                    mujoco.mj_step(model,sim);steps+=1
                    tip = sim.site("ee_site").xpos.copy()
                    tips.append(tip)
                    if np.linalg.norm(tip-target)<.03:
                        reached+=1;break
                else: break
            row.update(reached=reached,physics_steps=steps,success=reached==len(route) and limits and all(w.number==0 for w in sim.warning))
            row.update(path_metrics(points, tips))
        rows.append(row)
    return rows

def ppo():
    from stable_baselines3 import PPO
    import torch
    env=SO100TrackEnv(assets/"so100_pos_ctrl.xml",render_mode=None)
    policy=PPO("MlpPolicy",env,device="cpu")
    policy.policy.load_state_dict(torch.load(Path(__file__).with_name("ppo-policy.pt"),map_location="cpu",weights_only=True))
    rows=[]
    try:
        for seed in range(107200,107300):
            obs,_=env.reset(seed=seed)
            for _ in range(int(2/env.ctrl_timestep)):
                act,_=policy.predict(obs,deterministic=True)
                obs,_,term,trunc,info=env.step(act)
                if term or trunc:break
            err=float(info["ee_tracking_error"])
            rows.append(dict(seed=seed,final_error_m=err,success=err<.05))
    finally:env.close()
    return rows

p=argparse.ArgumentParser();p.add_argument("exercise",type=int,choices=(1,2,3));p.add_argument('--max-width', type=float, default=.20)
p.add_argument("--pid-gains", nargs=3, type=float, metavar=("KP", "KI", "KD"),
               help="override the default gains for the Ex2 comparison")
a=p.parse_args()
if a.pid_gains is not None and a.exercise != 2:
    p.error("--pid-gains applies to exercise 2")
rows=ppo() if a.exercise==3 else control(a.exercise, a.max_width, a.pid_gains or ())
with open(f"ex{a.exercise}.json","x") as f:json.dump(dict(success=sum(x["success"] for x in rows),total=len(rows),rows=rows),f,indent=2)
print(a.exercise,sum(x["success"] for x in rows),len(rows))
