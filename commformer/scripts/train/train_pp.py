#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
sys.path.append("../../")
from commformer.config import get_config
from commformer.envs.ic3net_envs.predator_capture_env import PredatorCaptureEnv
from commformer.envs.ic3net_envs.predator_prey_env import PredatorPreyEnv
from commformer.runner.shared.pp_runner import PPRunner as Runner
from commformer.envs.env_wrappers import SubprocVecEnv, DummyVecEnv

"""Train script for PP."""

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "PP":
                env = PredatorPreyEnv(all_args)
            elif all_args.env_name == "PCP":
                env = PredatorCaptureEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "PP":
                env = PredatorPreyEnv(all_args)
            elif all_args.env_name == "PCP":
                env = PredatorCaptureEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='PP', help="Which scenario to run on")
    parser.add_argument('--num_agents', type=int, default=1,
                    help="Number of agents (used in multiagent)")
    parser.add_argument('--nenemies', type=int, default=1,
                         help="Total number of preys in play")
    parser.add_argument('--dim', type=int, default=5,
                        help="Dimension of box")
    parser.add_argument('--vision', type=int, default=1,
                        help="Vision of predator")
    parser.add_argument('--moving_prey', action="store_true", default=False,
                        help="Whether prey is fixed or moving")
    parser.add_argument('--no_stay', action="store_true", default=False,
                        help="Whether predators have an action to stay in place")
    parser.add_argument('--mode', default='mixed', type=str,
                    help='cooperative|competitive|mixed (default: mixed)')
    parser.add_argument('--enemy_comm', action="store_true", default=False,
                        help="Whether prey can communicate.")
    parser.add_argument('--nfriendly_P', type=int, default=2,
                        help="Total number of friendly perception agents in play")
    parser.add_argument('--nfriendly_A', type=int, default=1,
                        help="Total number of friendly action agents in play")
    parser.add_argument('--tensor_obs', action="store_true", default=False,
                        help="Do you want a tensor observation")
    parser.add_argument('--second_reward_scheme', action="store_true", default=False,
                        help="Do you want a partial reward for capturing and partial for getting to it?")
    parser.add_argument('--A_vision', type=int, default=-1,
                        help="Vision of A agents. If -1, defaults to blind")
    parser.add_argument('--eval_episode_length', type=int, default=20)

    all_args = parser.parse_known_args(args)[0]

    all_args.nfriendly = all_args.nfriendly_P + all_args.nfriendly_A
    all_args.num_agents = all_args.nfriendly
    all_args.npredator = all_args.num_agents


    # Enemy comm
    if hasattr(all_args, 'enemy_comm') and all_args.enemy_comm:
        if hasattr(all_args, 'nenemies'):
            all_args.num_agents += all_args.nenemies
        else:
            raise RuntimeError("parser. needs to pass argument 'nenemy'.")

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if "dec" in all_args.algorithm_name:
        all_args.dec_actor = True
        all_args.share_actor = False


    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        import time
        timestr = time.strftime("%y%m%d-%H%M%S")
        curr_run = all_args.prefix_name + "-" + timestr
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.alg_seed)
    torch.cuda.manual_seed_all(all_args.alg_seed)
    np.random.seed(all_args.alg_seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
