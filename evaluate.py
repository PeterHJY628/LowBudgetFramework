import math
import time
import os
import random

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import experiment_util as util
import argparse
from pprint import pprint
from tqdm import tqdm
import core
import yaml
from core.helper_functions import *

try:
    from huggingface_hub import login
    login(token=os.environ.get("HF_TOKEN"))
except ImportError:
    pass

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument("--config", type=str, default=None,
                    help="Path to YAML config file. Defaults to configs/{dataset}.yaml")
parser.add_argument("--run_id", type=int, default=1)
parser.add_argument("--agent_seed", type=int, default=1)
parser.add_argument("--pool_seed", type=int, default=1)
parser.add_argument("--model_seed", type=int, default=1)
parser.add_argument("--agent", type=str, default="galaxy")
parser.add_argument("--dataset", type=str, default="topv2")
parser.add_argument("--query_size", type=int, default=50)
parser.add_argument("--encoded", type=int, default=0)
parser.add_argument("--restarts", type=int, default=50)
parser.add_argument("--fitting_mode", type=str, default=None,
                    choices=["from_scratch", "finetuning", "shrinking"],
                    help="Override classifier_fitting_mode from config. "
                         "'from_scratch' retrains from initial weights each AL round; "
                         "'finetuning' continues from the previous round's weights.")
##########################################################
parser.add_argument("--save_checkpoints", type=int, default=0,
                    help="Save model weights + labeled indices after each AL round")
parser.add_argument("--experiment_postfix", type=str, default=None)
parser.add_argument("--strict_deterministic", type=int, default=1)
args = parser.parse_args()
args.encoded = bool(args.encoded)
args.save_checkpoints = bool(args.save_checkpoints)
args.strict_deterministic = bool(args.strict_deterministic)

config_path = args.config if args.config else f"configs/{args.dataset.lower()}.yaml"


def configure_determinism(seed: int, strict: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if strict:
        # Required for deterministic cublas behavior
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)

run_id = args.run_id
max_run_id = run_id + args.restarts
while run_id < max_run_id:
    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.Loader)
    config["current_run_info"] = args.__dict__

    if args.fitting_mode is not None:
        config_key = "dataset_embedded" if args.encoded else "dataset"
        config[config_key]["classifier_fitting_mode"] = args.fitting_mode
        print(f"Overriding classifier_fitting_mode -> {args.fitting_mode}")

    print("Config:")
    pprint(config)
    print("Config End \n")

    pool_rng = np.random.default_rng(args.pool_seed + run_id)
    model_seed = args.model_seed + run_id
    configure_determinism(model_seed, args.strict_deterministic)
    data_loader_seed = 1

    AgentClass = get_agent_by_name(args.agent)
    DatasetClass = get_dataset_by_name(args.dataset)

    # Inject additional configuration into the dataset config (See BALD agent)
    AgentClass.inject_config(config)
    DatasetClass.inject_config(config)

    dataset = DatasetClass(args.data_folder, config, pool_rng, args.encoded)
    dataset = dataset.to(util.device)
    env = core.ALGame(dataset,
                      pool_rng,
                      model_seed=model_seed,
                      data_loader_seed=data_loader_seed,
                      device=util.device)
    agent = AgentClass(args.agent_seed, config, args.query_size)

    init_dir = f"init{dataset.initial_points_per_class}"
    if args.experiment_postfix is not None:
        base_path = os.path.join("runs", dataset.name, init_dir, str(args.query_size), f"{agent.name}_{args.experiment_postfix}")
    else:
        base_path = os.path.join("runs", dataset.name, init_dir, str(args.query_size), agent.name)
    log_path = os.path.join(base_path, f"run_{run_id}")

    print(f"Starting run {run_id}")
    time.sleep(0.1) # prevents printing uglyness with tqdm

    with core.EnvironmentLogger(env, log_path, util.is_cluster,
                                    save_checkpoints=args.save_checkpoints) as env:
        done = False
        dataset.reset()
        state = env.reset()
        iterations = math.ceil(env.env.budget / args.query_size)
        iterator = tqdm(range(iterations), miniters=2)
        for i in iterator:
            action = agent.predict(*state)
            state, reward, done, truncated, info = env.step(action)
            iterator.set_postfix({"accuracy": env.accuracies[1][-1]})
            if done or truncated:
                # triggered when sampling batch_size is >1
                break

    # collect results from all runs
    collect_results(base_path, "run_")
    save_meta_data(log_path, agent, env, dataset, config)
    run_id += 1
