"""IFC EE (energy efficiency, Lee 2025 eq.32) D=3, 3BA pool (Random, HillClimbing, CMAES).
Mirrors scripts/configs/dt/ifc.py (D=3 SE 3BA) with reward=EE so SE-vs-EE is apples-to-apples."""
from UtilsRL.misc.namespace import NameSpace

problem = "ifc"
seed = 0
debug = False
name = "dt"

id = "IFCEE"

x_type = "stochastic"
y_loss_coeff = 0.0
mix_method = "concat"
normalize_method = "random"
augment = False
prioritize = False
prioritize_alpha = 1.0
n_block = 100
filter_data = False

embed_dim = 256
num_layers = 12
num_heads = 8
attention_dropout = 0.1
residual_dropout = 0.1
embed_dropout = 0.1
pos_encoding = "embed"
clip_grad = None
use_abs_timestep = True
input_seq_len = 300
max_input_seq_len = 300
scale_clip_range = None

batch_size = 64
num_workers = 1

num_epoch = 3000
step_per_epoch = 100
eval_interval = 500
log_interval = 1
save_interval = 500
eval_episodes = 5
deterministic_eval = False

init_regrets = [0, 20, 50]
regret_strategy = "none"

root_dir = None
data_dir = './data/generated_data/ifc_ee'
cache_dir = './cache/ifc_ee/'

class optimizer_args(NameSpace):
    lr = 2e-4
    weight_decay = 1e-2
    betas = [0.9, 0.999]
    warmup_steps = 10_000

class wandb(NameSpace):
    entity = "lamda-rl"
    project = "IBO"

from scripts.configs.dataset_specs_ifc import (
    train_datasets,
    test_datasets,
)
