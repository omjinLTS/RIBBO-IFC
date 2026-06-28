"""IFC EE D=10, 3BA, HORIZON 1000 retraining (fair extended-budget comparison).
Identical to ifc_d10_ee.py except it reads the 1000-step trajectories and keeps the
full horizon (max_input_seq_len=1000), so the model learns timestep embeddings 0..999.
Run with: --input_seq_len 50 --max_input_seq_len 1000 (CLI overrides below anyway)."""
from UtilsRL.misc.namespace import NameSpace

problem = "ifc"
seed = 0
debug = False
name = "dt"

id = "IFCEED10"

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
input_seq_len = 50
max_input_seq_len = 1000
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
data_dir = './data/generated_data/ifc_d10_ee_h1000'
cache_dir = './cache/ifc_d10_ee_h1000/'

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
