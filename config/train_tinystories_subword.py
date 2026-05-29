# config/train_tinystories_subword.py

out_dir = 'out-tinystories-subword'
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_project = 'tinystories_subword'
wandb_run_name = 'clifford-gpt-subword'

dataset = 'tinystories' # Points to data/tinystories/
gradient_accumulation_steps = 2
batch_size = 16 # Balanced size for 3.6M parameter GPU training
block_size = 256 # Context length (in subword tokens)

# Optimized Clifford Baby-GPT model settings
n_layer = 8
n_head = 8
n_embd = 384
dropout = 0.1

# Hyperparameters for stable subword training
learning_rate = 5e-4 # Subword learning rates are typically smaller/gentler
max_iters = 20000 # 20,000 steps is a great target for subwords
lr_decay_iters = 20000
min_lr = 5e-5
beta2 = 0.99
warmup_iters = 200
