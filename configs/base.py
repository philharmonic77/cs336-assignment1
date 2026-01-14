cfg = {
    "data": {
        "train_token_npy_path": "artifacts/tokenized/tinystories_train.npy",
        "valid_token_npy_path": "artifacts/tokenized/tinystories_valid.npy",
        "batch_size": 4,
        "context_length": 128
    },
    "training": {
        "seed": 1234,
        "resume_from": None,
        "max_iters": 400,
        "device": "cpu",
        "dtype": "float32",
        "grad_clip": {
            "max_norm": 10000
        },
        "schedule": {
            "type": "cosine",
            "T_w": 0,
            "T_c": 1000,
            "alpha_min": 0.0
        },
        "eval_interval": 50,
        "eval_batches": 10,
    },
    "model": {
        "vocab_size": 32000,
        "d_model": 16,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 64,
        "rope": {
            "theta": 100000
        }
    },
    "optim": {
        "lr": 1e-3,
        "betas": (0.9,0.999),
        "eps": 1e-8,
        "weight_decay": 0.0
    },
    "io": {
        "out_dir": "runs/exp_lr1e-3_bs64",
        "run_name": "lr1e-3_bs64",
        "log_interval": 10,
        "save_interval": 200,
        "wandb": {
            "enabled": True,
            "project": "cs336-assignment1",
            "mode": "offline",
            "tags": ["tinystories", "baseline"],
        },
    }
}