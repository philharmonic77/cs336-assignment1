cfg = {
    "data": {
        "train_token_npy_path": "artifacts/tokenized/tinystories_train.npy",
        "valid_token_npy_path": "artifacts/tokenized/tinystories_valid.npy",
        "batch_size": 128,
        "context_length": 256
    },
    "training": {
        "seed": 1234,
        "resume_from": None,
        "max_iters": 10000,
        "device": "cpu",
        "dtype": "float32",
        "grad_clip": {
            "max_norm": 1.0
        },
        "schedule": {
            "type": "cosine",
            "T_w": 500,
            "T_c": 10000,
            "alpha_min": 3e-5
        },
        "eval_interval": 100,
        "eval_batches": 128,
    },
    "model": {
        "vocab_size": 10000,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
        "rope": {
            "theta": 10000
        }
    },
    "optim": {
        "lr": 3e-4,
        "betas": (0.9,0.95),
        "eps": 1e-8,
        "weight_decay": 0.1
    },
    "io": {
        "out_dir": "runs/test",
        "run_name": "test",
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