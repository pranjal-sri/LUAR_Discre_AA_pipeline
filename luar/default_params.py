from types import SimpleNamespace
transformer_path = "./pretrained_weights"

params = {
    ###### MISC parameters ##########
    "dataset_name": "raw_all",  # Specifies which dataset to use
    "experiment_id": "1727742308",  # Experiment identifier based on timestamp
    "version": None,  # PyTorch Lightning folder version name
    "log_dirname": "lightning_logs",  # Directory for logging
    "model_type": "roberta",  # Transformer backbone to use
    "text_key": "syms",  # Key where the text is located in the dataset
    "time_key": "hour",  # Key for time information in the dataset
    "do_learn": False,  # Whether to train on the training set
    "validate": False,  # Whether to validate on the dev set
    "evaluate": False,  # Whether to evaluate on the test set
    "validate_every": 5,  # Validate every N epochs
    "sanity": None,  # Subsamples N authors for debugging
    "random_seed": 777,  # Seed for random number generators
    "gpus": 1,  # Number of GPUs to use for training
    "period": 5,  # Period to save checkpoints when not validating
    "suffix": "",  # Suffix for data files
    "transformer_path": transformer_path,  # Path to the transformer model

    ##### Training parameters ##########
    "learning_rate": 2e-5,  # Specifies learning rate
    "learning_rate_scaling": False,  # Variance-based learning rate scaling toggle
    "batch_size": 128,  # Number of authors per batch
    "load_checkpoint": False,  # If True, load the latest checkpoint
    "precision": 16,  # Model precision
    "num_workers": 10,  # Number of workers for data loading
    "num_epoch": 20,  # Number of epochs for training
    "pin_memory": False,  # Pin memory for prefetching data
    "gradient_checkpointing": False,  # Activates gradient checkpointing
    "temperature": 0.01,  # Temperature for contrastive loss (SupCon)
    "multidomain_prob": None,  # Sampling probability for the Multi-Domain dataset
    "mask_bpe_percentage": 0.0,  # Percentage of BPE masking during training

    ##### Model Hyperparameters #####
    "episode_length": 16,  # Number of actions in an episode
    "token_max_length": 32,  # Maximum number of tokens per example
    "num_sample_per_author": 2,  # Number of samples per author during training
    "embedding_dim": 512,  # Output embedding dimension
    "attention_fn_name": "memory_efficient",  # Attention mechanism type
    "use_random_windows": False  # Use random windows for training
}

params = SimpleNamespace(**params)
