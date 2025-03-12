from run_discre import get_discre_embedding

from luar.utils import get_embedding_for_text
from luar.transformer import Transformer
from luar.model_utils import change_attention_to_FlashAttention
from transformers import AutoTokenizer
from luar.default_params import params

import pandas as pd
import pickle
import torch
import numpy as np
from embedding_models.models import DeepEmbeddingModelAvg

def discre_embed(file_path, output_path):
    _, average_embeddings = get_discre_embedding(file_path)

    with open(output_path, 'wb') as f:
        pickle.dump(average_embeddings, f)

def luar_embed(file_path, output_path):
    # Initialize model and tokenizer
    model = Transformer(params)
    change_attention_to_FlashAttention(model)
    #tokenizer = AutoTokenizer.from_pretrained('/local/nlp/pranjal-sri/dev/aa_hiatus/LUAR/pretrained_weights/paraphrase-distilroberta-base-v1')
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v1')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    df = pd.read_csv(file_path)
    file_ids = df['documentID'].tolist()
    texts = df['message'].tolist()

    # Get embeddings for each text
    embeddings = {}
    for text, file_id in zip(texts, file_ids):
        embedding = get_embedding_for_text(text, tokenizer, model, device)
        embeddings[file_id] = embedding

    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

def embed(luar_embeddings_dict, discre_embeddings_dict):
    """
    Embeds LUAR and DiscRE embeddings using the provided model.
    
    Args:
        luar_embeddings_dict: Dict mapping keys to numpy arrays of shape (512,)
        discre_embeddings_dict: Dict mapping keys to numpy arrays of shape (1, 845)
        model: Neural network model to combine the embeddings
        
    Returns:
        Dictionary mapping the same keys to combined embeddings from the model
    """
    # Verify dictionaries have same keys
    assert set(luar_embeddings_dict.keys()) == set(discre_embeddings_dict.keys()), "Dictionaries must have same keys"
    
    model = DeepEmbeddingModelAvg()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Convert dictionaries to tensors
    keys = list(luar_embeddings_dict.keys())
    luar_tensor = torch.tensor(np.stack([luar_embeddings_dict[k] for k in keys]), dtype=torch.float32)
    discre_tensor = torch.tensor(np.stack([discre_embeddings_dict[k][0] for k in keys]), dtype=torch.float32)
    
    # Move to device
    luar_tensor = luar_tensor.to(device)
    discre_tensor = discre_tensor.to(device)
    
    # Inference without gradient computation
    with torch.no_grad():
        output_tensor = model(luar_tensor, discre_tensor)
    
    # Convert back to dictionary
    output_dict = {
        key: output_tensor[i].cpu().numpy()
        for i, key in enumerate(keys)
    }
    
    return output_dict

def cleanup_files(base_dir = "./"):
    """
    Cleans up any files containing 'message_only' in their filename.
    
    Args:
        base_dir (str): Base directory to search for files. Defaults to current directory.
    """
    import os
    import glob

    # Find all files with 'message_only' in the name
    pattern = os.path.join(base_dir, "*message_only*")
    files_to_remove = glob.glob(pattern)
    
    # Remove each matching file
    for file in files_to_remove:
        try:
            os.remove(file)
        except OSError as e:
            print(f"Error removing {file}: {e}")
    

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python main.py <input_file>")
        sys.exit(1)
        
    try:
        input_file = sys.argv[1]
        luar_embed(input_file, 'luar_embeddings.pkl')
        discre_embed(input_file, 'discre_embeddings.pkl')

        with open('luar_embeddings.pkl', 'rb') as f:
            luar_embeddings = pickle.load(f)

        with open('discre_embeddings.pkl', 'rb') as f:
            discre_embeddings = pickle.load(f)

        embeddings = embed(luar_embeddings, discre_embeddings)

        with open('embeddings.pkl', 'wb') as f:
            pickle.dump(embeddings, f)

    finally:
        cleanup_files()




