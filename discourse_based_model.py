from glob import glob

import os
from datasets import DatasetDict, Dataset
from absl import logging
import pandas as pd
import torch
import numpy as np

from embedding_models.models import DeepEmbeddingModelAvg
from run_discre import get_discre_embedding

from luar.utils import get_embedding_for_text
from luar.transformer import Transformer
from luar.model_utils import change_attention_to_FlashAttention
from transformers import AutoTokenizer
from luar.default_params import params

from embedding_models.models import DeepEmbeddingModelAvg
from iarpa_utils import *

class SIV_Discourse_Based_Model():

    def __init__(self, input_dir, language='en', sideload_input_dir=''):
        self.input_dir = input_dir
        self.query_identifier = "authorIDs"
        self.candidate_identifier = "authorSetIDs"
        self.batch_size = 16
        self.author_level = True
        self.text_key = "fullText"
        self.token_max_length = 512
        self.document_batch_size = 32
        self.language = language
        self.sideload_input_dir = sideload_input_dir

        self.load_model(params)
        
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_author_level(self, author_level):
        self.author_level = author_level
    
    def set_token_max_length(self, token_max_length):
        self.token_max_length = token_max_length
    
    def set_text_key(self, text_key):
        self.text_key = text_key

    def get_features(self):
        logging.info("Loading TA1 features")
        self.dataset_path = get_dataset_path(self.input_dir)
        dataset = DatasetDict.load_from_disk(self.dataset_path)

        self.query_features = dataset["queries"]["features"]
        self.query_labels = dataset["queries"][self.query_identifier]
        self.candidate_features = dataset["candidates"]["features"]
        self.candidate_labels = dataset["candidates"][self.candidate_identifier]
        return self.query_features, self.candidate_features, self.query_labels, self.candidate_labels

    def load_luar(self, params):
        # Initialize model and tokenizer
        model = Transformer(params)
        change_attention_to_FlashAttention(model)
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v1')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.luar_model = model.to(self.device)
        self.luar_model.eval()

        
        
    def load_model(self, params):
        logging.info("Loading LUAR")

        self.load_luar(params)

        logging.info("Loading The Full Model")
        self.model = DeepEmbeddingModelAvg()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()
        
    def extract_embeddings(self, data_fname):
        data = pd.read_json(data_fname, lines=True)
        batch_size = self.batch_size

        if self.author_level:
            identifier = "authorIDs" if "queries" in data_fname else "authorSetIDs"
            data[identifier] = data[identifier].apply(lambda x: str(tuple(x)))
            # data = data[[identifier, self.text_key]].groupby(identifier).fullText.apply(list).reset_index()

            batch_size = 1
            logging.info("Setting batch size to 1 for author level embeddings with LUAR.")

        else:
            identifier = "documentID"

        # Get LUAR Embeddings
        luar_embeddings_ds = self.get_luar_embeddings(data, identifier, batch_size)
        keys = luar_embeddings_ds[identifier]
        luar_embeddings_avg = [np.mean(l) for l in luar_embeddings_ds["features"]]
        luar_tensor = torch.tensor(np.stack(luar_embeddings_avg), dtype=torch.float32)
        
        # Get Discre Embeddings
        grouped_by_author_df = data.groupby(identifier).agg({self.text_key: lambda x: "\n".join(list(x))}).reset_index()
        data["message"]    = grouped_by_author_df[self.text_key]
        data["documentID"] = grouped_by_author_df[identifier]
        #print(len(data))
        #print(data["message"].tolist()[:10])
        data.to_csv("./text_to_embed.csv")
        _, discre_embeddings_dict = get_discre_embedding("./text_to_embed.csv")
        discre_tensor = torch.tensor(np.stack([discre_embeddings_dict[k][0] for k in keys]), dtype=torch.float32)


        # Move to device
        luar_tensor = luar_tensor.to(self.device)
        discre_tensor = discre_tensor.to(self.device)

        # Inference without gradient computation
        with torch.no_grad():
            output_tensor = model(luar_tensor, discre_tensor)
        
        # Convert back to dictionary
        output_dict = {
            key: output_tensor[i].cpu().numpy()
            for i, key in enumerate(keys)
        }
        
        return output_dict

    def generate_sivs(self):
        queries_fname, candidates_fname = get_file_paths(self.input_dir)
        logging.info("Extracting Query Embeddings")
        queries = self.extract_embeddings(queries_fname)
        logging.info("Extracting Candidate Embeddings")
        candidates = self.extract_embeddings(candidates_fname)
        return queries, candidates

    def store_sivs(self, queries_fname, queries, candidates, output_dir, run_id):
        logging.info("Saving Dataset and Cosine as Distance Metric")
        save_files(queries_fname, queries, candidates, output_dir, run_id)

    def get_luar_embeddings(self, data, identifier, batch_size):
        all_identifiers, all_outputs = [], []

        for i in range(0, len(data), batch_size):
            if i % 10 == 0:
                print(f"Progress: {int((i/len(data))*100)}%")
            chunk = data.iloc[i:i+batch_size]
            raw_text = list(chunk[self.text_key])
            output = [get_embedding_for_text(text, self.tokenizer, self.luar_model, self.device) for text in raw_text]
            all_identifiers.extend(chunk[identifier])
            all_outputs.extend(output)
        
        # Helper to get organize document embeddings by author identifier
        author_to_features = {}
        for auth, output in zip(all_identifiers, all_outputs):
            if auth not in author_to_features:
                author_to_features[auth] = []
            author_to_features[auth].append(output)
        process = lambda x: torch.tensor(x)
        identifiers = sorted(list(author_to_features.keys()))

        dataset = Dataset.from_dict({
            identifier: identifiers,
            "features": [process(author_to_features[auth]) for auth in identifiers],
        })

        return dataset

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


def get_dataset_path(input_path):
    dataset_path = glob(os.path.join(input_path, "*TA2_queries*"))[0]
    return dataset_path

if __name__ == "__main__":
    siv = SIV_Discourse_Based_Model("/mnt/swordfish-pool2/milad/hiatus-data/V2/english_TA2_p1_and_p2_dev_20240207/")
    siv.generate_sivs()