import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import torch

def chunk_sentences(text, max_chunk_size):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ''

    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= max_chunk_size:
            current_chunk += ' ' + sentence.strip() if current_chunk else sentence.strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence.strip()

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def get_embedding_for_text(text, tokenizer, model, device):
    text_chunks  = chunk_sentences(text, 64)
    tokenized_text = tokenizer(text_chunks, padding = "max_length", max_length = 32, truncation = True, return_tensors='pt')
    data_tensor = [tokenized_text['input_ids'], tokenized_text['attention_mask']]

    data_tensor = [d.reshape(1, -1, 32) for d in data_tensor]
    data_tensor = [d.to(device = device) for d in data_tensor]


    with torch.no_grad():
        episode_embedding = model(input_ids = data_tensor[0], attention_mask = data_tensor[1])
    episode_embedding = episode_embedding.squeeze()
    return episode_embedding.cpu().numpy()



