import torch

def embed(luar_embeddings, discre_embeddings, model):
    """
    Embeds LUAR and DiscRE embeddings using the provided model.
    If CUDA is available, computation will be done on GPU.
    
    Args:
        luar_embeddings: Tensor of shape (..., 512) containing LUAR embeddings
        discre_embeddings: Tensor of shape (..., 845) containing DiscRE embeddings
        model: Neural network model to combine the embeddings
        
    Returns:
        Combined embeddings from the model (on CPU)
    """

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Move inputs to appropriate device
    luar_embeddings = luar_embeddings.to(device)
    discre_embeddings = discre_embeddings.to(device)
    
    # Inference without gradient computation
    with torch.no_grad():
        output = model(luar_embeddings, discre_embeddings)
    
    return output.cpu()  # Return result back on CPU