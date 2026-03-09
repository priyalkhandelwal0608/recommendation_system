import torch
import numpy as np
from src.model import TwoTowerModel
from src.data_loader import load_data

def generate_item_embeddings():
    # Load data
    movies, ratings = load_data()
    num_users = ratings.userId.max() + 1
    num_items = movies.movieId.max() + 1

    # Load trained model
    model = TwoTowerModel(num_users, num_items)
    model.load_state_dict(torch.load("models/two_tower_model.pth"))
    model.eval()

    # Generate transformed item embeddings
    item_ids = torch.arange(num_items)
    with torch.no_grad():
        raw_embeddings = model.item_embedding(item_ids)
        embeddings = model.item_fc(raw_embeddings)

    # Save embeddings
    np.save("models/item_embeddings.npy", embeddings.numpy())
    print("Item embeddings saved")

if __name__ == "__main__":
    generate_item_embeddings()