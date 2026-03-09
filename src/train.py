import torch
import torch.optim as optim
from src.model import TwoTowerModel
from src.data_loader import load_data

def train():
    movies, ratings = load_data()
    num_users = ratings.userId.max() + 1
    num_items = movies.movieId.max() + 1

    model = TwoTowerModel(num_users, num_items)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CosineEmbeddingLoss()

    user_ids = torch.tensor(ratings.userId.values)
    item_ids = torch.tensor(ratings.movieId.values)
    labels = torch.ones(len(user_ids))

    for epoch in range(50):
        optimizer.zero_grad()
        user_vec, item_vec = model(user_ids, item_ids)
        loss = loss_fn(user_vec, item_vec, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "models/two_tower_model.pth")
    print("Model saved")

if __name__ == "__main__":
    train()