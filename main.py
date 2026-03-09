import argparse
from src.train import train
from src.embeddings import generate_item_embeddings
from src.vector_store import build_index

def main():
    parser = argparse.ArgumentParser(description="Run the recommendation system pipeline")
    parser.add_argument("step", choices=["train", "embeddings", "index"], 
                        help="Which step to run: train, embeddings, or index")
    args = parser.parse_args()

    if args.step == "train":
        train()
    elif args.step == "embeddings":
        generate_item_embeddings()
    elif args.step == "index":
        build_index()

if __name__ == "__main__":
    main()