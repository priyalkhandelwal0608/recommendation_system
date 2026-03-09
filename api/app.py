"""
Movie Recommendation API
------------------------
FastAPI + PyTorch + FAISS demo with styled HTML interface.
"""

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
import torch
import faiss
import logging
from src.model import TwoTowerModel
from src.data_loader import load_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recommendation_api")

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="A demo recommender system built with FastAPI, PyTorch, and FAISS.",
    version="1.0.0"
)

# Load data
movies, ratings = load_data()
num_users = ratings.userId.max() + 1
num_items = movies.movieId.max() + 1

# Load trained model
model = TwoTowerModel(num_users, num_items)
model.load_state_dict(torch.load("models/two_tower_model.pth"))
model.eval()
logger.info("Model loaded successfully.")

# Load FAISS index
index = faiss.read_index("models/faiss_index.index")
logger.info("FAISS index loaded successfully.")

@app.get("/", response_class=HTMLResponse)
def root():
    """
    Root endpoint: colorful homepage with search box.
    """
    html_content = """
    <html>
        <head>
            <title>Movie Recommendation API</title>
            <style>
                body { font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #74ebd5 0%, #ACB6E5 100%); margin: 0; padding: 40px; color: #333; }
                h1 { color: #222; }
                .container { background: #fff; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); max-width: 600px; margin: auto; }
                input[type=number] { padding: 10px; border-radius: 6px; border: 1px solid #ccc; width: 100px; }
                button { padding: 10px 20px; border: none; border-radius: 6px; background: #4CAF50; color: white; cursor: pointer; }
                button:hover { background: #45a049; }
                a { color: #007BFF; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎬 Movie Recommendation API</h1>
                <p>Enter a <b>User ID</b> to get personalized movie recommendations.</p>
                <form action="/recommend" method="get">
                    <input type="number" name="user_id" placeholder="User ID" required>
                    <button type="submit">Get Recommendations</button>
                </form>
                <p style="margin-top:20px;">Or try: <a href="/recommend?user_id=10">Recommendations for User 10</a></p>
                <p>Explore API docs: <a href="/docs">Swagger UI</a></p>
            </div>
        </body>
    </html>
    """
    return html_content

@app.get("/recommend", response_class=HTMLResponse)
def recommend(user_id: int = Query(..., description="User ID to get recommendations for")):
    """
    Recommendation endpoint: returns top-N movie recommendations for a given user.
    """
    try:
        # Generate user embedding (transformed)
        user_tensor = torch.tensor([user_id])
        with torch.no_grad():
            user_embedding = model.user_fc(model.user_embedding(user_tensor))
        user_embedding = user_embedding.numpy()

        # Search FAISS index
        distances, indices = index.search(user_embedding, 5)

        # Map FAISS indices back to movieId column
        recommended_movies = []
        for idx in indices[0]:
            movie_row = movies[movies["movieId"] == idx]
            if not movie_row.empty:
                recommended_movies.append(movie_row.iloc[0]["title"])

        if not recommended_movies:
            return HTMLResponse(content=f"<h3>No recommendations found for User {user_id}</h3>", status_code=404)

        # Styled HTML output
        html_content = f"""
        <html>
            <head>
                <title>Recommendations for User {user_id}</title>
                <style>
                    body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5; margin: 0; padding: 40px; }}
                    .container {{ background: #fff; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); max-width: 600px; margin: auto; }}
                    h2 {{ color: #333; }}
                    ul {{ list-style-type: none; padding: 0; }}
                    li {{ background: #e3f2fd; margin: 8px 0; padding: 10px; border-radius: 6px; }}
                    a {{ color: #007BFF; text-decoration: none; }}
                    a:hover {{ text-decoration: underline; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>Top Recommendations for User {user_id}</h2>
                    <ul>
                        {''.join(f'<li>{movie}</li>' for movie in recommended_movies)}
                    </ul>
                    <p><a href="/">Back to Home</a></p>
                </div>
            </body>
        </html>
        """
        return html_content
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)