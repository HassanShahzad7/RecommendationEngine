from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import pandas as pd
from main import recommend_products, preprocessing_data, split_data, evaluate_recommendation

app = FastAPI()


@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: int):
    try:
        data = pd.read_csv('data.csv', encoding='ISO-8859-1')
        # Call recommendation engine function to get recommendations for the user
        data = preprocessing_data(data)
        recommendations = recommend_products(data, user_id)
        if recommendations:
            return {"user_id": user_id, "recommendations": recommendations}
        else:
            raise HTTPException(status_code=404, detail="User not found or no recommendations available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluation")
async def evaluate_recommendation_engine():
    try:
        # Implement evaluation mechanism
        # Track performance metrics, analyze response times, etc.
        # Return evaluation results
        data = pd.read_csv('data.csv', encoding='ISO-8859-1')
        data = preprocessing_data(data)
        train_df, test_df = split_data(data)
        evaluation_results = evaluate_recommendation(train_df, test_df, recommend_products, top_n=10)

        # Assuming evaluation_results is a list containing precision and recall
        precision, recall = evaluation_results

        # Create a dictionary to represent the results in JSON format
        results_json = {
            "precision": precision,
            "recall": recall
        }

        # Return the results as JSONResponse
        return JSONResponse(content=results_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Return the results as JSONResponse
    return JSONResponse(content=results_json)
