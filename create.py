from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
from typing import List
import uvicorn
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

app = FastAPI(title="Dummy Data & House Price Prediction API", description="FastAPI app with dummy items and sklearn house price model")

# In-memory dummy data
items_db: List[dict] = [
    {"id": 1, "name": "Item 1", "price": 10.5, "description": "Dummy item 1"},
    {"id": 2, "name": "Item 2", "price": 20.0, "description": "Dummy item 2"},
    {"id": 3, "name": "Item 3", "price": 15.75, "description": "Dummy item 3"},
    {"id": 4, "name": "Item 4", "price": 25.99, "description": "Dummy item 4"},
    {"id": 5, "name": "Item 5", "price": 5.0, "description": "Dummy item 5"},
    {"id": 6, "name": "Item 6", "price": 30.25, "description": "Dummy item 6"},
    {"id": 7, "name": "Item 7", "price": 12.5, "description": "Dummy item 7"},
    {"id": 8, "name": "Item 8", "price": 18.0, "description": "Dummy item 8"},
    {"id": 9, "name": "Item 9", "price": 22.99, "description": "Dummy item 9"},
    {"id": 10, "name": "Item 10", "price": 8.75, "description": "Dummy item 10"},
]

class ItemCreate(BaseModel):
    name: str
    price: float
    description: str

class Item(BaseModel):
    id: int
    name: str
    price: float
    description: str

# House Price Prediction Model
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, model.predict(X_test))

class HouseFeatures(BaseModel):
    medinc: float  # median income
    houseage: float
    averooms: float
    avebedrms: float
    population: float
    aveoccup: float
    latitude: float
    longitude: float

@app.get("/", tags=["root"])
async def root():
    return {"message": "Dummy Data API"}

@app.get("/items/", response_model=List[Item], tags=["items"])
async def get_items():
    return items_db

@app.get("/items/{item_id}", response_model=Item, tags=["items"])
async def get_item(item_id: int = Path(..., description="ID of the item")):
    for item in items_db:
        if item["id"] == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")

@app.post("/items/", response_model=Item, tags=["items"])
async def create_item(item: ItemCreate):
    new_id = max([i["id"] for i in items_db]) + 1 if items_db else 1
    new_item = {"id": new_id, **item.dict()}
    items_db.append(new_item)
    return new_item

@app.get("/model_info", tags=["house_price"])
async def model_info():
    return {
        "dataset": "California Housing",
        "features": housing.feature_names.tolist(),
        "train_r2": train_r2,
        "test_r2": test_r2
    }

@app.post("/predict_house_price", tags=["house_price"])
async def predict_house_price(features: HouseFeatures):
    feature_array = np.array([[features.medinc, features.houseage, features.averooms, features.avebedrms,
                               features.population, features.aveoccup, features.latitude, features.longitude]])
    prediction = model.predict(feature_array)[0]
    return {"predicted_price_usd": prediction * 100000}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)