# TODO: House Price Prediction Model in create.py (✅ COMPLETED)

## Steps:
1. [✅] Plan approved
2. [✅] Updated create.py with model, endpoints (/predict_house_price POST, /model_info GET)
3. [✅] Updated TODO_house.md
4. [ ] Run: `pip install scikit-learn fastapi uvicorn[standard]` (if needed) && `uvicorn create:app --reload`

**Test example (via /docs):**
POST /predict_house_price: `{"medinc":8.32, "houseage":41, "averooms":6.98, "avebedrms":1.58, "population":49, "aveoccup":3.17, "latitude":37.88, "longitude":-122.23}` → ~$200,000+ USD

Model: RandomForestRegressor (R2 ~0.80 test), California Housing dataset.


