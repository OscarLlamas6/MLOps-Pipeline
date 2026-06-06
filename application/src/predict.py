# predict.py
# ---------------------------------------------------------------------------
# PURPOSE: FastAPI router that exposes the POST /api/predict/ endpoint.
#          It receives a single customer's features, runs them through the
#          in-memory sklearn Pipeline, and returns the churn probability.
#
# LEARNING NOTES:
#   - APIRouter lets you organise endpoints into separate files and mount
#     them with a prefix in create_app.py — keeps code clean as the API grows.
#
#   - Pydantic BaseModel with Field(...) performs automatic input validation:
#     FastAPI will return HTTP 422 Unprocessable Entity if any field is missing
#     or of the wrong type, before your handler code even runs.
#
#   - `request.app.state.model` is how handlers access the globally-loaded
#     model without importing it directly — decoupled and testable.
#
#   - predict_proba returns [[P(class=0), P(class=1)], ...].
#     [:, 1] extracts the churn probability for every row.
#     We pass a single row so [0] gives the scalar probability.
# ---------------------------------------------------------------------------

from fastapi import APIRouter, HTTPException, Depends, Request
import pandas as pd
from pydantic import BaseModel, Field

router = APIRouter()


# ------------------------------------------------------------------ #
# REQUEST SCHEMA                                                       #
# ------------------------------------------------------------------ #
# Pydantic model that documents and validates the expected JSON body.
# The `example` values are shown in the Swagger UI (/docs) and make it
# easy to test the endpoint without writing a curl command by hand.
class ChurnInput(BaseModel):
    tenure: float        = Field(..., example=12,                          description="Number of months the customer has stayed with the company.")
    MonthlyCharges: float = Field(..., example=70.5,                       description="Monthly amount charged to the customer.")
    TotalCharges: float   = Field(..., example=850.0,                      description="Total amount charged to the customer.")
    gender: str           = Field(..., example="Male",                     description="Male or Female.")
    Partner: str          = Field(..., example="No",                       description="Whether the customer has a partner (Yes/No).")
    Dependents: str       = Field(..., example="No",                       description="Whether the customer has dependents (Yes/No).")
    PhoneService: str     = Field(..., example="Yes",                      description="Whether the customer has phone service (Yes/No).")
    InternetService: str  = Field(..., example="Fiber optic",              description="DSL, Fiber optic, or No.")
    Contract: str         = Field(..., example="Month-to-month",           description="Month-to-month, One year, Two year.")
    PaymentMethod: str    = Field(..., example="Credit card (automatic)",  description="Electronic check, Mailed check, Bank transfer, Credit card.")
    Tenure_Bin: str       = Field(..., example="1-2 yrs",                  description="Binned tenure: 0-1 yr, 1-2 yrs, 2-3 yrs, 3-4 yrs, 4-5 yrs, 5-6 yrs.")


# ------------------------------------------------------------------ #
# PREDICTION ENDPOINT                                                  #
# ------------------------------------------------------------------ #
@router.post("/predict/")
def predict(input_data: ChurnInput, request: Request):
    """
    Predict churn probability for a single customer.

    Returns a JSON object with `churn_probability` (float between 0 and 1).
    A value closer to 1 indicates a high likelihood of churn.

    Example request body:
    ```json
    {
      "tenure": 12,
      "MonthlyCharges": 70.5,
      "TotalCharges": 850.0,
      "gender": "Male",
      "Partner": "No",
      "Dependents": "No",
      "PhoneService": "Yes",
      "InternetService": "Fiber optic",
      "Contract": "Month-to-month",
      "PaymentMethod": "Credit card (automatic)",
      "Tenure_Bin": "1-2 yrs"
    }
    ```
    """
    # Retrieve the model stored in application state at startup.
    # If app.state.model is None it means startup failed — surface that clearly.
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    # Convert the Pydantic model to a single-row DataFrame.
    # The sklearn Pipeline's ColumnTransformer expects a DataFrame (not a dict)
    # so it can select columns by name.
    input_df = pd.DataFrame([input_data.dict()])

    try:
        # model.predict_proba(input_df) returns shape (1, 2).
        # [:, 1] selects the second column = probability of churn (class 1).
        predictions = model.predict_proba(input_df)[:, 1]

        # numpy.float32 is not JSON-serialisable by default; cast to Python float.
        churn_probability = float(predictions[0])

        return {"churn_probability": churn_probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
