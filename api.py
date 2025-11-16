import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from DataHandler import DataPreProcessor 
from typing import Optional
import numpy as np

app = FastAPI()

ridge_model = joblib.load("ridge_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")

preprocessor = DataPreProcessor()

class HouseFeatures(BaseModel):
    FirstFlrSF: int = Field(alias='1stFlrSF')
    SecondFlrSF: int = Field(alias='2ndFlrSF')
    ThreeSsnPorch: int = Field(alias='3SsnPorch')
    GrLivArea: int
    Neighborhood: str  
    YearBuilt: int
    YearRemodAdd: int
    OverallQual: int
    OverallCond: int
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    TotRmsAbvGrd: int
    Fireplaces: int
    PoolArea: int
    YrSold: int
    MoSold: int
    LotArea: float
    LotShape: str
    LandContour: str
    LandSlope: str
    LotConfig: str
    Condition1: str
    Condition2: str
    BldgType: str
    HouseStyle: str
    RoofStyle: str
    RoofMatl: str
    ExterQual: str
    ExterCond: str
    Foundation: str
    Heating: str
    HeatingQC: str
    CentralAir: str
    Functional: str
    PavedDrive: str
    SaleCondition: str
    MSSubClass: int


    Alley: Optional[str] = None
    Fence: Optional[str] = None
    PoolQC: Optional[str] = None
    MiscFeature: Optional[str] = None
    FireplaceQu: Optional[str] = None
    BsmtQual: Optional[str] = None
    BsmtCond: Optional[str] = None
    BsmtExposure: Optional[str] = None
    BsmtFinType1: Optional[str] = None
    BsmtFinType2: Optional[str] = None
    GarageType: Optional[str] = None
    GarageFinish: Optional[str] = None
    GarageQual: Optional[str] = None
    GarageCond: Optional[str] = None
    MasVnrType: Optional[str] = None
    MasVnrArea: Optional[float] = 0.0
    BsmtFullBath: Optional[float] = 0.0
    BsmtHalfBath: Optional[float] = 0.0
    BsmtFinSF1: Optional[float] = 0.0
    BsmtFinSF2: Optional[float] = 0.0
    BsmtUnfSF: Optional[float] = 0.0
    TotalBsmtSF: Optional[float] = 0.0
    GarageYrBlt: Optional[float] = 0.0
    GarageCars: Optional[float] = 0.0
    GarageArea: Optional[float] = 0.0
    LotFrontage: Optional[float] = None
    Electrical: Optional[str] = None
    Exterior1st: Optional[str] = None
    Exterior2nd: Optional[str] = None
    KitchenQual: Optional[str] = None
    SaleType: Optional[str] = None
    MSZoning: Optional[str] = None
    ScreenPorch: int
    WoodDeckSF: int
    EnclosedPorch: int

#endpoint
@app.post("/predict")
def predict(features: HouseFeatures):
    

    input_df = pd.DataFrame([features.dict()])

    df_engineered = preprocessor.features_engineer(input_df)
    df_skewed = preprocessor.handle_skew(df_engineered)
    df_aligned = preprocessor.encode_and_align(df_skewed)

    df_scaled = preprocessor.transform_with_scaler(df_aligned)

    ridge_pred = ridge_model.predict(df_scaled)
    xgb_pred = xgb_model.predict(df_scaled)

    hybrid_pred = (ridge_pred * 0.75) + (xgb_pred * 0.25)
    final_price = np.expm1(hybrid_pred)

    return {"predicted_price": final_price[0]}