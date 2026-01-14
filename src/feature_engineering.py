import pandas as pd
import datetime as dt

def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    NOW = dt.datetime(2011, 12, 10)

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (NOW - x.max()).days,
        "InvoiceNo": "nunique",
        "Quantity": "sum",
        "UnitPrice": "mean"
    })

    rfm.columns = ["Recency", "Frequency", "Quantity", "UnitPrice"]
    rfm["Monetary"] = rfm["Quantity"] * rfm["UnitPrice"]

    return rfm[["Recency", "Frequency", "Monetary"]]
