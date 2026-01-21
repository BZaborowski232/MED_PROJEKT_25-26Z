import pandas as pd

"""
    Wstępne czyszczenie danych:
    1. Usuwa rekordy bez CustomerID.
    2. Konwertuje daty.
    3. Oblicza całkowitą wartość pozycji (TotalPrice).
"""

def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    # Usunięcie brakujących ID klientów i utworzenie niezależnej kopii
    df = df.dropna(subset=["CustomerID"]).copy()
    
    # Konwersja daty (teraz bezpieczna, bo działamy na kopii)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    
    # Obliczenie wartości pozycji
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    
    return df