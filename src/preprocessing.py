import pandas as pd

"""
    Wstępne czyszczenie danych:
    1. Usuwa rekordy bez CustomerID.
    2. Konwertuje daty.
    3. Oblicza całkowitą wartość pozycji (TotalPrice).
    UWAGA: Nie usuwamy tutaj ujemnych Quantity (zwrotów), 
    ponieważ są potrzebne do Feature Engineeringu.
"""



def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    # Usunięcie brakujących ID klientów
    df = df.dropna(subset=["CustomerID"])
    
    # Konwersja daty
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    
    # Obliczenie wartości pozycji (może być ujemna dla zwrotów)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    
    return df