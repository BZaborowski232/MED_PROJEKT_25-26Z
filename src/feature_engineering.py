import pandas as pd
import datetime as dt
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    """
        Tworzy rozszerzony zestaw cech dla każdego klienta:
        - RFM (Recency, Frequency, Monetary)
        - Statystyki koszyka (Min, Max, Mean)
        - Statystyki produktowe (Unikalne produkty, Ilość sztuk)
        - Zachowania czasowe (Ulubiony dzień, Średni czas między zakupami)
        - Zwroty
    """

    # Data odniesienia (dzień po ostatniej transakcji w zbiorze)
    NOW = dt.datetime(2011, 12, 10)

    # Rozdzielenie zakupów i zwrotów
    # Zakupy: Quantity > 0
    purchases = df[df["Quantity"] > 0].copy()
    # Zwroty: Quantity < 0
    returns = df[df["Quantity"] < 0].copy()

    # --- 1. Podstawowe agregacje RFM i produktowe (na podstawie zakupów) ---
    # Najpierw grupujemy po fakturze, aby mieć poprawne sumy dla koszyka
    invoice_stats = purchases.groupby(["CustomerID", "InvoiceNo"]).agg({
        "TotalPrice": "sum",
        "InvoiceDate": "min"
    }).reset_index()

    # Agregacja per Klient
    features = purchases.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (NOW - x.max()).days, # Recency
        "InvoiceNo": "nunique",                        # Frequency
        "StockCode": "nunique",                        # Liczba różnych produktów
        "Quantity": "sum",                             # Całkowita liczba kupionych sztuk
        "TotalPrice": "sum"                            # Monetary (Suma wydatków)
    })
    
    features.columns = ["Recency", "Frequency", "UniqueProducts", "TotalQuantity", "Monetary"]

    # --- 2. Statystyki wartości transakcji (Min, Max, Avg) ---
    spending_stats = invoice_stats.groupby("CustomerID")["TotalPrice"].agg(
        ["mean", "min", "max"]
    )
    spending_stats.columns = ["AvgTicketValue", "MinTicketValue", "MaxTicketValue"]
    
    features = features.join(spending_stats)

    # --- 3. Statystyki czasowe (Behawioralne) ---
    
    # Funkcja: Średnia liczba dni między zakupami
    def calculate_avg_days_between(dates):
        if len(dates) < 2:
            return 0
        # Sortowanie i różnica w dniach
        return dates.sort_values().diff().dt.days.mean()

    # Funkcja: Ulubiony dzień tygodnia (0=Poniedziałek, 6=Niedziela)
    def get_favorite_day(dates):
        return dates.dt.dayofweek.mode()[0]

    time_stats = purchases.groupby("CustomerID")["InvoiceDate"].agg(
        [calculate_avg_days_between, get_favorite_day]
    )
    time_stats.columns = ["AvgDaysBetweenPurchases", "FavoriteDay"]
    # Wypełniamy NaN zerami (dla klientów z 1 zakupem avg_days będzie NaN)
    time_stats["AvgDaysBetweenPurchases"] = time_stats["AvgDaysBetweenPurchases"].fillna(0)
    
    features = features.join(time_stats)

    # --- 4. Statystyki Zwrotów ---
    
    return_stats = returns.groupby("CustomerID").agg({
        "InvoiceNo": "nunique",    # Liczba zwrotów
        "StockCode": "nunique"     # Liczba różnych zwróconych produktów
    })
    return_stats.columns = ["ReturnCount", "ReturnedUniqueProducts"]

    # Dołączamy do głównej tabeli (Left Join)
    features = features.join(return_stats)

    # Uzupełnienie braków dla osób, które nic nie zwróciły (NaN -> 0)
    features[["ReturnCount", "ReturnedUniqueProducts"]] = features[["ReturnCount", "ReturnedUniqueProducts"]].fillna(0)

    return features