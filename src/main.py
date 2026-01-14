# Import funkcji do wczytywania i przetwarzania danych
from data_loader import load_data             # funkcja wczytująca CSV do pandas DataFrame
from preprocessing import preprocess          # funkcja czyszcząca dane (usuwa braki, zwroty)
from feature_engineering import calculate_rfm # funkcja obliczająca cechy RFM
from segmentation import segment_customers   # funkcja segmentująca klientów metodą k-means

# Import modeli klasyfikacyjnych
from classifiers.decision_tree import DecisionTreeModel
from classifiers.naive_bayes import NaiveBayesModel
from evaluation import evaluate_model        # funkcja do oceny jakości klasyfikacji (raport + macierz konfuzji)

# Import narzędzia do podziału danych na trening i test
from sklearn.model_selection import train_test_split

# Wczytywanie danych
df = load_data("../data/OnlineRetail.csv")
# df teraz zawiera wszystkie transakcje w postaci DataFrame
# każda linia = jedna pozycja na fakturze, kolumny: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

# Preprocessing czyli czyszczenie danych
df = preprocess(df)
# usuwa wiersze bez CustomerID
# usuwa transakcje z Quantity <= 0 (np. zwroty)
# konwertuje InvoiceDate na typ datetime
# efekt: dane gotowe do analizy, poprawne i spójne

# Feature engineering – RFM
rfm = calculate_rfm(df)
# agreguje dane po CustomerID:
# - Recency: liczba dni od ostatniego zakupu
# - Frequency: liczba unikalnych faktur
# - Monetary: suma wydatków
# teraz 1 wiersz = 1 klient, kolumny = cechy opisujące klienta

# Segmentacja klientów – k-means
rfm, X = segment_customers(rfm)
# standardyzacja cech (średnia=0, odchylenie std=1)
# k-means tworzy 3 segmenty klientów (0,1,2)
# dodaje kolumnę "Segment" do rfm
# X = macierz cech wejściowych do klasyfikatorów

# etykiety do klasyfikacji
y = rfm["Segment"]  # Ground truth dla modeli klasyfikacyjnych

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# 70% danych do uczenia, 30% do testów
# random_state to parametr który zapewnia powtarzalność wyników, tego na razie oczekuję
# bez tego za każdym uruchomieniem modele będą trenowane na innych danych
# co utrudnia debugowanie i porównywanie wyników, 42 to umowna liczba jaką założyłem (lubię tę liczbę)

# Decision Tree – uczenie i ewaluacja (procwes sprawdzania jakości modelu)
dt = DecisionTreeModel()    # inicjalizacja modelu drzewa decyzyjnego
dt.train(X_train, y_train)  # nauka modelu na danych treningowych
y_pred_dt = dt.predict(X_test)  # predykcja segmentów dla danych testowych
evaluate_model(y_test, y_pred_dt, "Decision Tree")  # raport klasyfikacji + macierz konfuzji

# Naive Bayes – uczenie i ewaluacja (procwes sprawdzania jakości modelu)
nb = NaiveBayesModel()      # inicjalizacja modelu Naive Bayes
nb.train(X_train, y_train)  # nauka modelu
y_pred_nb = nb.predict(X_test)  # predykcja
evaluate_model(y_test, y_pred_nb, "Naive Bayes")    # raport klasyfikacji + macierz konfuzji
