## MGR - Rozszerzenie projektu i opis wstępnych wyników eksperymentów

Poniżej przedstawiono wyniki eksperymentu polegającego na segmentacji klientów (metody nienadzorowane) oraz próbie odtworzenia tych segmentów za pomocą klasyfikatorów (metody nadzorowane). Celem eksperymentu było nie tylko wyznaczenie grup klientów, ale również weryfikacja spójności matematycznej otrzymanych klastrów oraz zbadanie struktury danych w przestrzeni wielowymiarowej, rozszerzając projekt o wstępną implementację klasyfikatorów Random Forest oraz XGBoost, a także o algorytm DBSCAN.

### 1. Wyniki Segmentacji (Unsupervised Learning)
W pierwszej fazie eksperymentu porównano dwa odmienne podejścia do grupowania danych: oparte na odległościach (K-Means) oraz oparte na gęstości (DBSCAN).

#### 1.1. Algorytm K-Means

Zastosowanie algorytmu K-Means pozwoliło na wyodrębnienie trzech wyraźnych grup klientów. Podział ten, posiada wysoką użyteczność biznesową. Pozwala on na wdrożenie zróżnicowanych strategii marketingowych:

- Segment VIP: Klienci o najwyższej wartości transakcji.

- Segment Standard: Klienci regularni o przeciętnym koszyku.

- Segment Uśpieni: Klienci nieaktywni lub o niskim potencjale zakupowym.

#### 1.2. Algorytm DBSCAN

Zastosowanie algorytmu gęstościowego DBSCAN rzuciło nowe światło na topologię badanych danych. Algorytm zidentyfikował:

- 1 główny klaster obejmujący większość populacji.

- 33 punkty szumu (outliers), stanowiące anomalie.

Wynik ten sugeruje, że w naturalnej przestrzeni cech (12 wymiarów) dane klientów tworzą tzw. continuum. Oznacza to, że nie istnieją naturalne "wyspy" klientów oddzielone pustą przestrzenią, lecz jedna, ciągła chmura punktów, gdzie klienci płynnie przechodzą z jednej charakterystyki w drugą.

Choć DBSCAN poprawnie zidentyfikował strukturę ciągłą danych i skutecznie wykrył anomalie (nietypowe zachowania zakupowe), to w kontekście celu biznesowego pracy, jakim jest segmentacja marketingowa, algorytm K-Means okazał się bardziej praktyczny, dokonując dyskretyzacji tego ciągłego segmentu na mniejsze wartościowe w kontekście biznesowym grupy.

### 2. Wyniki Klasyfikacji (Supervised Learning)
W drugiej fazie eksperymentu wykorzystano etykiety wygenerowane przez algorytm K-Means jako zmienną celu dla modeli klasyfikacyjnych: Random Forest oraz XGBoost. Celem było sprawdzenie, czy zaawansowane modele są w stanie zrekonstruować reguły podziału na podstawie atrybutów wejściowych.

#### 2.1. Metryki Jakości Modeli

Oba modele osiągnęły bardzo wysokie, lecz realistyczne wyniki, co potwierdza silną korelację między cechami wejściowymi a wyznaczonymi segmentami.

```
--- Model: Random Forest ---

--- Ocena modelu: Random Forest ---
Accuracy: 0.9670

Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.97      0.98       527
           1       0.97      0.92      0.94       165
           2       0.95      0.98      0.97       610

    accuracy                           0.97      1302
   macro avg       0.97      0.95      0.96      1302
weighted avg       0.97      0.97      0.97      1302

Confusion Matrix (text):
[[512   0  15]
 [  0 151  14]
 [ 10   4 596]]
Zapisano wykres ważności cech dla Random Forest.

--- Model: XGBoost ---

--- Ocena modelu: XGBoost ---
Accuracy: 0.9747

Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.97      0.98       527
           1       0.98      0.95      0.97       165
           2       0.96      0.98      0.97       610

    accuracy                           0.97      1302
   macro avg       0.98      0.97      0.97      1302
weighted avg       0.97      0.97      0.97      1302

Confusion Matrix (text):
[[512   0  15]
 [  0 157   8]
 [  7   3 600]]
Zapisano wykres ważności cech: Visualizations/MGR/Feature_Importance_XGBoost.png

=== KONIEC EKSPERYMENTU ===
(Projekt) bartoszzaborowski@UniFi-63-200 Projekt % 
```

Model XGBoost okazał się nieznacznie lepszy, osiągając wyższą dokładność ogólną oraz lepszą precyzję w klasyfikacji kluczowego segmentu VIP. Wynik na poziomie ~97% świadczy o bardzo dobrej separowalności segmentów, przy zachowaniu marginesu błędu wynikającego z występowania klientów "granicznych" (znajdujących się pomiędzy centroidami).

#### 2.2 Analiza Macierzy Konfuzji


**Macierz Konfuzji dla XGBoost:**

```
   [[512   0  15]  - Klasa 0 (Uśpieni)
   [  0 157   8]   - Klasa 1 (VIP)
   [  7   3 600]]  - Klasa 2 (Standard)
```

Jak możemy zauważyć model ani razu nie pomylił klasy 0 (Uśpieni) z klasą 1 (VIP). Wartości w macierzy wskazują, że te dwie grupy są od siebie drastycznie różne.

Wszystkie błędy klasyfikacji (łącznie 33 przypadki w XGBoost) dotyczą mylenia segmentów skrajnych z segmentem środkowym ("Standard"). Przykładowo widzimy, że 8 klientów VIP zostało zaklasyfikowanych jako Standard. Są to prawdopodobnie klienci "aspirujący", którzy są blisko progu wejścia do grupy VIP, ale algorytm uznał ich parametry za zbyt słabe.

**Macierz Konfuzji dla Random Forest:**

```
   Confusion Matrix (text):
   [[512   0  15]    - Klasa 0 (Uśpieni)
   [  0 151  14]     - Klasa 1 (VIP)
   [ 10   4 596]]    - Klasa 2 (Standard)
```

W tym przypadku model również nie pomylił ani razu klasy 0 (Uśpieni) z klasą 1 (VIP).

Jak widać Random Forest pomylił 14 klientów VIP z grupą Standard, podczas gdy XGBoost tylko 8. Do zastosowań produkcyjnych lepszym wyborem okazałby się model XGBoost, ponieważ lepiej identyfikuje najcenniejszych klientów (wyższy Recall dla klasy 1: 95% vs 92% w RF).


### 3. Analiza Wizualna
W celu pogłębionej interpretacji wyników sporządzono szereg wizualizacji, które potwierdzają wnioski numeryczne.

#### 3.1. Redukcja Wymiarowości (PCA)

Wizualizacja rzutowania 12-wymiarowej przestrzeni cech na 2 wymiary za pomocą PCA (Principal Component Analysis) ukazuje wyraźnie odseparowane od siebie skupiska punktów (klastry).

Poszczególne kolory odpowiadające segmentom nie nakładają się na siebie w znaczący sposób i ewidentnie tworzą odseparowane skupiska. Jest to wizualny dowód na to, że algorytm K-Means znalazł logiczny i geometrycznie uzasadniony podział danych.

![PCA k-Means](Visualizations/MGR/PCA_K-Means_MGR.png)

#### 3.2. Ważność Cech (Feature Importance)

Analiza wykresów ważności cech dla modeli pozwala zidentyfikować atrybuty mające największy wpływ na przynależność do segmentu.

![Feature Importance XGBoost](Visualizations/MGR/Feature_Importance_XGBoost.png)

![Feature Importance RF](Visualizations/MGR/Feature_Importance_RF.png)

Największy wpływ na decyzje modeli mają między innymi cechy: TotalQuantity (całkowita liczba produktów), Monetary (łączna kwota wydatków) oraz Recency (czas od ostatniego zakupu). Nieznacznie niższy wynik osiągnęły takie cechy jak: AvgDaysBetweenPurchases, UniqueProducts czy też MaxTicketValue. Wyraźnie utwierdza nas to w przekonaniu, że modele efektywnie korzystają z szerokiego wektora cech.

### 4. Podsumowanie
Przeprowadzony eksperyment wykazał, że segmentacja K-Means jest stabilna i odtwarzalna, a modele klasyfikacyjne potrafią przewidzieć przydział do segmentu z dokładnością 97.5% na podstawie samych cech zakupowych. Błędy klasyfikacji (< ~3%) dotyczą wyłącznie przypadków granicznych, co jest zjawiskiem naturalnym w ciągłym zbiorze danych. Model XGBoost okazał się wstępnie lepszym narzędziem do klasyfikacji nowych klientów w systemie CRM, ze względu na wyższą czułość w wykrywaniu segmentu VIP.