
### Klasyfikatory Opis

* **Drzewo Decyzyjne (Decision Tree):**
 Model budujący strukturę w formie drzewa, gdzie każdy węzeł reprezentuje pytanie o cechę (np. "Czy wydatki > 1000?"), a gałęzie prowadzą do odpowiedzi. Dzieli przestrzeń danych na prostokątne obszary, dążąc do maksymalnej jednorodności w liściach. Jest łatwy w interpretacji.

* **Naiwny Klasyfikator Bayesa (Naive Bayes):**
 Model probabilistyczny oparty na twierdzeniu Bayesa. Zakłada (stąd "naiwny"), że wszystkie cechy są od siebie niezależne. Oblicza prawdopodobieństwo przynależności do klasy i wybiera najbardziej prawdopodobną. Działa szybko i stabilnie, często używany jako baseline.

* **Random Forest (Las Losowy):** 
 Metoda zespołowa (Bagging), która trenuje wiele drzew decyzyjnych na losowych podzbiorach danych i cech. Wynik końcowy to "głosowanie" wszystkich drzew. Redukuje ryzyko przeuczenia (overfittingu) i jest znacznie stabilniejszy niż pojedyncze drzewo.

* **XGBoost (eXtreme Gradient Boosting):** 
 Zaawansowana metoda zespołowa oparta na boostingu. Buduje drzewa sekwencyjnie - każde kolejne drzewo naprawia błędy popełnione przez poprzednie. Jest to obecnie jeden z najskuteczniejszych algorytmów dla danych tabelarycznych, charakteryzujący się wysoką precyzją.

### Algorytmy segmentacyjne opis:

* **K-Means:**
 Algorytm podziałowy, który grupuje dane w k klastrów. Działa iteracyjnie, przesuwając środki klastrów (centroidy), aby zminimalizować odległość punktów od centrum własnej grupy. Wymaga wcześniejszego podania liczby grup i najlepiej sprawdza się przy sferycznych skupiskach.

* **DBSCAN:**
 Algorytm oparty na gęstości. Nie wymaga podawania liczby klastrów. Łączy punkty leżące blisko siebie w obszary o wysokiej gęstości. Jego unikalną cechą jest umiejętność wykrywania szumu (outliers) – punktów, które nie pasują do żadnej grupy.



### Metryki oceny modelu (Evaluation Metrics)

* **Precision (Precyzja)**
Mówi o wiarygodności modelu: "Gdy model twierdzi, że klient jest VIP-em, to na ile procent ma rację?". Wysoka wartość oznacza, że model rzadko wszczyna fałszywe alarmy.
* **Recall (Czułość)**
Mówi o wykrywalności: "Jaki procent wszystkich prawdziwych VIP-ów udało się modelowi odnaleźć?". Wysoka wartość oznacza, że model prawie nikogo nie pominął z danej grupy.
* **F1-score**
To średnia harmoniczna precyzji i czułości (Recall), stanowiąca jedną, zbalansowaną ocenę jakości modelu dla danego segmentu. Jest szczególnie przydatna, gdy chcemy uniknąć sytuacji, w której model jest świetny w jednym aspekcie, a fatalny w drugim.
* **Support**
To po prostu liczba klientów z danego segmentu, która znalazła się w zbiorze testowym. Informuje nas, na jak dużej próbce danych obliczono powyższe wyniki (np. 1302 to łączna liczba testowanych klientów).
* **Accuracy (Dokładność)**
Ogólny procent poprawnych decyzji modelu dla wszystkich segmentów łącznie (wynik 0.92 oznacza 92% trafności). Jest to najprostsza miara sukcesu, mówiąca jak często model "trafia" w dobrą etykietę.
* **Macro avg**
Średnia wyników wszystkich segmentów, gdzie każdy segment (mały czy duży) traktowany jest tak samo ważnie. Pozwala sprawdzić, czy model nie zaniedbuje najmniejszych grup (np. nielicznych VIP-ów).
* **Weighted avg**
Średnia ważona wyników, gdzie waga zależy od liczebności grupy (Support). Lepiej oddaje skuteczność biznesową, ponieważ błędy na dużej grupie klientów obniżają ten wynik bardziej niż błędy na małej.

### Charakterystyka Segmentów (K-Means)

* **Segment 0: "Uśpieni / Odchodzący"** (Recency ~152 dni, Monetary ~271)
To klienci nieaktywni, którzy ostatni zakup zrobili średnio 5 miesięcy temu i wydają najmniej pieniędzy. Zazwyczaj są to osoby jednorazowe, które przestały wracać do sklepu.
* **Segment 1: "VIP / Hurt"** (Recency ~21 dni, Monetary ~10 178)
Elitarna grupa klientów odwiedzających sklep bardzo często (średnio co 3 tygodnie) i generująca ogromne obroty. Kupują oni towar w ilościach hurtowych, stanowiąc najcenniejszą część bazy.
* **Segment 2: "Standardowi / Lojalni"** (Recency ~59 dni, Monetary ~1400)
Grupa regularnych klientów, którzy powracają do sklepu średnio co 2 miesiące i robią zakupy o umiarkowanej wartości. Stanowią stabilny, liczny środek bazy klienckiej, plasujący się pomiędzy VIP-ami a odchodzącymi.

Jasne, oto krótkie definicje tych cech, pasujące stylem do poprzedniej notatki:

### Atrybuty Klienta (Cechy modelu, wektor cech)

**Analiza RFM (Podstawowa):**
* **Recency** (zmienna: `Recency`) - liczba dni od ostatniego zakupu klienta.
* **Frequency** (zmienna: `Frequency`) - liczba unikalnych transakcji (faktur).
* **Monetary** (zmienna: `Monetary`) - łączna suma wydatków klienta (po odjęciu zwrotów).

**Analiza Produktowa i Wolumenowa:**
* **Total Quantity** (zmienna: `TotalQuantity`) - całkowita liczba zakupionych sztuk produktów (fizyczna wielkość zamówień).
* **Unique Products** (zmienna: `UniqueProducts`) - różnorodność koszyka, mierzona liczbą unikalnych kodów produktów (`StockCode`).

**Analiza Wartości Koszyka:**
* **Average Ticket Value** (zmienna: `AvgTicketValue`) - średnia wartość pojedynczej transakcji.
* **Min Ticket Value** (zmienna: `MinTicketValue`) - wartość najmniejszego zamówienia.
* **Max Ticket Value** (zmienna: `MaxTicketValue`) - wartość największego zamówienia.

**Analiza Behawioralna i Czasowa:**
* **Average Days Between Purchases** (zmienna: `AvgDaysBetweenPurchases`) - średni odstęp czasu (w dniach) między kolejnymi zakupami.
* **Favorite Day** (zmienna: `FavoriteDay`) - dzień tygodnia, w którym klient najczęściej dokonuje zakupów (dominanta).

**Analiza Zwrotów:**
* **Return Count** (zmienna: `ReturnCount`) - liczba transakcji zwrotu (faktur z ujemną ilością).
* **Returned Unique Products** (zmienna: `ReturnedUniqueProducts`) - liczba unikalnych produktów, które klient zdecydował się zwrócić.


Oto skondensowana notatka, idealna do szybkiego zerkania podczas obrony. Zawiera wszystko, co musisz wiedzieć, w punktach.


#### TRANSFORMACJA LOGARYTMICZNA (`np.log1p`)

**1. Gdzie tego używam?**

* **Plik:** `segmentation.py`
* **Moment:** *Przed* skalowaniem (`StandardScaler`) i *przed* wrzuceniem danych do K-Means.
* **Cechy:** `Monetary`, `Recency`, `TotalQuantity`, `AvgTicketValue` (wszystkie cechy o rozkładzie skośnym).

**2. Jaki problem rozwiązuje?**

* **Problem:** Dane e-commerce mają **rozkład prawostronnie skośny** (Pareto). Mamy kilku klientów "wielorybów" (obrót 1 mln zł) i tysiące klientów detalicznych (obrót 50 zł).
* **Ryzyko:** Algorytm K-Means używa **odległości euklidesowej**. Bez logarytmu klient z milionem byłby tak daleko od reszty, że zdominowałby klastrowanie (powstałby klaster "tylko dla niego", a reszta klientów zostałaby ściśnięta w jeden worek).

**3. Jak to działa (Mechanizm)?**

* **Kompresja skali:** Logarytm "ściska" gigantyczne wartości, a małe zostawia w spokoju.
* Różnica między 10 zł a 100 zł jest dla logarytmu podobnie ważna jak różnica między 10 000 zł a 100 000 zł.


* **Normalizacja:** Zmienia rozkład danych z "długiego ogona" na zbliżony do **rozkładu normalnego (Gaussa)**, co jest optymalne dla K-Means i algorytmów klasyfikacyjnych (Bayes).

**4. Dlaczego `np.log1p`, a nie zwykłe `np.log`? (Pytanie pułapka)**

* Zwykły logarytm naturalny z zera to minus nieskończoność (). To wyrzuciłoby błąd w kodzie.
* W danych mamy zera (np. `Recency = 0` dla kogoś, kto kupił dzisiaj).
* **`np.log1p(x)`** liczy .
* Dla  mamy .
* Dzięki temu funkcja jest bezpieczna i nie wymaga usuwania zerowych rekordów.

```
--- Model: Random Forest ---

--- Ocena modelu: Random Forest ---
Accuracy: 1.0000

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       527
           1       1.00      1.00      1.00       165
           2       1.00      1.00      1.00       610

    accuracy                           1.00      1302
   macro avg       1.00      1.00      1.00      1302
weighted avg       1.00      1.00      1.00      1302

Confusion Matrix (text):
[[527   0   0]
 [  0 165   0]
 [  0   0 610]]
Zapisano wykres ważności cech dla Random Forest.

--- Model: XGBoost ---

--- Ocena modelu: XGBoost ---
Accuracy: 1.0000

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       527
           1       1.00      1.00      1.00       165
           2       1.00      1.00      1.00       610

    accuracy                           1.00      1302
   macro avg       1.00      1.00      1.00      1302
weighted avg       1.00      1.00      1.00      1302

Confusion Matrix (text):
[[527   0   0]
 [  0 165   0]
 [  0   0 610]]
Zapisano wykres ważności cech: Visualizations/MGR/Feature_Importance_XGBoost.png
```

### Interpretacja wyników eksperymentu (MGR)
W ramach eksperymentu porównano podejście klasyczne (K-Means) z gęstościowym (DBSCAN) oraz zweryfikowano spójność segmentów za pomocą klasyfikatorów.

#### **Wyniki Segmentacji:**

**K-Means:** Wymusił podział klientów na 3 segmenty (VIP, Standard, Uśpieni). Podział ten jest użyteczny biznesowo, pozwalając na różnicowanie strategii marketingowej.

**DBSCAN:** Wykrył 1 główny klaster oraz 33 punkty szumu (outliers). Oznacza to, że w przestrzeni cech dane klientów tworzą jedną, ciągłą chmurę ("continuum"), a nie wyraźnie odseparowane wyspy. Klienci płynnie przechodzą z jednej grupy do drugiej. Algorytm skutecznie zidentyfikował nietypowe przypadki (anomalie).

#### **Wyniki Klasyfikacji (Random Forest & XGBoost):**
Oba modele osiągnęły Accuracy = 1.00 (100%).
Perfekcyjny wynik wynika z faktu, że etykiety (Segmenty) zostały wygenerowane matematycznie przez K-Means na podstawie tych samych cech, na których uczyły się klasyfikatory.

**Dlaczego Accuracy wynosi 1.0 (100%)?**
To zjawisko w Data Science nazywamy "wyciekiem etykiet" (label leakage) w specyficznym kontekście, ale tutaj jest to oczekiwane.

Mechanizm jest taki, że algorytm K-Means podzielił klientów na grupy (Segmenty 0, 1, 2) na podstawie cech matematycznych (np. ile wydali, jak często kupują).

Klasyfikatory (XGBoost/RF) dostały te same cechy i dostały zadanie: "Zgadnij, jaki segment nadał K-Means".

**Wniosek:** Ponieważ K-Means dzieli przestrzeń bardzo precyzyjnie (matematycznie), potężne modele jak XGBoost czy Random Forest po prostu "nauczyły się" wzoru K-Means. To tak, jakbyś pociął ciasto nożem na 3 kawałki, a potem kazał robotowi zgadnąć, który kawałek jest który, na podstawie tego, gdzie leży. Robot zgadnie w 100%.

#### **Wniosek ogólny:** 
Świadczy to o tym, że wyznaczone segmenty są doskonale separowalne w przyjętej przestrzeni 12 cech. Modele ML bezbłędnie odtworzyły reguły podziału narzucone przez K-Means.


### Krotki opis wizualizacji

**Macierz Konfuzji (Confusion Matrix - tekstowa z terminala bo duzej nie generowalem dla maina mgr):**

Dla modeli RF i XGBoost macierz jest idealna (wartości tylko na przekątnej). Oznacza to brak błędów: model ani razu nie pomylił klienta VIP z Uśpionym itp. Potwierdza to deterministyczny charakter podziału K-Means w tym zbiorze danych.

**PCA (Principal Component Analysis):**

Wykres rzutuje 12-wymiarową przestrzeń cech na 2 wymiary (2D).

Widoczne, wyraźnie oddzielone kolory (grupy) potwierdzają, że algorytm K-Means znalazł logiczny podział danych, a segmenty nie nakładają się na siebie w znaczący sposób.

**Ważność Cech (Feature Importance):**

Wykres słupkowy pokazuje, które atrybuty miały największy wpływ na przypisanie klienta do segmentu.

Cechy takie jak TotalQuantity (całkowita liczba produktów), Monetary (wydana kwota) czy Recency (czas od zakupu) zazwyczaj dominują, co potwierdza, że segmentacja opiera się na rzeczywistych zachowaniach zakupowych, a nie na szumie.



### Ogólniki:
#### 1. SZYBKIE PORÓWNANIE: Segmentacja vs Klasyfikacja

| Cecha | **Algorytm Segmentacyjny (K-Means)** | **Klasyfikator (Drzewo Decyzyjne / Bayes)** |
| :--- | :--- | :--- |
| **Typ uczenia** | **Nienadzorowane** (Unsupervised) | **Nadzorowane** (Supervised) |
| **Główny cel** | **ODKRYWANIE** wiedzy. Szukanie naturalnych grup i wzorców w danych. | **PRZEWIDYWANIE**. Przypisywanie nowych obiektów do znanych już grup. |
| **Etykiety (Labels)** | **Brak.** Algorytm nie wie, kto jest kim. Sam musi stworzyć podział. | **Wymagane.** Algorytm musi mieć "nauczyciela" (korzysta z wyników segmentacji jako wzorca). |
| **W Twoim projekcie** | K-Means wziął surowe dane i **stworzył** grupy (VIP, Standard, Uśpieni). | Drzewo wzięło te grupy i **nauczyło się reguł**, jak rozpoznać VIP-a (np. "jeśli wydał > 10k"). |
| **Kiedy używamy?** | Gdy chcemy zrozumieć strukturę bazy klientów. | Gdy przychodzi **nowy klient** i chcemy automat, który od razu da mu etykietkę. |

#### 2. Uczenie NADZOROWANE VS NIENADZOROWANE
##### Uczenie Nadzorowane (Supervised Learning)
* **Dane:** Mamy dane wejściowe (cechy) ORAZ gotowe odpowiedzi (etykiety).
* **Cel:** Nauczyć model przewidywać te odpowiedzi dla nowych danych.
* **Analogia:** Uczeń w szkole z nauczycielem. Nauczyciel pokazuje obrazek i mówi: "To jest kot". Uczeń się uczy, żeby potem samemu rozpoznać kota.
* **W projekcie:**
    * To są Twoje **klasyfikatory** (Drzewo Decyzyjne, Naive Bayes).
    * One "widziały" etykiety (VIP, Standard, Uśpieni), które wcześniej stworzył K-Means, i uczyły się reguł, jak je rozpoznawać.

##### Uczenie Nienadzorowane (Unsupervised Learning)
* **Dane:** Mamy tylko surowe dane wejściowe. Nie ma "poprawnych odpowiedzi".
* **Cel:** Znaleźć ukrytą strukturę, wzorce lub grupy w danych.
* **Analogia:** Dziecko, które dostało pudełko klocków, ale bez instrukcji. Samo zaczyna układać je w grupy: "te są czerwone", "te są niebieskie", "te są duże".
* **W projekcie:**
    * To jest Twój algorytm **K-Means** (Segmentacja).
    * Na początku nie wiedziałeś, kto jest VIP-em. Algorytm sam to odkrył, analizując, kto jest podobny do kogo.
    * Do tej grupy należy też **PCA** (redukcja wymiarów), które samo znalazło najważniejsze osie wariancji.




    KOLEJNE SPOTKANIE śRODA, DO WTORKU WYSLAC PACZKE Z RZECZAMI NA ZALIXZENIE OBYDWU PRACOWNI  - od 9:00 do 14:30 max --> stworzyc spotkanie w ms teams zarezerwowac godzine na spotkanie