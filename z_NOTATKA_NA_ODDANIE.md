
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

### Atrybuty Klienta (Cechy modelu)

* **Recency (Świeżość zakupu)**
Liczba dni, jakie upłynęły od ostatniej transakcji klienta do momentu analizy. Im mniejsza wartość, tym klient jest bardziej „aktywny” i na bieżąco z ofertą.
* **Frequency (Częstotliwość)**
Liczba unikalnych transakcji (faktur), jakie klient zrealizował w całym badanym okresie. Wysoka wartość oznacza klienta powracającego, który regularnie robi zakupy.
* **Monetary (Wartość pieniężna)**
Łączna kwota pieniędzy wydana przez klienta w sklepie (suma wartości wszystkich jego koszyków). Jest to kluczowy wskaźnik dochodowości danego klienta dla biznesu.
* **TotalQuantity (Całkowita liczba sztuk)**
Suma wszystkich fizycznych produktów (sztuk towaru), jakie klient zakupił. Pozwala odróżnić klientów detalicznych (mało sztuk, wysoka cena) od hurtowników (dużo sztuk, często niższa marża).