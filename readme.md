# MED – Klasyfikacja klientów e-commerce

Projekt na przedmiot **Metody Eksploracji Danych (MED)**.

## Cel
Zastosowanie algorytmów:
- Drzewo decyzyjne
- Naiwny klasyfikator Bayesa

do klasyfikacji klientów e-commerce na podstawie ich zachowań zakupowych.

## Dane
Online Retail Dataset (Kaggle)

## Pipeline
1. Wczytanie danych
2. Czyszczenie
3. Feature engineering (RFM)
4. Segmentacja klientów (k-means)
5. Klasyfikacja segmentów:
   - Decision Tree
   - Naive Bayes
6. Ewaluacja modeli

## Uruchomienie
```bash
pip install -r requirements.txt
python src/main.py
```