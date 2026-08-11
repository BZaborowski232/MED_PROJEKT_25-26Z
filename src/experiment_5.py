import os
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Importy modułów
from data_loader import load_data
from preprocessing import preprocess
from feature_engineering import engineer_features
from segmentation import segment_customers

def query_llama(prompt, model="llama3.1"):
    """
    Wysyła zapytanie do lokalnie uruchomionego modelu LLaMA przez Ollama API.
    """
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.0 # Zerowa temperatura = maksymalna logika, brak halucynacji
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            print(f"Błąd API: {response.status_code}")
            return None
    except Exception as e:
        print(f"Błąd połączenia z Ollama: {e}")
        print("Upewnij się, że aplikacja Ollama jest włączona, a model pobrany (komenda: ollama run llama3)")
        return None

def generate_customer_prompt(row):
    """
    Tłumaczy wektor cech na ustrukturyzowany prompt tekstowy dla LLM.
    """
    prompt = f"""You are an expert e-commerce data analyst. 
Based on the following transactional data of a customer, classify them into ONE of three segments:
0 - 'Sleeping / Departing' (low monetary value, long time since last purchase, very few transactions)
1 - 'VIP / Wholesale' (extremely high monetary value, high quantity, very frequent purchases)
2 - 'Standard / Loyal' (average monetary value, regular shopping patterns)

Customer Data:
- Days since last purchase (Recency): {row['Recency']:.0f} days
- Total number of transactions (Frequency): {row['Frequency']:.0f}
- Total money spent (Monetary): £{row['Monetary']:.2f}
- Total items purchased: {row['TotalQuantity']:.0f} items
- Average ticket value: £{row['AvgTicketValue']:.2f}

Based on these metrics, which segment (0, 1, or 2) does this customer belong to?
Provide ONLY the digit (0, 1, or 2) as your answer, with no additional text, punctuation, or explanation.
"""
    return prompt

def main():
    print("=== EKSPERYMENT E5: Klasyfikacja LLM (LLaMA 3) Zero-Shot ===")
    
    # 1. Wczytanie i przygotowanie danych
    print("\n[1/4] Przygotowywanie wektora cech...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "OnlineRetail.csv")
    
    raw_data = load_data(data_path)
    clean_data = preprocess(raw_data)
    features_df = engineer_features(clean_data)
    
    # K-MEANS - Tworzymy etykiety (Ground Truth) do oceny odpowiedzi LLM
    df_kmeans, _ = segment_customers(features_df, n_clusters=3)
    
    # Ustalenie, która klasa to VIP, Standard, a która Uśpieni
    stats = df_kmeans.groupby('Segment')['Monetary'].mean().sort_values()
    uspieni_segment = stats.index[0]
    standard_segment = stats.index[1]
    vip_segment = stats.index[2]
    
    # Remapowanie Ground Truth, aby zgadzało się z naszym Promptem dla LLM (0=Uśpieni, 1=VIP, 2=Standard)
    segment_mapping = {uspieni_segment: 0, vip_segment: 1, standard_segment: 2}
    df_kmeans['Ground_Truth'] = df_kmeans['Segment'].map(segment_mapping)
    
    # 2. Losowanie próby badawczej (Ewaluacja całego zbioru trwałaby za długo)
    sample_size = 100
    print(f"\n[2/4] Losowanie próby badawczej ({sample_size} klientów) do analizy przez LLM...")
    # Używamy stratify, aby mieć pewność, że w 100 klientach są reprezentanci każdej grupy
    df_sample = df_kmeans.sample(n=sample_size, random_state=42, weights='Ground_Truth')
    
    # Alternatywnie wymuszone losowanie proporcjonalne:
    # df_sample = df_kmeans.groupby('Ground_Truth', group_keys=False).apply(lambda x: x.sample(min(len(x), 33)))
    
    y_true = []
    y_pred = []
    
    print("\n[3/4] Komunikacja z modelem LLaMA (Ollama)...")
    for index, row in df_sample.iterrows():
        # Generowanie promptu
        prompt = generate_customer_prompt(row)
        
        # Prawdziwa odpowiedź
        true_label = int(row['Ground_Truth'])
        y_true.append(true_label)
        
        # Zapytanie do modelu LLM
        response = query_llama(prompt, model="llama3.1")
        
        # Parsowanie odpowiedzi (na wypadek gdyby model dopisał kropkę lub spację)
        predicted_label = -1
        if response:
            try:
                # Szukamy pierwszej cyfry w odpowiedzi
                import re
                match = re.search(r'\d', response)
                if match:
                    predicted_label = int(match.group(0))
            except Exception:
                pass
                
        y_pred.append(predicted_label)
        print(f"Klient {index} -> K-Means: {true_label} | LLM LLaMA: {predicted_label}")
    
    # 4. Ewaluacja i Wizualizacja
    print("\n[4/4] Podsumowanie wyników LLM...")
    
    # Odfiltrowanie ewentualnych błędów parsowania (-1)
    valid_indices = [i for i, p in enumerate(y_pred) if p in [0, 1, 2]]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]
    
    print(f"\nSkutecznie przeanalizowano: {len(y_true_valid)}/{sample_size} próbek.")
    print(f"Dokładność LLM (Accuracy): {accuracy_score(y_true_valid, y_pred_valid):.4f}")
    print("\nRaport Klasyfikacji (LLM vs K-Means):")
    print(classification_report(y_true_valid, y_pred_valid))
    
    # Generowanie macierzy konfuzji (używamy logiki z Twojego visualization.py)
    os.makedirs("Visualizations/MGR", exist_ok=True)
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1, 2])
    
    segment_names = ["0: Uśpieni", "1: VIP", "2: Standardowi"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False, 
                xticklabels=segment_names, yticklabels=segment_names)
    plt.title('Macierz Konfuzji - LLM (LLaMA 3) Zero-Shot Classification')
    plt.ylabel('Prawdziwa klasa (K-Means Ground Truth)')
    plt.xlabel('Przewidziana klasa (LLM LLaMA)')
    plt.tight_layout()
    plt.savefig("Visualizations/MGR/E5_LLM_Confusion_Matrix.png", dpi=300)
    print("\n--- ZAPISANO WYKRES: Visualizations/MGR/E5_LLM_Confusion_Matrix.png ---")

if __name__ == "__main__":
    main()