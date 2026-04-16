"""
=============================================================================
ГЕНЕРАТОР СИНТЕТИЧЕСКИХ ДАННЫХ ДЛЯ ДИНАМИЧЕСКОГО ЦЕНООБРАЗОВАНИЯ
=============================================================================
Модуль создаёт реалистичные данные о продажах:
  - Учитывает ценовую эластичность спроса
  - Симулирует поведение конкурентов
  - Добавляет сезонность и случайный шум
=============================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Параметры товаров
PRODUCTS = {
    "Молоко 1л":     {"base_price": 89,  "base_demand": 120, "elasticity": -1.8, "cost": 55},
    "Хлеб ржаной":   {"base_price": 65,  "base_demand": 90,  "elasticity": -1.3, "cost": 35},
    "Масло 200г":    {"base_price": 135, "base_demand": 60,  "elasticity": -2.1, "cost": 85},
    "Сыр Гауда 200г":{"base_price": 210, "base_demand": 40,  "elasticity": -2.5, "cost": 130},
    "Йогурт 250г":   {"base_price": 75,  "base_demand": 80,  "elasticity": -1.6, "cost": 45},
}

def compute_demand(base_demand, base_price, current_price, elasticity, 
                   competitor_price, seasonality, noise_std=5.0):
    """Рассчитывает спрос с учётом эластичности, конкурента и сезонности"""
    price_change_pct = (current_price - base_price) / base_price
    own_price_effect = 1 + elasticity * price_change_pct
    price_gap = (current_price - competitor_price) / current_price
    competitor_effect = 1 - 0.3 * max(0, price_gap)
    raw_demand = base_demand * own_price_effect * competitor_effect * seasonality
    noisy_demand = raw_demand + np.random.normal(0, noise_std)
    return max(0, int(round(noisy_demand)))

def generate_competitor_price(base_price, date):
    """Генерирует цену конкурента"""
    base_competitor = base_price * random.uniform(0.90, 1.05)
    if date.weekday() == 4:   # пятница
        base_competitor *= 0.95
    if date.day == 1:         # первое число
        base_competitor *= 0.93
    return round(base_competitor, 0)

def generate_our_price(base_price, day_num):
    """Имитирует нашу ценовую политику"""
    cycle = (day_num // 10) % 3
    adjustments = [1.0, 0.95, 1.05]
    return round(base_price * adjustments[cycle] * random.uniform(0.98, 1.02), 0)

def get_seasonality(date):
    """Возвращает коэффициент сезонности"""
    weekly = {0: 0.8, 1: 0.75, 2: 0.85, 3: 0.9, 4: 1.2, 5: 1.35, 6: 1.25}
    season = weekly[date.weekday()]
    day_of_year = date.timetuple().tm_yday
    annual = 1.0 + 0.1 * np.sin(2 * np.pi * (day_of_year - 180) / 365)
    return round(season * annual, 3)

def generate_sales_data(n_days=90, start_date="2024-01-01"):
    """Главная функция генерации датасета"""
    np.random.seed(42)
    random.seed(42)
    records = []
    start = datetime.strptime(start_date, "%Y-%m-%d")

    for day_num in range(n_days):
        date = start + timedelta(days=day_num)
        season = get_seasonality(date)

        for product_name, params in PRODUCTS.items():
            our_price = generate_our_price(params["base_price"], day_num)
            comp_price = generate_competitor_price(params["base_price"], date)
            sales = compute_demand(
                base_demand=params["base_demand"],
                base_price=params["base_price"],
                current_price=our_price,
                elasticity=params["elasticity"],
                competitor_price=comp_price,
                seasonality=season,
            )
            revenue = our_price * sales
            profit = (our_price - params["cost"]) * sales

            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "product_name": product_name,
                "our_price": our_price,
                "competitor_price": comp_price,
                "cost": params["cost"],
                "sales_qty": sales,
                "revenue": round(revenue, 2),
                "profit": round(profit, 2),
                "seasonality": season,
            })

    df = pd.DataFrame(records)
    print(f"Датасет сгенерирован: {len(df)} строк, {df['product_name'].nunique()} товаров")
    return df

if __name__ == "__main__":
    df = generate_sales_data(n_days=90)
    df.to_csv("data/synthetic_sales.csv", index=False, encoding="utf-8-sig")
    print("Файл сохранён: data/synthetic_sales.csv")
    print("\nПервые строки:")
    print(df.head(10).to_string(index=False))
    print(f"\nСтатистика продаж:\n{df['sales_qty'].describe().round(1)}")
