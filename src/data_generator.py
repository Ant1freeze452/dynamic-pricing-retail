"""
=============================================================================
ГЕНЕРАТОР СИНТЕТИЧЕСКИХ ДАННЫХ ДЛЯ ДИНАМИЧЕСКОГО ЦЕНООБРАЗОВАНИЯ
=============================================================================
Модуль создаёт реалистичные данные о продажах:
  - Учитывает ценовую эластичность спроса (чем ниже цена → тем больше продаж)
  - Симулирует поведение конкурентов
  - Добавляет сезонность и случайный шум для правдоподобия
  - Генерирует CSV-файл, готовый к загрузке в алгоритм
=============================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


# ──────────────────────────────────────────────────
# ПАРАМЕТРЫ ТОВАРОВ
# Каждый товар описывается: базовой ценой, коэффициентами эластичности
# ──────────────────────────────────────────────────
PRODUCTS = {
    "Молоко 1л":     {"base_price": 89,  "base_demand": 120, "elasticity": -1.8, "cost": 55},
    "Хлеб ржаной":  {"base_price": 65,  "base_demand": 90,  "elasticity": -1.3, "cost": 35},
    "Масло 200г":    {"base_price": 135, "base_demand": 60,  "elasticity": -2.1, "cost": 85},
    "Сыр Гауда 200г":{"base_price": 210, "base_demand": 40,  "elasticity": -2.5, "cost": 130},
    "Йогурт 250г":   {"base_price": 75,  "base_demand": 80,  "elasticity": -1.6, "cost": 45},
}


def compute_demand(base_demand: float, base_price: float, current_price: float,
                   elasticity: float, competitor_price: float,
                   seasonality: float, noise_std: float = 5.0) -> int:
    """
    Рассчитывает спрос на основе:
      1. Ценовой эластичности   → как цена влияет на наш спрос
      2. Цены конкурента        → если он дешевле, часть покупателей уходит
      3. Сезонности             → недельные и праздничные колебания
      4. Случайного шума        → имитация непредсказуемости рынка

    Формула: Demand = D_base * (1 + e * ΔP/P) * competitor_effect * season

    Args:
        base_demand:      базовый уровень продаж при стандартной цене
        base_price:       стандартная (справочная) цена товара
        current_price:    наша текущая цена
        elasticity:       коэффициент эластичности (отрицательный, напр. -1.8)
        competitor_price: цена конкурента на аналогичный товар
        seasonality:      множитель сезонности (1.0 = норма)
        noise_std:        стандартное отклонение случайного шума

    Returns:
        Количество проданных единиц (целое, минимум 0)
    """
    # Эффект собственной цены (эластичность)
    price_change_pct = (current_price - base_price) / base_price
    own_price_effect = 1 + elasticity * price_change_pct

    # Эффект конкурента: если его цена ниже нашей, теряем долю рынка
    price_gap = (current_price - competitor_price) / current_price
    competitor_effect = 1 - 0.3 * max(0, price_gap)  # макс. -30% при большой разнице

    # Итоговый спрос с шумом
    raw_demand = base_demand * own_price_effect * competitor_effect * seasonality
    noisy_demand = raw_demand + np.random.normal(0, noise_std)

    return max(0, int(round(noisy_demand)))


def generate_competitor_price(base_price: float, date: datetime) -> float:
    """
    Генерирует цену конкурента: иногда он делает акции,
    иногда поднимает цену вслед за ростом закупочных расходов.

    Args:
        base_price: наша базовая цена как ориентир
        date:       дата (конкуренты чаще делают акции по пятницам)

    Returns:
        Цена конкурента, округлённая до 1 рубля
    """
    # Базовое отклонение: конкурент в среднем на 3% дешевле
    base_competitor = base_price * random.uniform(0.90, 1.05)

    # Акции по пятницам и в начале месяца
    if date.weekday() == 4:   # пятница
        base_competitor *= 0.95
    if date.day == 1:         # первое число — конкурент распродаёт остатки
        base_competitor *= 0.93

    return round(base_competitor, 0)


def generate_our_price(base_price: float, day_num: int) -> float:
    """
    Имитирует нашу ценовую политику: цена меняется раз в несколько дней
    в зависимости от закупочных периодов и плановых акций.

    Args:
        base_price: базовая цена товара
        day_num:    номер дня с начала периода

    Returns:
        Наша цена на данный день
    """
    # Раз в 7-14 дней — плановый пересмотр цены
    cycle = (day_num // 10) % 3
    adjustments = [1.0, 0.95, 1.05]  # нормальная → акция → наценка
    return round(base_price * adjustments[cycle] * random.uniform(0.98, 1.02), 0)


def get_seasonality(date: datetime) -> float:
    """
    Возвращает коэффициент сезонности для данной даты.
    Продажи растут в выходные и праздники, падают в начале недели.

    Args:
        date: дата

    Returns:
        Множитель (0.7 – 1.4)
    """
    weekday = date.weekday()

    # Недельная цикличность: пн-вт тихо, пт-вс оживлённо
    weekly = {0: 0.8, 1: 0.75, 2: 0.85, 3: 0.9, 4: 1.2, 5: 1.35, 6: 1.25}
    season = weekly[weekday]

    # Небольшая годовая синусоида (лето чуть хуже, декабрь лучше)
    day_of_year = date.timetuple().tm_yday
    annual = 1.0 + 0.1 * np.sin(2 * np.pi * (day_of_year - 180) / 365)

    return round(season * annual, 3)


def generate_sales_data(n_days: int = 90, start_date: str = "2026-01-01") -> pd.DataFrame:
    """
    Главная функция генерации датасета.
    Создаёт историю продаж за n_days дней для всех товаров в PRODUCTS.

    Args:
        n_days:     количество дней истории (по умолчанию 90)
        start_date: дата начала периода (YYYY-MM-DD)

    Returns:
        DataFrame с колонками:
          date, product, our_price, competitor_price,
          sales_units, revenue, profit, seasonality
    """
    np.random.seed(42)   # для воспроизводимости результатов
    random.seed(42)

    records = []
    start = datetime.strptime(start_date, "%Y-%m-%d")

    for day_num in range(n_days):
        date = start + timedelta(days=day_num)
        season = get_seasonality(date)

        for product_name, params in PRODUCTS.items():
            our_price      = generate_our_price(params["base_price"], day_num)
            comp_price     = generate_competitor_price(params["base_price"], date)
            sales          = compute_demand(
                                base_demand      = params["base_demand"],
                                base_price       = params["base_price"],
                                current_price    = our_price,
                                elasticity       = params["elasticity"],
                                competitor_price = comp_price,
                                seasonality      = season,
                             )
            revenue = our_price * sales
            profit  = (our_price - params["cost"]) * sales

            records.append({
                "date":             date.strftime("%Y-%m-%d"),
                "product":          product_name,
                "our_price":        our_price,
                "competitor_price": comp_price,
                "cost":             params["cost"],
                "sales_units":      sales,
                "revenue":          round(revenue, 2),
                "profit":           round(profit, 2),
                "seasonality":      season,
            })

    df = pd.DataFrame(records)
    print(f"Датасет сгенерирован: {len(df)} строк, {df['product'].nunique()} товара(ов), "
          f"период {df['date'].min()} – {df['date'].max()}")
    return df

# ──────────────────────────────────────────────────
# ТОЧКА ВХОДА
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    df = generate_sales_data(n_days=90)
    output_path = "data/sales_history.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Файл сохранён: {output_path}")
    print("\n Первые строки датасета:")
    print(df.head(10).to_string(index=False))
    print(f"\n Статистика продаж:\n{df['sales_units'].describe().round(1)}")
