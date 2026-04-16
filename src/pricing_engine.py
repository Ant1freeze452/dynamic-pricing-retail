"""
=============================================================================
АЛГОРИТМ ДИНАМИЧЕСКОГО ЦЕНООБРАЗОВАНИЯ
=============================================================================
Реализует двухуровневую стратегию расчёта оптимальной цены:

  Уровень 1 — Rule-based (Спринт 2):
    Быстрые эвристические правила. Нет математики — только бизнес-логика.
    «Если конкурент дешевле на 10% → снизить на 5%»

  Уровень 2 — Mathematical Optimization (Спринт 3):
    Находим максимум функции выручки: R(p) = p * Q(p)
    где Q(p) — спрос, оцененный через линейную регрессию.
    Оптимум: p* = (A + B * cost) / (2B)  — формула для линейной эластичности
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Tuple, Dict
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────
# УРОВЕНЬ 1: RULE-BASED АЛГОРИТМ
# ──────────────────────────────────────────────────

def apply_rule_based_pricing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Применяет набор бизнес-правил для коррекции цены.

    Правила (применяются последовательно, приоритет — сверху вниз):
      1. Если конкурент дешевле нас более чем на 10% → снизить цену на 5%
      2. Если продажи вчера = 0 → снизить на 1 рубль
      3. Если мы дешевле конкурента более чем на 15% → поднять цену на 3%
         (мы «оставляем деньги на столе» — можно было продать дороже)
      4. Иначе → цена не меняется

    Args:
        df: датафрейм с историей продаж (должен содержать last_day_sales)

    Returns:
        Копия датафрейма с добавленной колонкой 'rule_based_price'
    """
    result = df.copy()

    def _rule(row) -> float:
        price    = row["our_price"]
        comp     = row["competitor_price"]
        sales    = row.get("last_day_sales", row["sales_units"])

        # Правило 1: конкурент значительно дешевле → защищаем долю рынка
        if comp < price * 0.90:
            new_price = price * 0.95

        # Правило 2: нулевые продажи — нужно снизить цену
        elif sales == 0:
            new_price = price - 1.0

        # Правило 3: мы слишком дёшево продаём → можно поднять маржу
        elif price < comp * 0.85:
            new_price = price * 1.03

        # Правило 4: держим цену
        else:
            new_price = price

        return round(new_price, 0)

    result["rule_based_price"] = result.apply(_rule, axis=1)
    return result

# ──────────────────────────────────────────────────
# УРОВЕНЬ 2: МАТЕМАТИЧЕСКАЯ ОПТИМИЗАЦИЯ
# ──────────────────────────────────────────────────

def estimate_demand_model(df: pd.DataFrame, product: str) -> Tuple[float, float, float]:
    """
    Оценивает линейную модель спроса Q = A - B * P для конкретного товара.

    Почему линейная модель?
      Проста в интерпретации и достаточно точна при небольших
      диапазонах цен. Для каждого товара: Q(p) = A - B * p,
      где A — базовый спрос, B — чувствительность к цене.

    Args:
        df:      полный датафрейм с историей продаж
        product: название товара (строка, как в колонке 'product')

    Returns:
        Кортеж (A, B, R²):
          A  — свободный член (спрос при цене = 0, теоретически)
          B  — наклон (насколько падает спрос при росте цены на 1 руб)
          R² — коэффициент детерминации (качество подгонки модели)
    """
    subset = df[df["product"] == product].copy()

    X = subset[["our_price"]].values
    y = subset["sales_units"].values

    model = LinearRegression()
    model.fit(X, y)

    A   = model.intercept_          # свободный член
    B   = -model.coef_[0]           # делаем положительным: рост цены → падение спроса
    r2  = model.score(X, y)

    return A, B, r2


def find_optimal_price(A: float, B: float, cost: float,
                       min_price: float, max_price: float) -> Dict:
    """
    Находит цену, максимизирующую прибыль (не выручку!).

    Математика:
      Прибыль: π(p) = (p - cost) * Q(p) = (p - cost) * (A - B * p)
      Берём производную: dπ/dp = A - 2B*p + B*cost = 0
      Решение: p* = (A + B * cost) / (2B)

      Это классическая формула оптимальной цены монополиста.

    Args:
        A:         свободный член модели спроса
        B:         коэффициент при цене (положительный)
        cost:      себестоимость единицы товара
        min_price: нижняя граница цены (не продавать себе в убыток)
        max_price: верхняя граница цены (рыночное ограничение)

    Returns:
        Словарь с оптимальной ценой, спросом, выручкой и прибылью
    """
    if B <= 0:
        # Модель плохо подогнана — возвращаем середину диапазона
        return {
            "optimal_price":    round((min_price + max_price) / 2, 0),
            "expected_demand":  max(0, A - B * (min_price + max_price) / 2),
            "expected_revenue": 0,
            "expected_profit":  0,
            "method":           "fallback (B≤0)",
        }

    # Аналитический оптимум
    p_star = (A + B * cost) / (2 * B)

    # Ограничиваем ценовым диапазоном
    p_star = np.clip(p_star, min_price, max_price)

    expected_demand  = max(0, A - B * p_star)
    expected_revenue = p_star * expected_demand
    expected_profit  = (p_star - cost) * expected_demand

    return {
        "optimal_price":    round(p_star, 0),
        "expected_demand":  round(expected_demand, 1),
        "expected_revenue": round(expected_revenue, 2),
        "expected_profit":  round(expected_profit, 2),
        "method":           "analytical",
    }


def generate_pricing_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Главная функция: строит рекомендации по ценам для всех товаров.

    Для каждого товара:
      1. Обучает модель спроса на исторических данных
      2. Находит математически оптимальную цену
      3. Считает, как изменится выручка при переходе к новой цене
      4. Применяет rule-based корректировки как «защитный слой»

    Args:
        df: датафрейм с историей продаж (output data_generator.py)

    Returns:
        DataFrame с рекомендациями по каждому товару
    """
    from src.data_generator import PRODUCTS

    rows = []

    for product, params in PRODUCTS.items():
        if product not in df["product"].values:
            continue

        product_df = df[df["product"] == product]
        current_price = product_df["our_price"].iloc[-1]
        cost          = params["cost"]

        # ── Обучаем модель спроса ──
        A, B, r2 = estimate_demand_model(df, product)

        # ── Ищем оптимальную цену ──
        opt = find_optimal_price(
            A         = A,
            B         = B,
            cost      = cost,
            min_price = cost * 1.05,           # минимальная маржа 5%
            max_price = current_price * 1.30,  # не поднимать более чем на 30%
        )

        # ── Оцениваем текущую ситуацию ──
        last_7d     = product_df.tail(7)
        avg_sales   = last_7d["sales_units"].mean()
        avg_revenue = last_7d["revenue"].mean()
        avg_profit  = last_7d["profit"].mean()

        # ── Прогноз изменения ──
        delta_price   = opt["optimal_price"] - current_price
        delta_revenue = opt["expected_revenue"] - avg_revenue
        delta_profit  = opt["expected_profit"]  - avg_profit

        rows.append({
            "Товар":                  product,
            "Текущая цена (руб)":     current_price,
            "Оптимальная цена (руб)": opt["optimal_price"],
            "Δ Цена (руб)":           round(delta_price, 0),
            "Δ Цена (%)":             round(delta_price / current_price * 100, 1),
            "Прогноз продаж (шт/д)":  opt["expected_demand"],
            "Прогноз выручки (руб)":  opt["expected_revenue"],
            "Δ Выручка (руб)":        round(delta_revenue, 2),
            "Δ Прибыль (руб)":        round(delta_profit, 2),
            "R² модели":              round(r2, 3),
            "Метод":                  opt["method"],
        })

    report = pd.DataFrame(rows)
    return report

# ──────────────────────────────────────────────────
# ТОЧКА ВХОДА
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("data/sales_history.csv")

    print("Применяем rule-based правила...")
    df = apply_rule_based_pricing(df)

    print("Строим оптимальные цены через математическую модель...")
    report = generate_pricing_report(df)

    print("\nОТЧЁТ: Рекомендации по ценообразованию")
    print("=" * 80)
    print(report.to_string(index=False))

    report.to_csv("data/pricing_report.csv", index=False, encoding="utf-8-sig")
    print("\nОтчёт сохранён: data/pricing_report.csv")
