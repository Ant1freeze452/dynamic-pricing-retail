"""
=============================================================================
ТЕСТЫ АЛГОРИТМА ЦЕНООБРАЗОВАНИЯ
=============================================================================
Запуск: pytest tests/test_pricing.py -v

Тесты проверяют:
  1. Корректность генерации данных (эластичность работает?)
  2. Корректность rule-based логики
  3. Корректность математической оптимизации (p* > cost?)
=============================================================================
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from src.data_generator import generate_sales_data, compute_demand
from src.pricing_engine  import (apply_rule_based_pricing,
                                  estimate_demand_model,
                                  find_optimal_price)


class TestDataGenerator:
    """Проверяем, что генератор создаёт логичные данные."""

    def test_elasticity_direction(self):
        """При снижении цены спрос должен расти (отрицательная эластичность)."""
        demand_low  = compute_demand(100, 100, 80,  elasticity=-1.8, competitor_price=100, seasonality=1.0)
        demand_high = compute_demand(100, 100, 120, elasticity=-1.8, competitor_price=100, seasonality=1.0)
        assert demand_low > demand_high, "При низкой цене спрос должен быть выше"

    def test_competitor_effect(self):
        """Если конкурент дешевле, наши продажи должны упасть."""
        demand_comp_high = compute_demand(100, 100, 100, -1.8, competitor_price=120, seasonality=1.0)
        demand_comp_low  = compute_demand(100, 100, 100, -1.8, competitor_price=80,  seasonality=1.0)
        assert demand_comp_high > demand_comp_low

    def test_output_shape(self):
        """Датасет должен содержать правильное количество строк."""
        df = generate_sales_data(n_days=10)
        assert len(df) == 10 * 5  # 10 дней × 5 товаров

    def test_no_negative_sales(self):
        """Продажи не могут быть отрицательными."""
        df = generate_sales_data(n_days=30)
        assert (df["sales_units"] >= 0).all()

    def test_required_columns(self):
        """Все нужные колонки присутствуют."""
        df = generate_sales_data(n_days=10)
        required = {"date","product","our_price","competitor_price","sales_units","revenue","profit"}
        assert required.issubset(set(df.columns))


class TestRuleBasedPricing:
    """Проверяем бизнес-правила."""

    @pytest.fixture
    def sample_row(self):
        """Базовая строка данных для тестов."""
        return {
            "our_price": 100.0,
            "competitor_price": 100.0,
            "sales_units": 50,
            "last_day_sales": 50,
        }

    def test_competitor_much_cheaper_lowers_price(self):
        """Если конкурент на >10% дешевле → снижаем цену."""
        df = pd.DataFrame([{"our_price": 100, "competitor_price": 85, "sales_units": 20}])
        result = apply_rule_based_pricing(df)
        assert result["rule_based_price"].iloc[0] < 100

    def test_zero_sales_lowers_price(self):
        """При нулевых продажах снижаем на 1 рубль."""
        df = pd.DataFrame([{"our_price": 100, "competitor_price": 100, "sales_units": 0}])
        result = apply_rule_based_pricing(df)
        assert result["rule_based_price"].iloc[0] == 99.0

    def test_we_much_cheaper_raises_price(self):
        """Если мы значительно дешевле → поднимаем цену (не оставляем деньги на столе)."""
        df = pd.DataFrame([{"our_price": 70, "competitor_price": 100, "sales_units": 80}])
        result = apply_rule_based_pricing(df)
        assert result["rule_based_price"].iloc[0] > 70


class TestOptimization:
    """Проверяем математическую оптимизацию."""

    def test_optimal_price_above_cost(self):
        """Оптимальная цена всегда выше себестоимости."""
        opt = find_optimal_price(A=200, B=1.5, cost=50, min_price=52, max_price=200)
        assert opt["optimal_price"] >= 52

    def test_profit_positive(self):
        """Прибыль при оптимальной цене должна быть положительной."""
        opt = find_optimal_price(A=200, B=1.5, cost=50, min_price=52, max_price=200)
        assert opt["expected_profit"] > 0

    def test_demand_model_r2(self):
        """R² модели спроса должен быть разумным (>0.1) на синтетических данных."""
        df = generate_sales_data(n_days=90)
        product = "Молоко 1л"
        A, B, r2 = estimate_demand_model(df, product)
        assert r2 > 0.1, f"R²={r2:.3f} слишком мал — модель не обучилась"
        assert B > 0, "Коэффициент B должен быть положительным"
