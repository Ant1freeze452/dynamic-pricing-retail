"""
=============================================================================
STREAMLIT-ПРИЛОЖЕНИЕ: ДАШБОРД ДИНАМИЧЕСКОГО ЦЕНООБРАЗОВАНИЯ
=============================================================================
Запуск: streamlit run app.py

Что делает приложение:
  1. Генерирует (или загружает) синтетические данные продаж
  2. Показывает EDA: графики зависимости спроса от цены
  3. Запускает алгоритм и отображает рекомендации по ценам
  4. Позволяет симулировать «перемотку времени вперёд» — что будет,
     если применить новые цены на следующие N дней
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Добавляем корень проекта в путь — чтобы импортировать наши модули
sys.path.insert(0, os.path.dirname(__file__))

from src.data_generator import generate_sales_data, PRODUCTS
from src.pricing_engine  import (apply_rule_based_pricing,
                                  generate_pricing_report,
                                  estimate_demand_model,
                                  find_optimal_price)

# ──────────────────────────────────────────────────
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ──────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Динамическое ценообразование",
    page_icon   = "💰",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

st.title("Алгоритм динамического ценообразования")
st.caption("Образовательный MVP | Направление: Прикладной ИИ | 2-й семестр")

# ──────────────────────────────────────────────────
# САЙДБАР: ПАРАМЕТРЫ
# ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Параметры")

    n_days = st.slider(
        "Дней истории",
        min_value = 30,
        max_value = 180,
        value     = 90,
        step      = 10,
        help      = "Сколько дней исторических данных генерировать"
    )

    selected_product = st.selectbox(
        "Товар для анализа",
        options = list(PRODUCTS.keys()),
        help    = "Выберите товар для детального разбора"
    )

    sim_days = st.slider(
        "Симуляция: дней вперёд",
        min_value = 7,
        max_value = 30,
        value     = 14,
        help      = "Симулируем применение оптимальных цен на N дней"
    )

    st.divider()
    st.info("Алгоритм находит цену p*, максимизирующую прибыль:\n\n"
            "p* = (A + B·cost) / 2B\n\n"
            "где Q(p) = A − B·p — линейная модель спроса")

# ──────────────────────────────────────────────────
# ЗАГРУЗКА / ГЕНЕРАЦИЯ ДАННЫХ
# ──────────────────────────────────────────────────

@st.cache_data(show_spinner="Генерируем данные...")
def load_data(n: int) -> pd.DataFrame:
    """Кэшируем данные, чтобы не перегенерировать при каждом взаимодействии."""
    return generate_sales_data(n_days=n)

df = load_data(n_days)

# Применяем rule-based корректировки
df_ruled = apply_rule_based_pricing(df)

# ──────────────────────────────────────────────────
# ВКЛАДКИ ПРИЛОЖЕНИЯ
# ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["Данные и EDA", "Рекомендации", "Анализ товара", "Симуляция"])

# ════════════════════════════════════════════════
# TAB 1: ДАННЫЕ И EDA
# ════════════════════════════════════════════════
with tab1:
    st.subheader("Обзор датасета")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Строк в датасете",  f"{len(df):,}")
    c2.metric("Товаров",           df["product"].nunique())
    c3.metric("Период (дней)",     n_days)
    c4.metric("Суммарная выручка", f"{df['revenue'].sum():,.0f} ₽")

    st.dataframe(df.tail(20), use_container_width=True)

    st.subheader("Эластичность спроса: цена → продажи")

    # Scatter: для каждого товара — точки (цена, продажи)
    fig = px.scatter(
        df,
        x         = "our_price",
        y         = "sales_units",
        color     = "product",
        trendline = "ols",   # линия тренда через OLS
        title     = "Зависимость продаж от цены (каждая точка = 1 день × 1 товар)",
        labels    = {"our_price": "Наша цена (руб)", "sales_units": "Продажи (шт)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Гистограмма выручки по дням недели
    df["weekday"] = pd.to_datetime(df["date"]).dt.day_name()
    weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    weekday_ru    = {"Monday":"Пн","Tuesday":"Вт","Wednesday":"Ср",
                     "Thursday":"Чт","Friday":"Пт","Saturday":"Сб","Sunday":"Вс"}
    df["weekday_ru"] = df["weekday"].map(weekday_ru)

    fig2 = px.bar(
        df.groupby("weekday_ru")["revenue"].mean().reset_index(),
        x     = "weekday_ru",
        y     = "revenue",
        title = "Средняя дневная выручка по дням недели",
        labels = {"weekday_ru": "День недели", "revenue": "Средняя выручка (руб)"},
        color = "revenue",
        color_continuous_scale = "teal",
    )
    st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════
# TAB 2: РЕКОМЕНДАЦИИ
# ════════════════════════════════════════════════
with tab2:
    st.subheader("Оптимальные цены: итоговый отчёт")

    report = generate_pricing_report(df)

    # Цветовая подсветка: рост прибыли → зелёный, снижение → красный
    def color_delta(val):
        color = "green" if val > 0 else ("red" if val < 0 else "black")
        return f"color: {color}"

    styled = report.style.map(
        color_delta,
        subset=["Δ Выручка (руб)", "Δ Прибыль (руб)", "Δ Цена (%)"]
    )
    st.dataframe(styled, use_container_width=True)

    # Визуализация: старая vs новая цена
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name = "Текущая цена",
        x    = report["Товар"],
        y    = report["Текущая цена (руб)"],
        marker_color = "steelblue",
    ))
    fig3.add_trace(go.Bar(
        name = "Оптимальная цена",
        x    = report["Товар"],
        y    = report["Оптимальная цена (руб)"],
        marker_color = "darkorange",
    ))
    fig3.update_layout(
        barmode = "group",
        title   = "Текущая цена vs Оптимальная цена",
        xaxis_title = "Товар",
        yaxis_title = "Цена (руб)",
    )
    st.plotly_chart(fig3, use_container_width=True)

    total_delta_profit = report["Δ Прибыль (руб)"].sum()
    st.success(f"Применение оптимальных цен даёт прирост прибыли: "
               f"**{total_delta_profit:+,.0f} ₽/день** суммарно по всем товарам")


# ════════════════════════════════════════════════
# TAB 3: АНАЛИЗ ТОВАРА
# ════════════════════════════════════════════════
with tab3:
    st.subheader(f"📈 Детальный анализ: {selected_product}")

    product_df = df[df["product"] == selected_product].copy()
    params     = PRODUCTS[selected_product]
    A, B, r2   = estimate_demand_model(df, selected_product)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Коэффициент A (базовый спрос)", f"{A:.1f}")
        st.metric("Коэффициент B (ценовая чувствительность)", f"{B:.3f}")
        st.metric("R² модели спроса", f"{r2:.3f}", help="Чем ближе к 1 — тем лучше модель")

    with col2:
        opt = find_optimal_price(
            A         = A,
            B         = B,
            cost      = params["cost"],
            min_price = params["cost"] * 1.05,
            max_price = product_df["our_price"].max() * 1.3,
        )
        st.metric("Оптимальная цена", f"{opt['optimal_price']:.0f} ₽")
        st.metric("Прогнозируемый спрос", f"{opt['expected_demand']:.0f} шт/день")
        st.metric("Прогнозируемая прибыль", f"{opt['expected_profit']:.0f} ₽/день")

    # График: кривая прибыли в зависимости от цены
    prices       = np.linspace(params["cost"] * 1.01, params["base_price"] * 1.5, 200)
    demands      = np.maximum(0, A - B * prices)
    profits      = (prices - params["cost"]) * demands
    revenues     = prices * demands

    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4.add_trace(go.Scatter(x=prices, y=profits,  name="Прибыль (руб)",  line=dict(color="green")))
    fig4.add_trace(go.Scatter(x=prices, y=revenues, name="Выручка (руб)", line=dict(color="blue", dash="dot")), secondary_y=True)
    fig4.add_vline(x=opt["optimal_price"], line_color="red", line_dash="dash",
                   annotation_text=f"p* = {opt['optimal_price']:.0f} ₽")
    fig4.update_layout(title="Кривые выручки и прибыли в зависимости от цены")
    fig4.update_xaxes(title_text="Цена (руб)")
    fig4.update_yaxes(title_text="Прибыль (руб)", secondary_y=False)
    fig4.update_yaxes(title_text="Выручка (руб)", secondary_y=True)
    st.plotly_chart(fig4, use_container_width=True)

    # Временной ряд: продажи и цена
    fig5 = make_subplots(specs=[[{"secondary_y": True}]])
    fig5.add_trace(go.Scatter(
        x=product_df["date"], y=product_df["sales_units"],
        name="Продажи (шт)", fill="tozeroy", line=dict(color="teal")
    ))
    fig5.add_trace(go.Scatter(
        x=product_df["date"], y=product_df["our_price"],
        name="Наша цена (руб)", line=dict(color="orange")
    ), secondary_y=True)
    fig5.add_trace(go.Scatter(
        x=product_df["date"], y=product_df["competitor_price"],
        name="Цена конкурента (руб)", line=dict(color="red", dash="dot")
    ), secondary_y=True)
    fig5.update_layout(title="История продаж и цен")
    st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════
# TAB 4: СИМУЛЯЦИЯ
# ════════════════════════════════════════════════
with tab4:
    st.subheader("🕹️ Симуляция: что будет, если применить оптимальные цены?")
    st.info("Алгоритм применяет рекомендованные цены → генерирует новый спрос → "
            "пересчитывает цены → и так на каждый день симуляции.")

    if st.button("Запустить симуляцию", type="primary"):
        with st.spinner("Симулируем..."):
            # Берём последний день как стартовую точку
            simulation_rows = []

            for product, params in PRODUCTS.items():
                pdata        = df[df["product"] == product]
                A, B, r2     = estimate_demand_model(df, product)
                current_price = pdata["our_price"].iloc[-1]

                for day in range(sim_days):
                    comp_price = current_price * np.random.uniform(0.88, 1.05)

                    # Оптимизируем цену
                    opt = find_optimal_price(
                        A, B, params["cost"],
                        min_price = params["cost"] * 1.05,
                        max_price = params["base_price"] * 1.5,
                    )
                    new_price = opt["optimal_price"]

                    # Симулируем спрос при новой цене
                    demand   = max(0, A - B * new_price + np.random.normal(0, 3))
                    revenue  = new_price * demand
                    profit   = (new_price - params["cost"]) * demand

                    simulation_rows.append({
                        "День":     day + 1,
                        "Товар":    product,
                        "Цена":     new_price,
                        "Продажи":  round(demand, 0),
                        "Выручка":  round(revenue, 2),
                        "Прибыль":  round(profit, 2),
                    })

                    current_price = new_price  # на следующий день стартуем с новой

            sim_df = pd.DataFrame(simulation_rows)

            # Суммарные метрики симуляции
            total_rev    = sim_df["Выручка"].sum()
            total_profit = sim_df["Прибыль"].sum()

            m1, m2 = st.columns(2)
            m1.metric("Суммарная выручка за симуляцию", f"{total_rev:,.0f} ₽")
            m2.metric("Суммарная прибыль за симуляцию",  f"{total_profit:,.0f} ₽")

            # График выручки по дням
            fig6 = px.line(
                sim_df.groupby("День")["Выручка"].sum().reset_index(),
                x     = "День",
                y     = "Выручка",
                title = "Суммарная выручка по дням симуляции (все товары)",
                markers = True,
            )
            st.plotly_chart(fig6, use_container_width=True)

            st.dataframe(sim_df, use_container_width=True)
