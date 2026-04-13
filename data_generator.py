import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_data(num_days=100, num_products=5, start_date="2024-01-01", seed=42):
    np.random.seed(seed)
    dates = [datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=i) for i in range(num_days)]

    products = []
    for i in range(1, num_products + 1):
        products.append({
            "product_id": i,
            "product_name": f"Товар_{i}",
            "base_price": np.random.uniform(500, 2000),
            "elasticity": np.random.uniform(0.3, 0.8),
            "cross_elasticity": np.random.uniform(0.5, 1.5),
            "base_demand": np.random.uniform(500, 1500)
        })

    data = []
    for product in products:
        for date in dates:
            our_price = product["base_price"] + np.random.normal(0, 20)
            our_price = max(our_price, 10)

            competitor_price = our_price * np.random.uniform(0.85, 1.15)

            demand = (product["base_demand"] - product["elasticity"] * our_price + product["cross_elasticity"] * (competitor_price - our_price)
                      + np.random.normal(0, 5))

            demand = max(0, int(round(demand)))

            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "product_id": product["product_id"],
                "product_name": product["product_name"],
                "our_price": round(our_price, 2),
                "competitor_price": round(competitor_price, 2),
                "sales_qty": demand
            })

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("data/synthetic_sales.csv", index=False)
    print("Данные сгенерированы и сохранены в data/synthetic_sales.csv")
    print(df.head())
