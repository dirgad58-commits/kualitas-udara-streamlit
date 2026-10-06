"""
Script Generator Dataset Sintetis Realistis Kualitas Udara (PM2.5)
Menghasilkan 720 baris data per jam (30 hari) dengan pola temporal, cuaca, dan lalu lintas
yang realistis untuk pengujian pipeline di Google Colab dan MySQL.
"""

import os
import math
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def generate_environmental_dataset(output_dir="data", days=30, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    start_date = datetime(2026, 9, 1, 0, 0, 0)
    total_hours = days * 24
    
    records = []
    location_id = "LOC-JKT-01"
    
    # Baseline konsentrasi PM2.5 awal
    current_pm25 = 35.0
    
    for i in range(total_hours):
        current_time = start_date + timedelta(hours=i)
        hour = current_time.hour
        day_of_week = current_time.weekday() # 0 = Senin, 6 = Minggu
        is_weekend = 1 if day_of_week in [5, 6] else 0
        is_holiday = 1 if (day_of_week == 6 and i % 168 == 0) else 0
        
        # 1. Siklus Suhu & Kelembaban (Diurnal Cycle)
        # Suhu memuncak di jam 13-14 (32-34C), minimum di jam 05 (24-25C)
        temp_cycle = math.sin((hour - 8) / 24 * 2 * math.pi)
        temperature = 28.5 + 4.5 * temp_cycle + np.random.normal(0, 0.8)
        temperature = round(float(temperature), 2)
        
        # Kelembaban berbanding terbalik dengan suhu (Tinggi dini hari, rendah siang)
        humidity = 82.0 - 22.0 * temp_cycle + np.random.normal(0, 3.0)
        humidity = round(float(np.clip(humidity, 45.0, 98.0)), 2)
        
        # 2. Kecepatan & Arah Angin
        wind_speed = 8.0 + 5.0 * max(0, temp_cycle) + np.random.exponential(3.0)
        wind_speed = round(float(np.clip(wind_speed, 1.5, 35.0)), 2)
        wind_direction = round(float(np.random.uniform(0, 360)), 1)
        
        # 3. Curah Hujan (Sebagian besar 0, sesekali hujan sore hari)
        rain_prob = 0.15 if 14 <= hour <= 19 else 0.05
        if np.random.rand() < rain_prob:
            rainfall = round(float(np.random.exponential(6.0)), 2)
        else:
            rainfall = 0.0
            
        # 4. Indeks Lalu Lintas (Traffic Congestion Index: 0 - 100)
        # Puncak jam sibuk pagi (07:00 - 09:00) dan sore (17:00 - 19:00)
        if is_weekend:
            # Weekend trafik lebih merata siang hari
            base_traffic = 35.0 + 25.0 * math.sin((hour - 6) / 18 * math.pi) if 6 <= hour <= 23 else 15.0
        else:
            if 7 <= hour <= 9:
                base_traffic = 82.0 + np.random.normal(0, 5.0)
            elif 17 <= hour <= 19:
                base_traffic = 88.0 + np.random.normal(0, 4.0)
            elif 10 <= hour <= 16:
                base_traffic = 55.0 + np.random.normal(0, 6.0)
            elif 20 <= hour <= 23:
                base_traffic = 40.0 + np.random.normal(0, 5.0)
            else:
                base_traffic = 15.0 + np.random.normal(0, 4.0)
        traffic_index = round(float(np.clip(base_traffic, 5.0, 98.0)), 2)
        
        # 5. Emisi & Dinamika Konsentrasi PM2.5 (ug/m3)
        # - Meningkat saat traffic tinggi
        # - Meningkat saat kelembaban tinggi (partikel higroskopis & pembentukan kabut)
        # - Menurun saat angin kencang (dispersi atmosfer)
        # - Menurun drastis saat hujan (wet deposition / washout effect)
        traffic_contrib = (traffic_index / 100.0) * 35.0
        weather_retention = (humidity / 100.0) * 12.0
        wind_dispersion = (wind_speed / 20.0) * 15.0
        rain_washout = min(rainfall * 4.0, 30.0)
        
        target_pm25 = 20.0 + traffic_contrib + weather_retention - wind_dispersion - rain_washout + np.random.normal(0, 3.5)
        target_pm25 = np.clip(target_pm25, 4.0, 160.0)
        
        # Smooth transition (autoregressive nature of air quality)
        current_pm25 = 0.65 * current_pm25 + 0.35 * target_pm25
        pm25 = round(float(current_pm25), 2)
        pm10 = round(float(pm25 * 1.55 + np.random.normal(0, 3.0)), 2)
        
        records.append({
            "recorded_at": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "location_id": location_id,
            "pm25": pm25,
            "pm10": pm10,
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "wind_speed_kmh": wind_speed,
            "wind_direction_deg": wind_direction,
            "rainfall_mm": rainfall,
            "traffic_index": traffic_index,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "hour_of_day": hour,
            "day_of_week": day_of_week
        })
        
    df = pd.DataFrame(records)
    
    # Hitung Lag dan Rolling Features
    df["pm25_lag_1h"] = df["pm25"].shift(1)
    df["pm25_lag_24h"] = df["pm25"].shift(24)
    df["pm25_rolling_mean_6h"] = df["pm25"].rolling(window=6, min_periods=1).mean().round(2)
    
    # Backfill nilai awal agar tidak ada NaN
    df["pm25_lag_1h"] = df["pm25_lag_1h"].bfill()
    df["pm25_lag_24h"] = df["pm25_lag_24h"].bfill()
    
    # Simpan dataset lengkap (Master Feature Store)
    mfs_path = os.path.join(output_dir, "sample_dataset.csv")
    df.to_csv(mfs_path, index=False)
    print(f" Berhasil membuat dataset Master Feature Store: {mfs_path} ({len(df)} baris)")
    
    # Simpan juga 3 tabel staging mentah secara terpisah
    df_air = df[["recorded_at", "location_id", "pm25", "pm10"]].copy()
    df_air.rename(columns={"pm25": "pm25_raw", "pm10": "pm10_raw"}, inplace=True)
    df_air["location_name"] = "Stasiun Pusat Kota"
    df_air["source_api"] = "OpenAQ"
    # Introduksi sedikit missing values untuk simulasi Data Assessing & Cleaning
    df_air.loc[np.random.choice(df_air.index, 8, replace=False), "pm25_raw"] = np.nan
    df_air.to_csv(os.path.join(output_dir, "raw_air_quality.csv"), index=False)
    
    df_weather = df[["recorded_at", "location_id", "temperature_c", "humidity_pct", "wind_speed_kmh", "wind_direction_deg", "rainfall_mm"]].copy()
    df_weather["source_api"] = "Open-Meteo"
    df_weather.loc[np.random.choice(df_weather.index, 5, replace=False), "temperature_c"] = np.nan
    df_weather.to_csv(os.path.join(output_dir, "raw_weather.csv"), index=False)
    
    df_traffic = df[["recorded_at", "location_id", "traffic_index", "is_weekend", "is_holiday"]].copy()
    df_traffic["source_api"] = "TomTom/Scraper"
    df_traffic.to_csv(os.path.join(output_dir, "raw_traffic_calendar.csv"), index=False)
    
    print(" Berhasil membuat 3 CSV tabel staging mentah di folder", output_dir)
    return df

if __name__ == "__main__":
    generate_environmental_dataset(output_dir="data", days=30)
