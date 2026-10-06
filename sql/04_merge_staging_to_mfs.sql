-- =======================================================================
-- Proyek Data Science: Prediksi PM2.5 & Estimasi AQI Menggunakan XGBoost
-- File 04: Query Transformasi & Integrasi (LEFT JOIN 3 Staging ke MFS)
-- =======================================================================

USE aqi_prediction_db;

-- Menggabungkan data dari 3 tabel staging mentah ke master_feature_store
-- dengan memanfaatkan MySQL 8.0 Window Functions (LAG, AVG OVER)
INSERT INTO master_feature_store (
    recorded_at,
    location_id,
    pm25,
    pm10,
    temperature_c,
    humidity_pct,
    wind_speed_kmh,
    wind_direction_deg,
    rainfall_mm,
    traffic_index,
    is_weekend,
    is_holiday,
    pm25_lag_1h,
    pm25_lag_24h,
    pm25_rolling_mean_6h,
    hour_of_day,
    day_of_week
)
SELECT 
    base.recorded_at,
    base.location_id,
    base.pm25_raw AS pm25,
    base.pm10_raw AS pm10,
    COALESCE(w.temperature_c, 28.0) AS temperature_c,
    COALESCE(w.humidity_pct, 75.0) AS humidity_pct,
    COALESCE(w.wind_speed_kmh, 10.0) AS wind_speed_kmh,
    COALESCE(w.wind_direction_deg, 180.0) AS wind_direction_deg,
    COALESCE(w.rainfall_mm, 0.0) AS rainfall_mm,
    COALESCE(t.traffic_index, 50.0) AS traffic_index,
    COALESCE(t.is_weekend, CASE WHEN DAYOFWEEK(base.recorded_at) IN (1, 7) THEN 1 ELSE 0 END) AS is_weekend,
    COALESCE(t.is_holiday, 0) AS is_holiday,
    LAG(base.pm25_raw, 1) OVER (PARTITION BY base.location_id ORDER BY base.recorded_at) AS pm25_lag_1h,
    LAG(base.pm25_raw, 24) OVER (PARTITION BY base.location_id ORDER BY base.recorded_at) AS pm25_lag_24h,
    AVG(base.pm25_raw) OVER (
        PARTITION BY base.location_id 
        ORDER BY base.recorded_at 
        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
    ) AS pm25_rolling_mean_6h,
    HOUR(base.recorded_at) AS hour_of_day,
    WEEKDAY(base.recorded_at) AS day_of_week
FROM raw_air_quality base
LEFT JOIN raw_weather w 
    ON base.recorded_at = w.recorded_at 
    AND base.location_id = w.location_id
LEFT JOIN raw_traffic_calendar t 
    ON base.recorded_at = t.recorded_at 
    AND base.location_id = t.location_id
WHERE base.pm25_raw IS NOT NULL
ON DUPLICATE KEY UPDATE
    pm25 = VALUES(pm25),
    pm10 = VALUES(pm10),
    temperature_c = VALUES(temperature_c),
    humidity_pct = VALUES(humidity_pct),
    wind_speed_kmh = VALUES(wind_speed_kmh),
    wind_direction_deg = VALUES(wind_direction_deg),
    rainfall_mm = VALUES(rainfall_mm),
    traffic_index = VALUES(traffic_index),
    is_weekend = VALUES(is_weekend),
    is_holiday = VALUES(is_holiday),
    pm25_lag_1h = VALUES(pm25_lag_1h),
    pm25_lag_24h = VALUES(pm25_lag_24h),
    pm25_rolling_mean_6h = VALUES(pm25_rolling_mean_6h),
    hour_of_day = VALUES(hour_of_day),
    day_of_week = VALUES(day_of_week);
