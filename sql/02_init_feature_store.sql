-- =======================================================================
-- Proyek Data Science: Prediksi PM2.5 & Estimasi AQI Menggunakan XGBoost
-- File 02: DDL Skema Master Feature Store
-- =======================================================================

USE aqi_prediction_db;

CREATE TABLE IF NOT EXISTS master_feature_store (
    feature_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    recorded_at DATETIME NOT NULL,
    location_id VARCHAR(50) NOT NULL,
    
    -- Target Variabel Fisik (PM2.5) & Polutan Pendukung
    pm25 DECIMAL(8, 2) NOT NULL COMMENT 'Target kontinu regresi (ug/m3)',
    pm10 DECIMAL(8, 2) NULL,
    
    -- Fitur Meteorologi
    temperature_c DECIMAL(5, 2) NOT NULL,
    humidity_pct DECIMAL(5, 2) NOT NULL,
    wind_speed_kmh DECIMAL(6, 2) NOT NULL,
    wind_direction_deg DECIMAL(5, 1) NULL,
    rainfall_mm DECIMAL(6, 2) NOT NULL DEFAULT 0.00,
    
    -- Fitur Aktivitas & Mobilitas Antropogenik
    traffic_index DECIMAL(5, 2) NOT NULL DEFAULT 0.00,
    is_weekend TINYINT(1) NOT NULL DEFAULT 0,
    is_holiday TINYINT(1) NOT NULL DEFAULT 0,
    
    -- Fitur Rekayasa Temporal & Lag (Time-Series Features)
    pm25_lag_1h DECIMAL(8, 2) NULL COMMENT 'Konsentrasi PM2.5 1 jam sebelumnya',
    pm25_lag_24h DECIMAL(8, 2) NULL COMMENT 'Konsentrasi PM2.5 24 jam sebelumnya (siklus harian)',
    pm25_rolling_mean_6h DECIMAL(8, 2) NULL COMMENT 'Rata-rata bergerak PM2.5 6 jam terakhir',
    hour_of_day TINYINT NOT NULL COMMENT 'Jam pengambilan data (0 - 23)',
    day_of_week TINYINT NOT NULL COMMENT 'Hari dalam minggu (0 = Senin, 6 = Minggu)',
    
    -- Metadata Pipeline
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    UNIQUE KEY uk_mfs_time_loc (recorded_at, location_id),
    INDEX idx_mfs_recorded_at (recorded_at),
    INDEX idx_mfs_location (location_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
