-- =======================================================================
-- Proyek Data Science: Prediksi PM2.5 & Estimasi AQI Menggunakan XGBoost
-- File 01: DDL Skema 3 Tabel Staging (Raw Data Layer)
-- =======================================================================

USE aqi_prediction_db;

-- 1. Tabel Staging: raw_air_quality (Data Sensor Polutan Partikulat)
CREATE TABLE IF NOT EXISTS raw_air_quality (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    recorded_at DATETIME NOT NULL,
    location_id VARCHAR(50) NOT NULL,
    location_name VARCHAR(100) DEFAULT 'Pusat Kota',
    pm25_raw DECIMAL(8, 2) NULL COMMENT 'Konsentrasi PM2.5 mentah (ug/m3)',
    pm10_raw DECIMAL(8, 2) NULL COMMENT 'Konsentrasi PM10 mentah (ug/m3)',
    source_api VARCHAR(50) DEFAULT 'OpenAQ',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_air_recorded_at (recorded_at),
    INDEX idx_air_location_id (location_id),
    UNIQUE KEY uk_air_time_loc (recorded_at, location_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. Tabel Staging: raw_weather (Data Sensor Meteorologi)
CREATE TABLE IF NOT EXISTS raw_weather (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    recorded_at DATETIME NOT NULL,
    location_id VARCHAR(50) NOT NULL,
    temperature_c DECIMAL(5, 2) NULL COMMENT 'Suhu udara (Celcius)',
    humidity_pct DECIMAL(5, 2) NULL COMMENT 'Kelembapan relatif (%)',
    wind_speed_kmh DECIMAL(6, 2) NULL COMMENT 'Kecepatan angin (km/jam)',
    wind_direction_deg DECIMAL(5, 1) NULL COMMENT 'Arah angin (derajat 0-360)',
    rainfall_mm DECIMAL(6, 2) NULL COMMENT 'Curah hujan akumulasi (mm)',
    source_api VARCHAR(50) DEFAULT 'Open-Meteo',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_weather_recorded_at (recorded_at),
    INDEX idx_weather_location_id (location_id),
    UNIQUE KEY uk_weather_time_loc (recorded_at, location_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3. Tabel Staging: raw_traffic_calendar (Data Indeks Mobilitas & Kalender)
CREATE TABLE IF NOT EXISTS raw_traffic_calendar (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    recorded_at DATETIME NOT NULL,
    location_id VARCHAR(50) NOT NULL,
    traffic_index DECIMAL(5, 2) NULL COMMENT 'Indeks kemacetan lalu lintas (skala 0 - 100)',
    is_weekend TINYINT(1) DEFAULT 0 COMMENT 'Flag akhir pekan: 1 = Akhir Pekan, 0 = Hari Kerja',
    is_holiday TINYINT(1) DEFAULT 0 COMMENT 'Flag libur nasional: 1 = Hari Libur, 0 = Hari Normal',
    source_api VARCHAR(50) DEFAULT 'TomTom/Scraper',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_traffic_recorded_at (recorded_at),
    INDEX idx_traffic_location_id (location_id),
    UNIQUE KEY uk_traffic_time_loc (recorded_at, location_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
