-- =======================================================================
-- Proyek Data Science: Prediksi PM2.5 & Estimasi AQI Menggunakan XGBoost
-- File 03: DDL Skema Tabel Prediction Logs (Audit Trail & Serving)
-- =======================================================================

USE aqi_prediction_db;

CREATE TABLE IF NOT EXISTS prediction_logs (
    prediction_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    location_id VARCHAR(50) NOT NULL DEFAULT 'LOC-JKT-01',
    model_version VARCHAR(50) NOT NULL DEFAULT 'v1.0.0-xgb',
    
    -- Nilai Hasil Prediksi Model & Konversi Standar EPA
    predicted_pm25 DECIMAL(8, 2) NOT NULL COMMENT 'Hasil regresi kontinu PM2.5 (ug/m3)',
    actual_pm25 DECIMAL(8, 2) NULL COMMENT 'Ground truth riil jika sudah tersedia',
    calculated_aqi INT NOT NULL COMMENT 'Skor AQI standar US EPA (0 - 500)',
    aqi_category VARCHAR(50) NOT NULL COMMENT 'Kategori risiko (Baik, Sedang, Tidak Sehat, dll)',
    health_implication TEXT NOT NULL COMMENT 'Rekomendasi mitigasi kesehatan publik',
    
    -- Snapshot Input Fitur (JSON) untuk Reproduksibilitas & Audit Drift
    input_features_json JSON NOT NULL COMMENT 'Snapshot payload nilai fitur masukan model',
    
    -- Error Tracking (diisi saat actual_pm25 tersedia)
    absolute_error DECIMAL(8, 2) GENERATED ALWAYS AS (
        CASE WHEN actual_pm25 IS NOT NULL THEN ABS(actual_pm25 - predicted_pm25) ELSE NULL END
    ) STORED,
    
    INDEX idx_pred_time (predicted_at),
    INDEX idx_pred_location (location_id),
    INDEX idx_pred_model (model_version),
    INDEX idx_pred_category (aqi_category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
