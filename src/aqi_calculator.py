"""
Modul Kalkulator Standar US-EPA AQI (Air Quality Index) untuk PM2.5
Berdasarkan Dokumen Resmi: US EPA Technical Assistance Document for the Reporting of Daily Air Quality (2024 Revision)

Rumus Piecewise Linear Interpolation:
    Ip = [(I_Hi - I_Lo) / (BP_Hi - BP_Lo)] * (Cp - BP_Lo) + I_Lo

Di mana:
    Ip    = Skor Indeks Kualitas Udara (AQI)
    Cp    = Konsentrasi PM2.5 (ug/m3) dibulatkan ke 1 tempat desimal
    BP_Hi = Titik batas konsentrasi atas rentang
    BP_Lo = Titik batas konsentrasi bawah rentang
    I_Hi  = Nilai indeks AQI batas atas
    I_Lo  = Nilai indeks AQI batas bawah
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

# Tabel Breakpoints Standar US EPA PM2.5 (2024 Revised Standard)
EPA_PM25_BREAKPOINTS = [
    {
        "bp_low": 0.0,
        "bp_high": 9.0,
        "i_low": 0,
        "i_high": 50,
        "category": "Baik (Good)",
        "category_en": "Good",
        "color": "#00E400",
        "implication": "Kualitas udara sangat memuaskan, risiko kesehatan minim atau tidak ada sama sekali.",
        "action": "Aman untuk seluruh aktivitas di luar ruangan."
    },
    {
        "bp_low": 9.1,
        "bp_high": 35.4,
        "i_low": 51,
        "i_high": 100,
        "category": "Sedang (Moderate)",
        "category_en": "Moderate",
        "color": "#FFFF00",
        "implication": "Kualitas udara dapat diterima; kelompok yang sangat sensitif mungkin mengalami gejala ringan.",
        "action": "Masyarakat umum dapat beraktivitas normal; kelompok sangat sensitif kurangi aktivitas berat di luar ruangan."
    },
    {
        "bp_low": 35.5,
        "bp_high": 55.4,
        "i_low": 101,
        "i_high": 150,
        "category": "Tidak Sehat bagi Kelompok Sensitif (USG)",
        "category_en": "Unhealthy for Sensitive Groups",
        "color": "#FF7E00",
        "implication": "Anak-anak, lansia, dan penderita asma/penyakit paru berisiko mengalami gangguan pernapasan.",
        "action": "Kelompok sensitif disarankan memakai masker di luar dan menyalakan air purifier di dalam ruang."
    },
    {
        "bp_low": 55.5,
        "bp_high": 125.4,
        "i_low": 151,
        "i_high": 200,
        "category": "Tidak Sehat (Unhealthy)",
        "category_en": "Unhealthy",
        "color": "#FF0000",
        "implication": "Seluruh populasi mulai berpotensi merasakan efek buruk kualitas udara.",
        "action": "Kurangi aktivitas di luar ruang; tutup ventilasi saat jam sibuk; gunakan masker N95 jika harus bepergian."
    },
    {
        "bp_low": 125.5,
        "bp_high": 225.4,
        "i_low": 201,
        "i_high": 300,
        "category": "Sangat Tidak Sehat (Very Unhealthy)",
        "category_en": "Very Unhealthy",
        "color": "#8F3F97",
        "implication": "Peringatan darurat kesehatan: risiko efek serius meningkat pada seluruh populasi.",
        "action": "Hindari seluruh aktivitas di luar ruangan; nyalakan pembersih udara; gunakan masker berstandar tinggi."
    },
    {
        "bp_low": 225.5,
        "bp_high": 500.4,
        "i_low": 301,
        "i_high": 500,
        "category": "Berbahaya (Hazardous)",
        "category_en": "Hazardous",
        "color": "#7E0023",
        "implication": "Kondisi darurat polusi kritis: seluruh populasi terdampak secara akut.",
        "action": "Tetap di dalam ruangan dengan penyaring udara menyala; isolasi jendela dan pintu."
    }
]


def calculate_pm25_aqi(pm25_concentration: float) -> Dict[str, Any]:
    """
    Menghitung skor AQI dan informasi kategori kesehatan berdasarkan nilai konsentrasi PM2.5.
    
    Args:
        pm25_concentration (float): Nilai konsentrasi PM2.5 (ug/m3)
        
    Returns:
        Dict[str, Any]: Dictionary berisi konsentrasi, skor AQI, kategori, warna hex, dan rekomendasi
    """
    if pm25_concentration is None or np.isnan(pm25_concentration):
        return {
            "pm25": None,
            "aqi": None,
            "category": "Data Tidak Tersedia",
            "category_en": "Unavailable",
            "color": "#9E9E9E",
            "implication": "Sensor tidak membaca data valid.",
            "action": "Periksa koneksi sensor."
        }
        
    # Sesuai pedoman EPA, konsentrasi PM2.5 dibulatkan ke 1 tempat desimal
    cp = round(float(pm25_concentration), 1)
    
    if cp < 0:
        cp = 0.0
        
    # Cari breakpoint yang cocok
    for bp in EPA_PM25_BREAKPOINTS:
        if bp["bp_low"] <= cp <= bp["bp_high"]:
            # Rumus Interpolasi Linier EPA
            i_ratio = (bp["i_high"] - bp["i_low"]) / (bp["bp_high"] - bp["bp_low"])
            aqi_val = round(i_ratio * (cp - bp["bp_low"]) + bp["i_low"])
            
            return {
                "pm25": cp,
                "aqi": int(aqi_val),
                "category": bp["category"],
                "category_en": bp["category_en"],
                "color": bp["color"],
                "implication": bp["implication"],
                "action": bp["action"]
            }
            
    # Jika melebihi batas atas tabel (> 500.4 ug/m3)
    return {
        "pm25": cp,
        "aqi": 500,
        "category": "Berbahaya Ekstrem (> 500)",
        "category_en": "Beyond Index (Hazardous)",
        "color": "#7E0023",
        "implication": "Kondisi bencana polusi ekstrem melebihi batas ukur normal indeks EPA.",
        "action": "Evakuasi atau isolasi udara total di ruangan bersekat kedap."
    }


def calculate_aqi_dataframe(df: pd.DataFrame, pm25_col: str = "pm25") -> pd.DataFrame:
    """
    Menambahkan kolom skor AQI dan kategori kesehatan ke DataFrame secara vektorisasi.
    
    Args:
        df (pd.DataFrame): DataFrame yang memuat kolom konsentrasi PM2.5
        pm25_col (str): Nama kolom PM2.5 di DataFrame
        
    Returns:
        pd.DataFrame: DataFrame baru dengan tambahan kolom calculated_aqi, aqi_category, aqi_color
    """
    res_df = df.copy()
    
    def _map_row(val):
        info = calculate_pm25_aqi(val)
        return pd.Series([info["aqi"], info["category"], info["color"]])
        
    res_df[["calculated_aqi", "aqi_category", "aqi_color"]] = res_df[pm25_col].apply(_map_row)
    return res_df


if __name__ == "__main__":
    # Test Contoh Sederhana
    test_cases = [5.0, 15.2, 45.0, 68.4, 155.0, 310.0]
    print("=== PENGUJIAN RUMUS US-EPA AQI (PM2.5) ===")
    for c in test_cases:
        r = calculate_pm25_aqi(c)
        print(f"PM2.5: {r['pm25']:5.1f} ug/m3 -> AQI: {r['aqi']:3d} | Kategori: {r['category']:<38} | Warna: {r['color']}")
