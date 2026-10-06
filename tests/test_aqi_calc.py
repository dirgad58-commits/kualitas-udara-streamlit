"""
Unit Test untuk Verifikasi Rumus Matematis US EPA AQI
"""

import unittest
import sys
import os

# Menambahkan root project ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.aqi_calculator import calculate_pm25_aqi

class TestAQICalculator(unittest.TestCase):
    
    def test_good_category(self):
        # 4.5 ug/m3 berada di tengah interval 0.0 - 9.0 (AQI 0 - 50)
        res = calculate_pm25_aqi(4.5)
        self.assertEqual(res["category_en"], "Good")
        self.assertTrue(0 <= res["aqi"] <= 50)
        self.assertEqual(res["color"], "#00E400")
        
    def test_moderate_category(self):
        # 20.0 ug/m3 berada di rentang 9.1 - 35.4 (AQI 51 - 100)
        res = calculate_pm25_aqi(20.0)
        self.assertEqual(res["category_en"], "Moderate")
        self.assertTrue(51 <= res["aqi"] <= 100)
        self.assertEqual(res["color"], "#FFFF00")
        
    def test_unhealthy_for_sensitive_groups(self):
        # 40.0 ug/m3 berada di rentang 35.5 - 55.4 (AQI 101 - 150)
        res = calculate_pm25_aqi(40.0)
        self.assertEqual(res["category_en"], "Unhealthy for Sensitive Groups")
        self.assertTrue(101 <= res["aqi"] <= 150)
        self.assertEqual(res["color"], "#FF7E00")
        
    def test_unhealthy_category(self):
        # 80.0 ug/m3 berada di rentang 55.5 - 125.4 (AQI 151 - 200)
        res = calculate_pm25_aqi(80.0)
        self.assertEqual(res["category_en"], "Unhealthy")
        self.assertTrue(151 <= res["aqi"] <= 200)
        self.assertEqual(res["color"], "#FF0000")
        
    def test_edge_zero(self):
        res = calculate_pm25_aqi(0.0)
        self.assertEqual(res["aqi"], 0)
        self.assertEqual(res["category_en"], "Good")

if __name__ == "__main__":
    unittest.main()
