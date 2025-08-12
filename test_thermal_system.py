# test_thermal_system.py
import unittest
from thermal_monitor import ThermalMonitor, ThermalConfig
import os
import time

class TestThermalSystem(unittest.TestCase):
    def setUp(self):
        self.monitor = ThermalMonitor(mock_mode=True)
    
    def test_sensor_reading(self):
        """Test basic sensor reading functionality"""
        temps = self.monitor.read_thermal_frame()
        self.assertIsNotNone(temps)
        self.assertIn('min_temp', temps)
        self.assertIn('max_temp', temps)
        self.assertIn('avg_temp', temps)
    
    def test_data_logging(self):
        """Test CSV data logging"""
        # Clean up any existing test file
        test_file = "test_thermal_data.csv"
        if os.path.exists(test_file):
            os.remove(test_file)
        
        # Temporarily change data file
        original_file = ThermalConfig.DATA_FILE
        ThermalConfig.DATA_FILE = test_file
        
        # Test logging
        temps = self.monitor.read_thermal_frame()
        self.monitor.log_data(temps)
        
        # Verify file creation
        self.assertTrue(os.path.exists(test_file))
        
        # Cleanup
        ThermalConfig.DATA_FILE = original_file
        os.remove(test_file)
    
    def test_temperature_alerts(self):
        """Test temperature alert system"""
        # Test warning threshold
        temps = {'max_temp': 65.0, 'min_temp': 20.0, 'avg_temp': 40.0}
        status = self.monitor.check_temperature_alerts(temps)
        self.assertEqual(status, "WARNING")
        
        # Test critical threshold
        temps = {'max_temp': 85.0, 'min_temp': 20.0, 'avg_temp': 50.0}
        status = self.monitor.check_temperature_alerts(temps)
        self.assertEqual(status, "CRITICAL")

if __name__ == '__main__':
    unittest.main()