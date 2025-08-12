import unittest
from thermal_monitor import ThermalMonitor, ThermalConfig
import os

class TestThermalSystem(unittest.TestCase):
    def setUp(self):
        # Always use mock mode for tests to avoid requiring hardware
        self.monitor = ThermalMonitor(mock_mode=True)

    def test_sensor_reading(self):
        """Test basic sensor reading functionality"""
        temps = self.monitor.read_thermal_frame()
        self.assertIsInstance(temps, dict)
        self.assertIn('min_temp', temps)
        self.assertIn('max_temp', temps)
        self.assertIn('avg_temp', temps)
        self.assertIsInstance(temps['min_temp'], (int, float))
        self.assertIsInstance(temps['max_temp'], (int, float))
        self.assertIsInstance(temps['avg_temp'], (int, float))

    def test_data_logging(self):
        """Test CSV data logging"""
        test_file = "test_thermal_data.csv"
        if os.path.exists(test_file):
            os.remove(test_file)

        # Temporarily change data file
        original_file = ThermalConfig.DATA_FILE
        ThermalConfig.DATA_FILE = test_file

        temps = self.monitor.read_thermal_frame()
        self.monitor.log_data(temps)

        # Verify file creation
        self.assertTrue(os.path.exists(test_file))

        # Cleanup
        ThermalConfig.DATA_FILE = original_file
        if os.path.exists(test_file):
            os.remove(test_file)

    def test_temperature_alerts(self):
        """Test temperature alert system"""
        # WARNING threshold
        temps = {'max_temp': ThermalConfig.WARNING_THRESHOLD + 1,
                 'min_temp': 20.0, 'avg_temp': 40.0}
        status = self.monitor.check_temperature_alerts(temps)
        self.assertEqual(status, "WARNING")

        # CRITICAL threshold
        temps = {'max_temp': ThermalConfig.CRITICAL_THRESHOLD + 1,
                 'min_temp': 20.0, 'avg_temp': 50.0}
        status = self.monitor.check_temperature_alerts(temps)
        self.assertEqual(status, "CRITICAL")

if __name__ == '__main__':
    unittest.main()
