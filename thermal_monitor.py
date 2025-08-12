# thermal_monitor.py - Main application
import time
import json
import csv
from datetime import datetime
import logging

# Configuration
class ThermalConfig:
    REFRESH_RATE = 2.0  # Hz
    LOG_INTERVAL = 900  # seconds (15 minutes)
    DATA_FILE = "thermal_data.csv"
    LOG_FILE = "thermal_monitor.log"
    
    # Temperature thresholds
    TEMP_WARNING = 60.0   # °C
    TEMP_CRITICAL = 80.0  # °C

class ThermalMonitor:
    def __init__(self, mock_mode=True):
        self.mock_mode = mock_mode
        self.setup_logging()
        
        if mock_mode:
            from mock_mlx90640 import MockMLX90640
            self.sensor = MockMLX90640()
        else:
            # Real sensor initialization (for when Pi is available)
            import board
            import busio
            import adafruit_mlx90640
            i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
            self.sensor = adafruit_mlx90640.MLX90640(i2c)
            self.sensor.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(ThermalConfig.LOG_FILE),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def read_thermal_frame(self):
        """Read and process thermal camera frame"""
        try:
            if self.mock_mode:
                frame = self.sensor.getFrame()
            else:
                frame = [0] * 768  # Initialize array
                self.sensor.getFrame(frame)
            
            # Calculate statistics
            temps = {
                'timestamp': datetime.now().isoformat(),
                'min_temp': min(frame),
                'max_temp': max(frame),
                'avg_temp': sum(frame) / len(frame),
                'frame_data': frame if self.should_log_full_frame() else None
            }
            
            return temps
            
        except Exception as e:
            self.logger.error(f"Error reading thermal frame: {e}")
            return None
    
    def should_log_full_frame(self):
        """Decide whether to log full frame data"""
        # Only log full frame if temperature exceeds threshold
        return False  # Implement logic based on requirements
    
    def check_temperature_alerts(self, temps):
        """Check for temperature threshold violations"""
        max_temp = temps['max_temp']
        
        if max_temp > ThermalConfig.TEMP_CRITICAL:
            self.logger.critical(f"CRITICAL TEMPERATURE: {max_temp:.1f}°C")
            return "CRITICAL"
        elif max_temp > ThermalConfig.TEMP_WARNING:
            self.logger.warning(f"Warning temperature: {max_temp:.1f}°C")
            return "WARNING"
        
        return "NORMAL"
    
    def log_data(self, temps):
        """Log temperature data to CSV file"""
        try:
            file_exists = False
            try:
                with open(ThermalConfig.DATA_FILE, 'r'):
                    file_exists = True
            except FileNotFoundError:
                pass
            
            with open(ThermalConfig.DATA_FILE, 'a', newline='') as csvfile:
                fieldnames = ['timestamp', 'min_temp', 'max_temp', 'avg_temp', 'status']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                status = self.check_temperature_alerts(temps)
                writer.writerow({
                    'timestamp': temps['timestamp'],
                    'min_temp': f"{temps['min_temp']:.2f}",
                    'max_temp': f"{temps['max_temp']:.2f}",
                    'avg_temp': f"{temps['avg_temp']:.2f}",
                    'status': status
                })
                
        except Exception as e:
            self.logger.error(f"Error logging data: {e}")
    
    def run_monitoring(self, duration_minutes=None):
        """Main monitoring loop"""
        self.logger.info("Starting thermal monitoring...")
        
        last_log_time = 0
        start_time = time.time()
        
        try:
            while True:
                current_time = time.time()
                
                # Read thermal data
                temps = self.read_thermal_frame()
                if temps:
                    # Log to file at specified interval
                    if current_time - last_log_time >= ThermalConfig.LOG_INTERVAL:
                        self.log_data(temps)
                        last_log_time = current_time
                        self.logger.info(f"Temperature range: {temps['min_temp']:.1f}°C to {temps['max_temp']:.1f}°C")
                    
                    # Check for alerts every reading
                    self.check_temperature_alerts(temps)
                
                # Exit condition for testing
                if duration_minutes and (current_time - start_time) > (duration_minutes * 60):
                    break
                
                # Sleep based on refresh rate
                time.sleep(1.0 / ThermalConfig.REFRESH_RATE)
                
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")

# Test the system
if __name__ == "__main__":
    monitor = ThermalMonitor(mock_mode=True)
    monitor.run_monitoring(duration_minutes=5)  # 5-minute test run