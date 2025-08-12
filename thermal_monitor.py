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
    def __init__(self, mock_mode=False):
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
    monitor.run_monitoring(duration_minutes=5)  # 5-minute test run# thermal_monitor.py - Optimized for real MLX90640 transformer monitoring
import time
import json
import csv
import numpy as np
from datetime import datetime
import logging
import os

# Configuration optimized for transformer monitoring
class ThermalConfig:
    REFRESH_RATE = 1.0  # Hz - Reduced for better stability and accuracy
    LOG_INTERVAL = 300  # seconds (5 minutes) - More frequent for transformer monitoring
    DATA_FILE = "thermal_data.csv"
    LOG_FILE = "thermal_monitor.log"
    
    # Transformer-specific temperature thresholds (based on IEEE standards)
    TEMP_WARNING = 65.0   # °C - Transformer loading concern
    TEMP_CRITICAL = 85.0  # °C - Immediate attention required
    TEMP_EMERGENCY = 105.0 # °C - Emergency shutdown recommended
    
    # Calibration settings
    AMBIENT_OFFSET = 0.0  # °C - Adjust based on calibration
    EMISSIVITY_CORRECTION = 1.0  # Multiplier for emissivity correction
    
    # Data quality settings
    MIN_VALID_TEMP = -10.0  # °C - Reject readings below this
    MAX_VALID_TEMP = 150.0  # °C - Reject readings above this
    SMOOTHING_WINDOW = 3    # Number of readings to average for stability

class ThermalMonitor:
    def __init__(self, mock_mode=False):
        self.mock_mode = mock_mode
        self.setup_logging()
        self.temperature_buffer = []  # For smoothing
        self.calibration_offset = 0.0  # Will be set during calibration
        
        if mock_mode:
            from mock_mlx90640 import MockMLX90640
            self.sensor = MockMLX90640()
            self.logger.info("🔧 Running in MOCK MODE")
        else:
            try:
                import board
                import busio
                import adafruit_mlx90640
                
                # Initialize I2C with optimal settings for MLX90640
                i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)  # Reduced frequency for reliability
                self.sensor = adafruit_mlx90640.MLX90640(i2c)
                
                # Configure sensor for accuracy
                self.sensor.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_1_HZ
                
                self.logger.info("🌡️ Real MLX90640 sensor initialized successfully")
                
                # Perform initial sensor warm-up
                self.warm_up_sensor()
                
            except Exception as e:
                self.logger.error(f"Failed to initialize MLX90640: {e}")
                raise
    
    def setup_logging(self):
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/{ThermalConfig.LOG_FILE}'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def warm_up_sensor(self):
        """Warm up sensor for stable readings - MLX90640 needs ~2 minutes for thermal stability"""
        if not self.mock_mode:
            self.logger.info("🔥 Warming up MLX90640 sensor for accurate readings...")
            for i in range(5):
                try:
                    frame = [0] * 768
                    self.sensor.getFrame(frame)
                    time.sleep(2)  # Wait between readings
                    self.logger.info(f"   Warm-up reading {i+1}/5 completed")
                except Exception as e:
                    self.logger.warning(f"Warm-up reading {i+1} failed: {e}")
            self.logger.info("✅ Sensor warm-up completed")
    
    def validate_temperature_data(self, frame):
        """Validate and filter temperature readings"""
        valid_frame = []
        invalid_count = 0
        
        for temp in frame:
            if ThermalConfig.MIN_VALID_TEMP <= temp <= ThermalConfig.MAX_VALID_TEMP:
                # Apply calibration corrections
                corrected_temp = temp + ThermalConfig.AMBIENT_OFFSET
                corrected_temp *= ThermalConfig.EMISSIVITY_CORRECTION
                valid_frame.append(corrected_temp)
            else:
                invalid_count += 1
        
        if invalid_count > 50:  # More than ~6% invalid readings
            self.logger.warning(f"High number of invalid readings: {invalid_count}/768")
        
        return valid_frame if len(valid_frame) > 600 else None  # Need at least 78% valid data
    
    def smooth_temperature_reading(self, temps):
        """Apply smoothing to reduce sensor noise"""
        self.temperature_buffer.append(temps)
        
        # Keep only recent readings for smoothing
        if len(self.temperature_buffer) > ThermalConfig.SMOOTHING_WINDOW:
            self.temperature_buffer.pop(0)
        
        if len(self.temperature_buffer) < 2:
            return temps  # Not enough data for smoothing
        
        # Calculate smoothed values
        smoothed = {
            'timestamp': temps['timestamp'],
            'min_temp': np.mean([t['min_temp'] for t in self.temperature_buffer]),
            'max_temp': np.mean([t['max_temp'] for t in self.temperature_buffer]),
            'avg_temp': np.mean([t['avg_temp'] for t in self.temperature_buffer]),
            'std_temp': np.mean([t['std_temp'] for t in self.temperature_buffer]),
            'valid_pixels': temps['valid_pixels']
        }
        
        return smoothed

    def read_thermal_frame(self):
        """Read and process thermal camera frame with validation"""
        try:
            if self.mock_mode:
                frame = self.sensor.getFrame()
            else:
                frame = [0] * 768
                self.sensor.getFrame(frame)
            
            # Validate the data
            valid_frame = self.validate_temperature_data(frame)
            if not valid_frame:
                self.logger.error("Invalid temperature data received")
                return None
            
            # Calculate comprehensive statistics
            temps = {
                'timestamp': datetime.now().isoformat(),
                'min_temp': float(min(valid_frame)),
                'max_temp': float(max(valid_frame)),
                'avg_temp': float(np.mean(valid_frame)),
                'std_temp': float(np.std(valid_frame)),
                'valid_pixels': len(valid_frame),
                'frame_data': valid_frame if self.should_log_full_frame(valid_frame) else None
            }
            
            # Apply smoothing for stability
            smoothed_temps = self.smooth_temperature_reading(temps)
            
            return smoothed_temps
            
        except Exception as e:
            self.logger.error(f"Error reading thermal frame: {e}")
            return None
    
    def should_log_full_frame(self, frame):
        """Log full frame data when temperatures exceed warning threshold"""
        max_temp = max(frame)
        return max_temp > ThermalConfig.TEMP_WARNING
    
    def check_temperature_alerts(self, temps):
        """Enhanced alert system for transformer monitoring"""
        max_temp = temps['max_temp']
        avg_temp = temps['avg_temp']
        
        # Multi-level alert system
        if max_temp > ThermalConfig.TEMP_EMERGENCY:
            self.logger.critical(f"🚨 EMERGENCY: Max temp {max_temp:.1f}°C - IMMEDIATE ACTION REQUIRED!")
            return "EMERGENCY"
        elif max_temp > ThermalConfig.TEMP_CRITICAL:
            self.logger.critical(f"🔴 CRITICAL: Max temp {max_temp:.1f}°C, Avg temp {avg_temp:.1f}°C")
            return "CRITICAL"
        elif max_temp > ThermalConfig.TEMP_WARNING:
            self.logger.warning(f"⚠️ WARNING: Max temp {max_temp:.1f}°C, Avg temp {avg_temp:.1f}°C")
            return "WARNING"
        elif avg_temp > 45.0:  # Elevated but normal operating temperature
            self.logger.info(f"📈 ELEVATED: Max temp {max_temp:.1f}°C, Avg temp {avg_temp:.1f}°C")
            return "ELEVATED"
        else:
            self.logger.info(f"✅ NORMAL: Max temp {max_temp:.1f}°C, Avg temp {avg_temp:.1f}°C")
            return "NORMAL"
    
    def log_data(self, temps):
        """Enhanced logging with transformer-specific data"""
        try:
            file_exists = os.path.exists(ThermalConfig.DATA_FILE)
            
            with open(ThermalConfig.DATA_FILE, 'a', newline='') as csvfile:
                fieldnames = [
                    'timestamp', 'min_temp', 'max_temp', 'avg_temp', 'std_temp', 
                    'valid_pixels', 'status', 'temp_gradient', 'stability_index'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                    self.logger.info(f"📝 Created thermal data log: {ThermalConfig.DATA_FILE}")
                
                status = self.check_temperature_alerts(temps)
                
                # Calculate additional metrics for transformer monitoring
                temp_gradient = temps['max_temp'] - temps['min_temp']
                stability_index = temps['std_temp']  # Lower values indicate more uniform heating
                
                writer.writerow({
                    'timestamp': temps['timestamp'],
                    'min_temp': f"{temps['min_temp']:.2f}",
                    'max_temp': f"{temps['max_temp']:.2f}",
                    'avg_temp': f"{temps['avg_temp']:.2f}",
                    'std_temp': f"{temps['std_temp']:.2f}",
                    'valid_pixels': temps['valid_pixels'],
                    'status': status,
                    'temp_gradient': f"{temp_gradient:.2f}",
                    'stability_index': f"{stability_index:.2f}"
                })
                
                # Log full frame data if critical temperature detected
                if temps['frame_data'] and status in ['CRITICAL', 'EMERGENCY']:
                    frame_filename = f"thermal_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(f"logs/{frame_filename}", 'w') as frame_file:
                        json.dump({
                            'timestamp': temps['timestamp'],
                            'status': status,
                            'frame_data': temps['frame_data'],
                            'statistics': {
                                'min': temps['min_temp'],
                                'max': temps['max_temp'],
                                'avg': temps['avg_temp'],
                                'std': temps['std_temp']
                            }
                        }, frame_file, indent=2)
                    self.logger.info(f"💾 Full frame data saved: {frame_filename}")
                
        except Exception as e:
            self.logger.error(f"Error logging data: {e}")
    
    def display_live_stats(self, temps):
        """Enhanced live display for transformer monitoring"""
        status = self.check_temperature_alerts(temps)
        status_icon = {
            'NORMAL': '✅',
            'ELEVATED': '📈', 
            'WARNING': '⚠️',
            'CRITICAL': '🔴',
            'EMERGENCY': '🚨'
        }.get(status, '❓')
        
        print(f"\r{status_icon} Min: {temps['min_temp']:.1f}°C | " +
              f"Max: {temps['max_temp']:.1f}°C | " +
              f"Avg: {temps['avg_temp']:.1f}°C | " +
              f"Pixels: {temps['valid_pixels']}/768 | " +
              f"Time: {datetime.now().strftime('%H:%M:%S')}", end='', flush=True)
    
    def run_monitoring(self, duration_minutes=None):
        """Enhanced monitoring loop for transformer applications"""
        self.logger.info("🚀 Starting transformer thermal monitoring...")
        self.logger.info(f"📊 Data logging interval: {ThermalConfig.LOG_INTERVAL} seconds")
        self.logger.info(f"⚠️ Warning threshold: {ThermalConfig.TEMP_WARNING}°C")
        self.logger.info(f"🔴 Critical threshold: {ThermalConfig.TEMP_CRITICAL}°C")
        self.logger.info(f"🚨 Emergency threshold: {ThermalConfig.TEMP_EMERGENCY}°C")
        
        last_log_time = 0
        start_time = time.time()
        reading_count = 0
        error_count = 0
        
        try:
            while True:
                current_time = time.time()
                
                # Read thermal data
                temps = self.read_thermal_frame()
                if temps:
                    reading_count += 1
                    error_count = 0  # Reset error counter on successful reading
                    
                    # Display live stats
                    self.display_live_stats(temps)
                    
                    # Log to file at specified interval
                    if current_time - last_log_time >= ThermalConfig.LOG_INTERVAL:
                        print()  # New line after live stats
                        self.log_data(temps)
                        last_log_time = current_time
                        self.logger.info(f"📈 Readings: {reading_count}, Errors: {error_count}")
                else:
                    error_count += 1
                    if error_count > 10:
                        self.logger.error("Multiple consecutive sensor reading failures")
                        break
                
                # Exit condition for testing
                if duration_minutes and (current_time - start_time) > (duration_minutes * 60):
                    print(f"\n⏰ Monitoring completed after {duration_minutes} minutes")
                    break
                
                # Sleep based on refresh rate
                time.sleep(1.0 / ThermalConfig.REFRESH_RATE)
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Monitoring stopped by user after {reading_count} readings")
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            print(f"\n💥 Error during monitoring: {e}")
            self.logger.error(f"Monitoring error: {e}")
        
        # Final summary
        total_time = time.time() - start_time
        self.logger.info(f"📊 Session summary: {reading_count} readings in {total_time:.1f} seconds")
        self.logger.info(f"📈 Average rate: {reading_count/total_time:.1f} readings/second")

def calibrate_sensor():
    """Simple calibration routine - compare with known reference temperature"""
    print("🔧 MLX90640 Calibration Mode")
    print("Point sensor at a known temperature source and enter the reference temperature:")
    
    try:
        reference_temp = float(input("Reference temperature (°C): "))
        
        monitor = ThermalMonitor(mock_mode=False)
        
        print("Taking calibration readings...")
        readings = []
        for i in range(10):
            temps = monitor.read_thermal_frame()
            if temps:
                readings.append(temps['avg_temp'])
                print(f"Reading {i+1}: {temps['avg_temp']:.2f}°C")
            time.sleep(1)
        
        if readings:
            measured_avg = np.mean(readings)
            offset = reference_temp - measured_avg
            print(f"\nCalibration Results:")
            print(f"Measured average: {measured_avg:.2f}°C")
            print(f"Reference: {reference_temp:.2f}°C")
            print(f"Suggested offset: {offset:.2f}°C")
            print(f"\nUpdate ThermalConfig.AMBIENT_OFFSET = {offset:.2f}")
        
    except Exception as e:
        print(f"Calibration failed: {e}")

def main():
    """Main entry point with options for normal monitoring or calibration"""
    print("🔥 MLX90640 Transformer Thermal Monitor")
    print("=" * 50)
    
    mode = input("Select mode: (m)onitoring or (c)alibration: ").lower().strip()
    
    if mode.startswith('c'):
        calibrate_sensor()
        return
    
    # Normal monitoring mode
    try:
        duration = input("Enter monitoring duration in minutes (or press Enter for continuous): ").strip()
        duration = float(duration) if duration else None
    except ValueError:
        duration = None
        print("Invalid input, running continuously. Press Ctrl+C to stop.")
    
    # Create the monitor instance
    monitor = ThermalMonitor(mock_mode=False)
    
    # Get first reading using same method as in run_monitoring
    temps = monitor.read_thermal_frame()
    if temps:
        monitor.display_live_stats(temps)
        print()  # Line break for neatness
    
    # Now start monitoring loop
    monitor.run_monitoring(duration_minutes=duration)
    
    # Show data file location
    if os.path.exists(ThermalConfig.DATA_FILE):
        print(f"\n📄 Data saved to: {os.path.abspath(ThermalConfig.DATA_FILE)}")
        print(f"📁 Logs directory: {os.path.abspath('logs/')}")
    

if __name__ == "__main__":
    main()