"""
thermal_monitor_production.py

Merged, production-safe, feature-rich MLX90640 thermal monitor.
- Combines validation, smoothing, logging, calibration, mock mode, warm-up, and full-frame storage
  from the user's comprehensive script.
- Adds production safety: sensor health check, read retries, auto-shutdown on emergency,
  configurable refresh/read intervals, and simpler initialization.

Usage:
  - Run normally on a device with MLX90640 sensor attached.
  - Use mock_mode=True for development/testing without hardware.

"""

import time
import json
import csv
import sys
import logging
import os
from datetime import datetime

try:
    import numpy as np
except Exception:
    print("numpy is required. Install with: pip install numpy")
    raise

# Optional hardware imports; may fail in mock mode
_HW_AVAILABLE = True
try:
    import board
    import busio
    import adafruit_mlx90640
except Exception:
    _HW_AVAILABLE = False


# ====== CONFIGURATION ======
class ThermalConfig:
    # Sensor / read settings
    REFRESH_RATE_HZ = 1.0                # Hz - default effective refresh rate
    READ_INTERVAL = 1.0 / REFRESH_RATE_HZ
    FRAME_RETRIES = 3                    # Retry attempts when a frame read fails
    DATA_FILE = "thermal_data.csv"
    LOG_DIR = "logs"
    LOG_FILE = "thermal_monitor.log"

    # Safety thresholds (transformer-appropriate)
    TEMP_WARNING = 65.0
    TEMP_CRITICAL = 85.0
    TEMP_EMERGENCY = 105.0
    AUTO_SHUTDOWN_CRITICAL = True       # If True, program will exit on EMERGENCY

    # Calibration / emissivity
    AMBIENT_OFFSET = 0.0
    EMISSIVITY_CORRECTION = 1.0

    # Data quality / smoothing
    MIN_VALID_TEMP = -10.0
    MAX_VALID_TEMP = 150.0
    SMOOTHING_WINDOW = 3

    # Logging / behavior
    MIN_VALID_PERCENT = 0.78   # need at least this fraction of valid pixels
    LOG_FULL_FRAME_ON = ["CRITICAL", "EMERGENCY"]


# ====== LOGGER SETUP ======
os.makedirs(ThermalConfig.LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(ThermalConfig.LOG_DIR, ThermalConfig.LOG_FILE)),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("thermal_monitor")


# ====== MOCK SENSOR (for development without hardware) ======
class MockMLX90640:
    """Simple mock that produces realistic-ish frames for testing."""
    def __init__(self):
        self.frame_count = 0

    def getFrame(self, frame=None):
        # Return a synthetic 768-length frame (32x24) as list; if frame provided, mutate it
        base = 40.0 + 0.5 * np.sin(time.time())
        noise = np.random.normal(scale=0.5, size=768)
        values = (base + noise).tolist()
        if frame is None:
            return values
        else:
            for i, v in enumerate(values):
                frame[i] = v
            return frame


# ====== THERMAL MONITOR CLASS ======
class ThermalMonitor:
    def __init__(self, mock_mode=False, refresh_rate_hz=None):
        self.mock_mode = mock_mode
        self.temperature_buffer = []  # for smoothing
        self.calibration_offset = ThermalConfig.AMBIENT_OFFSET
        self.sensor = None

        # Allow dynamic refresh rate override
        if refresh_rate_hz:
            ThermalConfig.REFRESH_RATE_HZ = refresh_rate_hz
            ThermalConfig.READ_INTERVAL = 1.0 / refresh_rate_hz

        logger.info("Initializing ThermalMonitor (mock_mode=%s)", self.mock_mode)
        self._initialize_sensor()
        self.sensor_health_check()
        if not self.mock_mode:
            self.warm_up_sensor()

    def _initialize_sensor(self):
        if self.mock_mode or not _HW_AVAILABLE:
            self.sensor = MockMLX90640()
            logger.info("🔧 Running in MOCK MODE (no hardware required)")
            return

        try:
            i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
            self.sensor = adafruit_mlx90640.MLX90640(i2c)

            # Choose a conservative refresh rate if hardware supports it
            try:
                # pick the closest supported enum based on requested rate
                if ThermalConfig.REFRESH_RATE_HZ >= 2:
                    self.sensor.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
                else:
                    self.sensor.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_1_HZ
            except Exception:
                # If setting fails, leave default
                logger.warning("Unable to set sensor refresh rate; leaving default")

            logger.info("🌡️ Real MLX90640 sensor initialized")
        except Exception as e:
            logger.critical(f"Failed to initialize MLX90640: {e}")
            raise

    def sensor_health_check(self):
        """Perform a quick frame read to ensure the sensor is returning realistic data."""
        try:
            frame = [0] * 768
            if self.mock_mode:
                frame = self.sensor.getFrame()
            else:
                self.sensor.getFrame(frame)

            # Basic checks
            if len(frame) != 768:
                logger.critical("Sensor returned unexpected frame length: %s", len(frame))
                raise SystemExit(1)

            if all((isinstance(v, (int, float)) and v == 0) for v in frame):
                logger.critical("⚠ Sensor returning all-zero values. Aborting.")
                raise SystemExit(1)

            logger.info("✅ Sensor health check passed")
        except Exception as e:
            logger.critical(f"Sensor health check failed: {e}")
            raise SystemExit(1)

    def warm_up_sensor(self):
        """Warm-up to improve sensor stability. Uses a few frames spaced out."""
        logger.info("🔥 Warming up sensor for stable readings...")
        for i in range(5):
            try:
                frame = [0] * 768
                self.sensor.getFrame(frame)
                time.sleep(1)
                logger.info(f"   Warm-up reading {i+1}/5 completed")
            except Exception as e:
                logger.warning(f"Warm-up reading {i+1} failed: {e}")
        logger.info("✅ Sensor warm-up completed")

    def validate_temperature_data(self, frame):
        """Validate and filter temperature readings, apply emissivity and offset."""
        valid_frame = []
        invalid_count = 0
        for temp in frame:
            try:
                if ThermalConfig.MIN_VALID_TEMP <= temp <= ThermalConfig.MAX_VALID_TEMP:
                    corrected_temp = (temp + self.calibration_offset) * ThermalConfig.EMISSIVITY_CORRECTION
                    valid_frame.append(corrected_temp)
                else:
                    invalid_count += 1
            except Exception:
                invalid_count += 1

        if invalid_count > 50:
            logger.warning(f"High number of invalid readings: {invalid_count}/768")

        if len(valid_frame) / 768.0 < ThermalConfig.MIN_VALID_PERCENT:
            logger.error("Not enough valid pixels: %d/768 (%.2f%%)", len(valid_frame), 100.0 * len(valid_frame) / 768.0)
            return None

        return valid_frame

    def smooth_temperature_reading(self, temps):
        """Apply a moving-average smoothing across recent aggregated stats."""
        self.temperature_buffer.append(temps)
        if len(self.temperature_buffer) > ThermalConfig.SMOOTHING_WINDOW:
            self.temperature_buffer.pop(0)

        if len(self.temperature_buffer) < 2:
            return temps

        smoothed = {
            'timestamp': temps['timestamp'],
            'min_temp': float(np.mean([t['min_temp'] for t in self.temperature_buffer])),
            'max_temp': float(np.mean([t['max_temp'] for t in self.temperature_buffer])),
            'avg_temp': float(np.mean([t['avg_temp'] for t in self.temperature_buffer])),
            'std_temp': float(np.mean([t['std_temp'] for t in self.temperature_buffer])),
            'valid_pixels': int(np.mean([t['valid_pixels'] for t in self.temperature_buffer])),
            'frame_data': temps.get('frame_data')
        }
        return smoothed

    def should_log_full_frame(self, frame):
        try:
            return max(frame) > ThermalConfig.TEMP_WARNING
        except Exception:
            return False

    def read_thermal_frame(self):
        """Read a frame with retries, validate, aggregate and smooth."""
        for attempt in range(ThermalConfig.FRAME_RETRIES):
            try:
                frame = [0] * 768
                if self.mock_mode:
                    # Mock returns list
                    frame = self.sensor.getFrame()
                else:
                    self.sensor.getFrame(frame)

                valid_frame = self.validate_temperature_data(frame)
                if not valid_frame:
                    logger.warning("Read produced insufficient valid data; attempt %d/%d", attempt+1, ThermalConfig.FRAME_RETRIES)
                    time.sleep(1)
                    continue

                temps = {
                    'timestamp': datetime.now().isoformat(),
                    'min_temp': float(min(valid_frame)),
                    'max_temp': float(max(valid_frame)),
                    'avg_temp': float(np.mean(valid_frame)),
                    'std_temp': float(np.std(valid_frame)),
                    'valid_pixels': len(valid_frame),
                    'frame_data': valid_frame if self.should_log_full_frame(valid_frame) else None
                }

                smoothed = self.smooth_temperature_reading(temps)
                return smoothed

            except Exception as e:
                logger.warning(f"I²C/read failed (attempt {attempt+1}/{ThermalConfig.FRAME_RETRIES}): {e}")
                time.sleep(1)

        logger.error("❌ Sensor read failed after retries. Aborting monitoring loop.")
        raise SystemExit(1)

    def check_temperature_alerts(self, temps):
        max_temp = temps['max_temp']
        avg_temp = temps['avg_temp']

        if max_temp >= ThermalConfig.TEMP_EMERGENCY:
            logger.critical(f"🚨 EMERGENCY: Max temp {max_temp:.1f}°C - IMMEDIATE ACTION REQUIRED!")
            if ThermalConfig.AUTO_SHUTDOWN_CRITICAL:
                logger.critical("Auto-shutdown enabled: exiting program.")
                raise SystemExit(1)
            return "EMERGENCY"
        elif max_temp >= ThermalConfig.TEMP_CRITICAL:
            logger.critical(f"🔴 CRITICAL: Max temp {max_temp:.1f}°C, Avg {avg_temp:.1f}°C")
            return "CRITICAL"
        elif max_temp >= ThermalConfig.TEMP_WARNING:
            logger.warning(f"⚠️ WARNING: Max temp {max_temp:.1f}°C, Avg {avg_temp:.1f}°C")
            return "WARNING"
        elif avg_temp > 45.0:
            logger.info(f"📈 ELEVATED: Max {max_temp:.1f}°C, Avg {avg_temp:.1f}°C")
            return "ELEVATED"
        else:
            logger.info(f"✅ NORMAL: Max {max_temp:.1f}°C, Avg {avg_temp:.1f}°C")
            return "NORMAL"

    def log_data(self, temps, status=None):
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
                    logger.info(f"📝 Created thermal data log: {ThermalConfig.DATA_FILE}")

                if status is None:
                    status = self.check_temperature_alerts(temps)

                temp_gradient = temps['max_temp'] - temps['min_temp']
                stability_index = temps['std_temp']

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

            # Save full frame on severe statuses
            if temps.get('frame_data') and status in ThermalConfig.LOG_FULL_FRAME_ON:
                frame_filename = f"thermal_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(os.path.join(ThermalConfig.LOG_DIR, frame_filename), 'w') as frame_file:
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
                logger.info(f"💾 Full frame data saved: {frame_filename}")

        except Exception as e:
            logger.error(f"Error logging data: {e}")

    def display_live_stats(self, temps):
        status = None
        try:
            status = self.check_temperature_alerts(temps)
        except SystemExit:
            # re-raise so caller can handle
            raise
        except Exception:
            status = "UNKNOWN"

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
        logger.info("🚀 Starting transformer thermal monitoring...")
        logger.info(f"📊 Read interval: {ThermalConfig.READ_INTERVAL}s | Refresh rate: {ThermalConfig.REFRESH_RATE_HZ}Hz")

        last_log_time = 0
        start_time = time.time()
        reading_count = 0
        error_count = 0

        try:
            while True:
                current_time = time.time()
                try:
                    temps = self.read_thermal_frame()
                except SystemExit:
                    logger.critical("Aborting monitoring due to critical failure or emergency.")
                    break

                if temps:
                    reading_count += 1
                    error_count = 0

                    # Display
                    self.display_live_stats(temps)

                    # Periodic log (use READ_INTERVAL as default cadence if not otherwise)
                    if current_time - last_log_time >= max(1.0, ThermalConfig.READ_INTERVAL):
                        print()  # newline after live status
                        try:
                            status = self.check_temperature_alerts(temps)
                        except SystemExit:
                            logger.critical("Emergency triggered during status check.")
                            break
                        self.log_data(temps, status=status)
                        last_log_time = current_time
                        logger.info(f"📈 Readings: {reading_count}, Errors: {error_count}")
                else:
                    error_count += 1
                    if error_count > 10:
                        logger.error("Multiple consecutive sensor reading failures. Exiting.")
                        break

                # Duration cutoff for testing
                if duration_minutes and (current_time - start_time) > (duration_minutes * 60):
                    print(f"\n⏰ Monitoring completed after {duration_minutes} minutes")
                    break

                time.sleep(ThermalConfig.READ_INTERVAL)

        except KeyboardInterrupt:
            print(f"\n⏹️ Monitoring stopped by user after {reading_count} readings")
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error during monitoring: {e}")
        finally:
            total_time = time.time() - start_time
            if total_time > 0:
                logger.info(f"📊 Session summary: {reading_count} readings in {total_time:.1f} seconds")
                logger.info(f"📈 Average rate: {reading_count/total_time:.2f} readings/second")


# ====== Calibration helper (interactive) ======
def calibrate_sensor_interactive(mock_mode=False):
    print("🔧 MLX90640 Calibration Mode")
    try:
        reference_temp = float(input("Reference temperature (°C): "))
        monitor = ThermalMonitor(mock_mode=mock_mode)
        print("Taking calibration readings...")
        readings = []
        for i in range(10):
            temps = monitor.read_thermal_frame()
            if temps:
                readings.append(temps['avg_temp'])
                print(f"Reading {i+1}: {temps['avg_temp']:.2f}°C")
            time.sleep(1)

        if readings:
            measured_avg = float(np.mean(readings))
            offset = reference_temp - measured_avg
            print(f"\nCalibration Results:")
            print(f"Measured average: {measured_avg:.2f}°C")
            print(f"Reference: {reference_temp:.2f}°C")
            print(f"Suggested offset: {offset:.2f}°C")
            print(f"\nUpdate ThermalConfig.AMBIENT_OFFSET = {offset:.2f}")
        else:
            print("No valid calibration readings were collected.")

    except Exception as e:
        print(f"Calibration failed: {e}")


# ====== CLI Entrypoint ======
def main():
    print("=== Transformer Thermal Camera ===")
    print("1. Calibrate Sensor")
    print("2. Start Monitoring")
    print("3. Start Monitoring (mock mode)")
    print("4. Exit")

    choice = input("Select an option (1/2/3/4): ").strip()

    if choice == "1":
        mock = input("Use mock mode for calibration? (y/n): ").strip().lower() == 'y'
        calibrate_sensor_interactive(mock_mode=mock)
        return

    elif choice == "2":
        monitor = ThermalMonitor(mock_mode=False)
        duration = input("Enter duration in minutes (leave empty for unlimited): ").strip()
        duration = float(duration) if duration else None
        monitor.run_monitoring(duration_minutes=duration)

    elif choice == "3":
        monitor = ThermalMonitor(mock_mode=True)
        duration = input("Enter duration in minutes (leave empty for unlimited): ").strip()
        duration = float(duration) if duration else None
        monitor.run_monitoring(duration_minutes=duration)

    elif choice == "4":
        print("👋 Exiting.")
        return

    else:
        print("❌ Invalid choice.")


if __name__ == '__main__':
    main()
