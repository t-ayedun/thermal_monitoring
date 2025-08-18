#!/usr/bin/env python3
"""
Production-Safe IoT-Enabled MLX90640 Thermal Monitor
Merged version combining field reliability with cloud connectivity

Features from Production Script:
- Temperature threshold monitoring (warning/critical/emergency)
- Emergency auto-shutdown logic
- Frame validation and smoothing (invalid pixel filtering, moving averages)
- Sensor warm-up and health checks
- Detailed anomaly logging with full frame dumps

Features from IoT Enhancement:
- AWS IoT Core MQTT publishing with TLS certificates
- 1-minute statistical aggregation of thermal data
- Accuenergy power meter integration via Modbus/TCP
- JSON configuration management for AWS and testbed
- Graceful shutdown handling (SIGINT/SIGTERM)
- CSV persistence of aggregated data

Usage:
  python3 thermal_monitor_merged.py [--simulation] [--create-config] [--duration MINUTES]
"""

import time
import json
import csv
import sys
import logging
import os
import statistics
import signal
import argparse
from datetime import datetime, timezone
from collections import deque
from pathlib import Path

# Core dependencies
try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Required package missing: {e}")
    print("Install with: pip install numpy pandas")
    sys.exit(1)

# Hardware imports (optional - for real sensor mode)
_HW_AVAILABLE = True
try:
    import board
    import busio
    import adafruit_mlx90640
except ImportError:
    _HW_AVAILABLE = False

# IoT and connectivity imports (optional)
_IOT_AVAILABLE = True
try:
    import paho.mqtt.client as mqtt
    import ssl
    from awscrt import io, mqtt as aws_mqtt, auth, http
    from awsiot import mqtt_connection_builder
except ImportError:
    _IOT_AVAILABLE = False

# Modbus imports (optional - for power meter integration)
_MODBUS_AVAILABLE = True
try:
    from pymodbus.client.sync import ModbusTcpClient
except ImportError:
    _MODBUS_AVAILABLE = False


# ====== PRODUCTION CONFIGURATION (from original thermal_monitor_production.py) ======
class ThermalConfig:
    """Centralized configuration for thermal monitoring system"""
    
    # === SENSOR SETTINGS (Production Safety) ===
    REFRESH_RATE_HZ = 1.0                # Hz - conservative for stability
    READ_INTERVAL = 1.0 / REFRESH_RATE_HZ
    FRAME_RETRIES = 3                    # Retry attempts when frame read fails
    
    # === SAFETY THRESHOLDS (Production Safety) ===
    TEMP_WARNING = 65.0                  # °C - transformer warning level
    TEMP_CRITICAL = 85.0                 # °C - critical intervention needed
    TEMP_EMERGENCY = 105.0               # °C - immediate shutdown required
    AUTO_SHUTDOWN_CRITICAL = True        # Exit program on emergency
    
    # === DATA VALIDATION (Production Safety) ===
    MIN_VALID_TEMP = -10.0              # °C - filter impossible readings
    MAX_VALID_TEMP = 150.0              # °C - filter sensor errors
    MIN_VALID_PERCENT = 0.78            # Require 78% valid pixels minimum
    SMOOTHING_WINDOW = 3                # Moving average window size
    
    # === CALIBRATION (Production Safety) ===
    AMBIENT_OFFSET = 0.0                # °C - calibration offset
    EMISSIVITY_CORRECTION = 1.0         # Emissivity adjustment factor
    
    # === AGGREGATION SETTINGS (IoT Enhancement) ===
    AGGREGATION_INTERVAL = 60           # seconds - 1-minute summaries
    
    # === FILE PATHS ===
    DATA_FILE = "thermal_data.csv"
    LOG_DIR = "logs"
    LOG_FILE = "thermal_monitor.log"
    AWS_CONFIG_FILE = "aws_iot_config.json"
    TESTBED_CONFIG_FILE = "testbed_config.json"
    
    # === LOGGING BEHAVIOR (Production Safety) ===
    LOG_FULL_FRAME_ON = ["CRITICAL", "EMERGENCY"]  # When to save full thermal frames


# ====== LOGGING SETUP (Production Safety) ======
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


# ====== MOCK SENSOR (from Production Safety + IoT Enhancement) ======
class MockMLX90640:
    """Realistic mock sensor for development and testing without hardware"""
    
    def __init__(self):
        self.frame_count = 0
        self.base_temp = 25.0
        
    def getFrame(self, frame=None):
        """Generate realistic thermal frame data"""
        # Simulate realistic thermal variations
        self.frame_count += 1
        time_factor = time.time() * 0.1
        
        # Create realistic temperature distribution
        base = self.base_temp + 2.0 * np.sin(time_factor) + np.random.normal(0, 0.5)
        
        # Add some hot spots and gradients like real thermal data
        thermal_image = np.random.normal(base, 2.0, (24, 32))
        
        # Add a few hot spots
        if self.frame_count % 30 == 0:  # Occasional hot spot
            hot_y, hot_x = np.random.randint(0, 24), np.random.randint(0, 32)
            thermal_image[hot_y-2:hot_y+2, hot_x-2:hot_x+2] += np.random.uniform(10, 20)
        
        values = thermal_image.flatten().tolist()
        
        if frame is None:
            return values
        else:
            for i, v in enumerate(values):
                frame[i] = v
            return frame


# ====== DATA AGGREGATOR (IoT Enhancement) ======
class ThermalDataAggregator:
    """Handles 1-minute aggregation of thermal readings"""
    
    def __init__(self, aggregation_interval=60):
        self.aggregation_interval = aggregation_interval
        self.readings_buffer = deque()
        self.last_aggregation_time = time.time()
        logger.info(f"Data aggregator initialized: {aggregation_interval}s intervals")
        
    def add_reading(self, thermal_data):
        """Add thermal reading to aggregation buffer"""
        timestamp = time.time()
        self.readings_buffer.append({
            'min_temp': thermal_data['min_temp'],
            'avg_temp': thermal_data['avg_temp'],
            'max_temp': thermal_data['max_temp'],
            'std_temp': thermal_data['std_temp'],
            'valid_pixels': thermal_data['valid_pixels'],
            'timestamp': timestamp
        })
        
        # Remove readings older than aggregation interval
        cutoff_time = timestamp - self.aggregation_interval
        while self.readings_buffer and self.readings_buffer[0]['timestamp'] < cutoff_time:
            self.readings_buffer.popleft()
    
    def should_aggregate(self):
        """Check if it's time to create aggregated summary"""
        current_time = time.time()
        return (current_time - self.last_aggregation_time) >= self.aggregation_interval
    
    def get_aggregated_data(self):
        """Calculate 1-minute statistical summary"""
        if not self.readings_buffer:
            return None
            
        # Extract arrays for statistical analysis
        min_temps = [r['min_temp'] for r in self.readings_buffer]
        avg_temps = [r['avg_temp'] for r in self.readings_buffer]
        max_temps = [r['max_temp'] for r in self.readings_buffer]
        std_temps = [r['std_temp'] for r in self.readings_buffer]
        valid_pixels = [r['valid_pixels'] for r in self.readings_buffer]
        
        # Calculate comprehensive statistics
        aggregated = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'period_minutes': self.aggregation_interval / 60,
            'sample_count': len(self.readings_buffer),
            'temperature_stats': {
                'min_temp': {
                    'value': min(min_temps),
                    'average': statistics.mean(min_temps),
                    'std_dev': statistics.stdev(min_temps) if len(min_temps) > 1 else 0
                },
                'avg_temp': {
                    'value': statistics.mean(avg_temps),
                    'min': min(avg_temps),
                    'max': max(avg_temps),
                    'std_dev': statistics.stdev(avg_temps) if len(avg_temps) > 1 else 0
                },
                'max_temp': {
                    'value': max(max_temps),
                    'average': statistics.mean(max_temps),
                    'std_dev': statistics.stdev(max_temps) if len(max_temps) > 1 else 0
                },
                'thermal_stability': {
                    'avg_std_dev': statistics.mean(std_temps),
                    'pixel_consistency': statistics.mean(valid_pixels) / 768.0
                }
            }
        }
        
        self.last_aggregation_time = time.time()
        return aggregated


# ====== AWS IOT PUBLISHER (IoT Enhancement) ======
class AWSIoTPublisher:
    """Handles secure AWS IoT Core MQTT communication"""
    
    def __init__(self, config_file=None):
        if config_file is None:
            config_file = ThermalConfig.AWS_CONFIG_FILE
        self.config = self.load_config(config_file)
        self.connection = None
        self.is_connected = False
        
    def load_config(self, config_file):
        """Load AWS IoT configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            required_fields = ['endpoint', 'cert_path', 'key_path', 'ca_path', 'client_id', 'topic']
            missing_fields = [field for field in required_fields if field not in config]
            
            if missing_fields:
                raise ValueError(f"Missing required config fields: {missing_fields}")
                
            # Verify certificate files exist
            for cert_field in ['cert_path', 'key_path', 'ca_path']:
                if not os.path.exists(config[cert_field]):
                    raise FileNotFoundError(f"Certificate file not found: {config[cert_field]}")
                    
            return config
        except FileNotFoundError:
            logger.warning(f"AWS IoT config file not found: {config_file}")
            return None
        except Exception as e:
            logger.error(f"Error loading AWS IoT config: {e}")
            return None
    
    def connect(self):
        """Establish secure connection to AWS IoT Core"""
        if not self.config or not _IOT_AVAILABLE:
            logger.warning("AWS IoT not configured or SDK not available - skipping cloud integration")
            return False
            
        try:
            # AWS SDK event loop setup
            event_loop_group = io.EventLoopGroup(1)
            host_resolver = io.DefaultHostResolver(event_loop_group)
            client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)
            
            # Build mutual TLS connection
            self.connection = mqtt_connection_builder.mtls_from_path(
                endpoint=self.config['endpoint'],
                cert_filepath=self.config['cert_path'],
                pri_key_filepath=self.config['key_path'],
                client_bootstrap=client_bootstrap,
                ca_filepath=self.config['ca_path'],
                client_id=self.config['client_id'],
                clean_session=False,
                keep_alive_secs=30
            )
            
            # Establish connection
            connect_future = self.connection.connect()
            connect_future.result(timeout=10)  # 10 second timeout
            self.is_connected = True
            
            logger.info(f"✅ Connected to AWS IoT Core: {self.config['endpoint']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to AWS IoT Core: {e}")
            self.is_connected = False
            return False
    
    def publish_data(self, data, alert_level=None):
        """Publish thermal data to AWS IoT Core with alert context"""
        if not self.is_connected or not self.connection:
            logger.debug("Not connected to AWS IoT Core - data not published")
            return False
            
        try:
            # Enhanced payload with alert context (Production Safety)
            message = {
                'device_id': self.config['client_id'],
                'sensor_type': 'MLX90640',
                'location': self.config.get('location', 'unknown'),
                'alert_level': alert_level or 'NORMAL',
                'data': data,
                'system_health': {
                    'connection_stable': True,
                    'data_quality': 'good' if data.get('sample_count', 0) > 10 else 'limited'
                }
            }
            
            # Choose topic based on alert level (Production Safety)
            topic = self.config['topic']
            if alert_level in ['CRITICAL', 'EMERGENCY']:
                topic = f"{self.config['topic']}/alerts"
            
            # Publish with QoS 1 for reliable delivery
            publish_future = self.connection.publish(
                topic=topic,
                payload=json.dumps(message, indent=2),
                qos=aws_mqtt.QoS.AT_LEAST_ONCE
            )
            publish_future.result(timeout=5)
            
            logger.info(f"📤 Published to AWS IoT: {topic} (alert: {alert_level})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to publish to AWS IoT Core: {e}")
            return False
    
    def disconnect(self):
        """Clean disconnection from AWS IoT Core"""
        if self.connection and self.is_connected:
            try:
                disconnect_future = self.connection.disconnect()
                disconnect_future.result(timeout=5)
                logger.info("🔌 Disconnected from AWS IoT Core")
            except Exception as e:
                logger.error(f"Error during AWS IoT disconnect: {e}")
            finally:
                self.is_connected = False


# ====== TESTBED INTEGRATION (IoT Enhancement) ======
class TestbedIntegration:
    """Integration with Accuenergy meter and office testbed infrastructure"""
    
    def __init__(self, config_file=None):
        if config_file is None:
            config_file = ThermalConfig.TESTBED_CONFIG_FILE
        self.config = self.load_config(config_file)
        self.accuenergy_client = None
        self.connection_healthy = False
        
    def load_config(self, config_file):
        """Load testbed integration configuration"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Testbed config not found: {config_file}")
            return {'accuenergy': {'enabled': False}, 'teltonika': {'enabled': False}}
        except Exception as e:
            logger.error(f"Error loading testbed config: {e}")
            return {'accuenergy': {'enabled': False}, 'teltonika': {'enabled': False}}
    
    def connect_accuenergy(self):
        """Connect to Accuenergy power meter via Modbus/TCP"""
        if not self.config.get('accuenergy', {}).get('enabled', False) or not _MODBUS_AVAILABLE:
            return False
            
        try:
            meter_config = self.config['accuenergy']
            self.accuenergy_client = ModbusTcpClient(
                host=meter_config['host'],
                port=meter_config.get('port', 502),
                timeout=meter_config.get('timeout', 3)
            )
            
            if self.accuenergy_client.connect():
                logger.info(f"🔌 Connected to Accuenergy meter: {meter_config['host']}")
                self.connection_healthy = True
                return True
            else:
                logger.error("❌ Failed to connect to Accuenergy meter")
                return False
                
        except Exception as e:
            logger.error(f"Accuenergy connection error: {e}")
            return False
    
    def read_power_data(self):
        """Read power consumption data from Accuenergy meter"""
        if not self.accuenergy_client or not self.connection_healthy:
            return None
            
        try:
            meter_config = self.config['accuenergy']
            registers = meter_config.get('registers', {})
            
            # Read power registers (addresses depend on meter model)
            result = self.accuenergy_client.read_holding_registers(
                address=registers.get('active_power', 1000),
                count=4
            )
            
            if result.isError():
                logger.warning("Error reading Accuenergy registers")
                self.connection_healthy = False
                return None
                
            return {
                'active_power': result.registers[0] if len(result.registers) > 0 else 0,
                'reactive_power': result.registers[1] if len(result.registers) > 1 else 0,
                'voltage': result.registers[2] if len(result.registers) > 2 else 0,
                'current': result.registers[3] if len(result.registers) > 3 else 0,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error reading power data: {e}")
            self.connection_healthy = False
            return None


# ====== MAIN THERMAL MONITOR (Merged Production + IoT) ======
class ProductionThermalMonitor:
    """Main thermal monitoring system combining production safety with IoT capabilities"""
    
    def __init__(self, simulation_mode=False):
        self.simulation_mode = simulation_mode
        self.running = True
        
        # === PRODUCTION SAFETY COMPONENTS ===
        self.temperature_buffer = []        # For smoothing (Production Safety)
        self.calibration_offset = ThermalConfig.AMBIENT_OFFSET
        self.sensor = None
        self.reading_count = 0
        self.error_count = 0
        
        # === IOT ENHANCEMENT COMPONENTS ===
        self.data_aggregator = ThermalDataAggregator(ThermalConfig.AGGREGATION_INTERVAL)
        self.aws_publisher = AWSIoTPublisher()
        self.testbed = TestbedIntegration()
        
        # === DATA PERSISTENCE ===
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_file = f"thermal_aggregated_{self.session_id}.csv"
        self.detailed_csv = f"thermal_detailed_{datetime.now().strftime('%Y%m%d')}.csv"  # Daily file
        self.init_csv_files()
        
        # === SIGNAL HANDLERS (IoT Enhancement) ===
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("🔧 ThermalMonitor initialized (simulation=%s)", simulation_mode)
    
    def init_csv_files(self):
        """Initialize CSV files with improved naming and structure"""
        
        # === AGGREGATED DATA CSV (1-minute summaries) ===
        agg_headers = [
            'timestamp', 'session_id', 'period_minutes', 'sample_count',
            'min_temp', 'avg_temp', 'max_temp', 'temp_stability', 'temp_gradient',
            'power_active_w', 'power_voltage_v', 'alert_level', 'data_quality'
        ]
        
        df_agg = pd.DataFrame(columns=agg_headers)
        df_agg.to_csv(self.csv_file, index=False)
        logger.info(f"📝 Initialized aggregated CSV: {self.csv_file}")
        
        # === DETAILED DATA CSV (individual readings) ===
        detailed_headers = [
            'timestamp', 'session_id', 'reading_number',
            'min_temp', 'max_temp', 'avg_temp', 'std_temp', 'valid_pixels', 
            'temp_gradient', 'stability_index', 'alert_level'
        ]
        
        # Check if daily detailed file exists, if not create with headers
        if not os.path.exists(self.detailed_csv):
            df_detailed = pd.DataFrame(columns=detailed_headers)
            df_detailed.to_csv(self.detailed_csv, index=False)
            logger.info(f"📝 Initialized detailed CSV: {self.detailed_csv}")
        else:
            logger.info(f"📝 Appending to existing detailed CSV: {self.detailed_csv}")
    
    def signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully (IoT Enhancement)"""
        logger.info(f"📶 Received signal {sig}, initiating graceful shutdown...")
        self.running = False
    
    # === SENSOR INITIALIZATION (Production Safety) ===
    def _initialize_sensor(self):
        """Initialize MLX90640 with production safety checks"""
        if self.simulation_mode or not _HW_AVAILABLE:
            self.sensor = MockMLX90640()
            logger.info("🔧 Running in SIMULATION MODE")
            return True

        try:
            i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)  # Conservative frequency
            self.sensor = adafruit_mlx90640.MLX90640(i2c)

            # Set conservative refresh rate for stability
            try:
                if ThermalConfig.REFRESH_RATE_HZ >= 2:
                    self.sensor.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
                else:
                    self.sensor.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_1_HZ
                logger.info(f"🌡️ Sensor refresh rate set to {ThermalConfig.REFRESH_RATE_HZ}Hz")
            except Exception:
                logger.warning("Unable to set sensor refresh rate; using default")

            logger.info("🌡️ Real MLX90640 sensor initialized")
            return True
            
        except Exception as e:
            logger.critical(f"❌ Failed to initialize MLX90640: {e}")
            return False

    def sensor_health_check(self):
        """Comprehensive sensor health validation (Production Safety)"""
        logger.info("🔍 Performing sensor health check...")
        
        try:
            frame = [0] * 768
            if self.simulation_mode:
                frame = self.sensor.getFrame()
            else:
                self.sensor.getFrame(frame)

            # Validate frame structure
            if len(frame) != 768:
                logger.critical(f"❌ Invalid frame length: {len(frame)} (expected 768)")
                return False

            # Check for sensor errors (all zeros, all same value)
            if all(v == 0 for v in frame):
                logger.critical("❌ Sensor returning all-zero values")
                return False
                
            if len(set(frame)) < 10:  # Too little variation
                logger.critical("❌ Sensor returning insufficient variation")
                return False

            # Check temperature range sanity
            min_temp, max_temp = min(frame), max(frame)
            if not (ThermalConfig.MIN_VALID_TEMP <= min_temp <= ThermalConfig.MAX_VALID_TEMP):
                logger.critical(f"❌ Invalid temperature range: {min_temp:.1f} to {max_temp:.1f}°C")
                return False

            logger.info(f"✅ Sensor health check passed (temp range: {min_temp:.1f} to {max_temp:.1f}°C)")
            return True
            
        except Exception as e:
            logger.critical(f"❌ Sensor health check failed: {e}")
            return False

    def warm_up_sensor(self):
        """Sensor warm-up for stable readings (Production Safety)"""
        logger.info("🔥 Warming up sensor for stable readings...")
        for i in range(5):
            try:
                frame = [0] * 768
                if self.simulation_mode:
                    frame = self.sensor.getFrame()
                else:
                    self.sensor.getFrame(frame)
                time.sleep(1)
                logger.info(f"   Warm-up reading {i+1}/5: {np.mean(frame):.1f}°C average")
            except Exception as e:
                logger.warning(f"Warm-up reading {i+1} failed: {e}")
        logger.info("✅ Sensor warm-up completed")

    # === DATA ACQUISITION (Production Safety) ===
    def validate_temperature_data(self, frame):
        """Validate and filter temperature readings with calibration (Production Safety)"""
        valid_frame = []
        invalid_count = 0
        
        for temp in frame:
            try:
                if ThermalConfig.MIN_VALID_TEMP <= temp <= ThermalConfig.MAX_VALID_TEMP:
                    # Apply calibration corrections
                    corrected_temp = (temp + self.calibration_offset) * ThermalConfig.EMISSIVITY_CORRECTION
                    valid_frame.append(corrected_temp)
                else:
                    invalid_count += 1
            except Exception:
                invalid_count += 1

        # Production safety: warn about data quality issues
        if invalid_count > 50:
            logger.warning(f"⚠️ High number of invalid readings: {invalid_count}/768")

        # Production safety: ensure minimum data quality
        valid_percent = len(valid_frame) / 768.0
        if valid_percent < ThermalConfig.MIN_VALID_PERCENT:
            logger.error(f"❌ Insufficient valid pixels: {len(valid_frame)}/768 ({valid_percent:.1%})")
            return None

        return valid_frame

    def smooth_temperature_reading(self, temps):
        """Apply moving average smoothing (Production Safety)"""
        self.temperature_buffer.append(temps)
        if len(self.temperature_buffer) > ThermalConfig.SMOOTHING_WINDOW:
            self.temperature_buffer.pop(0)

        if len(self.temperature_buffer) < 2:
            return temps

        # Calculate smoothed values
        smoothed = {
            'timestamp': temps['timestamp'],
            'min_temp': float(np.mean([t['min_temp'] for t in self.temperature_buffer])),
            'max_temp': float(np.mean([t['max_temp'] for t in self.temperature_buffer])),
            'avg_temp': float(np.mean([t['avg_temp'] for t in self.temperature_buffer])),
            'std_temp': float(np.mean([t['std_temp'] for t in self.temperature_buffer])),
            'valid_pixels': int(np.mean([t['valid_pixels'] for t in self.temperature_buffer])),
            'frame_data': temps.get('frame_data')  # Keep latest frame data
        }
        return smoothed

    def read_thermal_frame(self):
        """Read thermal frame with retries and validation (Production Safety)"""
        for attempt in range(ThermalConfig.FRAME_RETRIES):
            try:
                frame = [0] * 768
                if self.simulation_mode:
                    frame = self.sensor.getFrame()
                else:
                    self.sensor.getFrame(frame)

                # Validate frame data
                valid_frame = self.validate_temperature_data(frame)
                if not valid_frame:
                    logger.warning(f"⚠️ Frame validation failed; attempt {attempt+1}/{ThermalConfig.FRAME_RETRIES}")
                    time.sleep(1)
                    continue

                # Calculate frame statistics
                temps = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'min_temp': float(min(valid_frame)),
                    'max_temp': float(max(valid_frame)),
                    'avg_temp': float(np.mean(valid_frame)),
                    'std_temp': float(np.std(valid_frame)),
                    'valid_pixels': len(valid_frame),
                    'frame_data': valid_frame if self.should_log_full_frame(valid_frame) else None
                }

                # Apply smoothing (Production Safety)
                smoothed = self.smooth_temperature_reading(temps)
                return smoothed

            except Exception as e:
                logger.warning(f"⚠️ Frame read failed (attempt {attempt+1}/{ThermalConfig.FRAME_RETRIES}): {e}")
                time.sleep(1)

        logger.error("❌ Sensor read failed after all retries")
        return None

    def should_log_full_frame(self, frame):
        """Determine if full frame should be logged (Production Safety)"""
        try:
            return max(frame) > ThermalConfig.TEMP_WARNING
        except Exception:
            return False

    # === SAFETY AND ALERTING (Production Safety) ===
    def check_temperature_alerts(self, temps):
        """Check temperature thresholds and trigger alerts (Production Safety)"""
        max_temp = temps['max_temp']
        avg_temp = temps['avg_temp']
        
        # Emergency shutdown logic
        if max_temp >= ThermalConfig.TEMP_EMERGENCY:
            logger.critical(f"🚨 EMERGENCY: Max temp {max_temp:.1f}°C - IMMEDIATE ACTION REQUIRED!")
            if ThermalConfig.AUTO_SHUTDOWN_CRITICAL:
                logger.critical("🛑 Auto-shutdown enabled: terminating monitoring")
                self.running = False  # Trigger graceful shutdown
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
            logger.debug(f"✅ NORMAL: Max {max_temp:.1f}°C, Avg {avg_temp:.1f}°C")
            return "NORMAL"

    def log_detailed_frame_data(self, temps, status):
        """Log individual readings to daily detailed CSV (Production Safety)"""
        try:
            temp_gradient = temps['max_temp'] - temps['min_temp']
            stability_index = temps['std_temp']
            
            # Enhanced detailed logging with session tracking
            row_data = {
                'timestamp': temps['timestamp'],
                'session_id': self.session_id,
                'reading_number': self.reading_count,
                'min_temp': f"{temps['min_temp']:.2f}",
                'max_temp': f"{temps['max_temp']:.2f}",
                'avg_temp': f"{temps['avg_temp']:.2f}",
                'std_temp': f"{temps['std_temp']:.2f}",
                'valid_pixels': temps['valid_pixels'],
                'temp_gradient': f"{temp_gradient:.2f}",
                'stability_index': f"{stability_index:.2f}",
                'alert_level': status
            }
            
            # Append to daily detailed file
            df = pd.DataFrame([row_data])
            df.to_csv(self.detailed_csv, mode='a', header=False, index=False)

            # Save full frame on severe conditions (Production Safety)
            if temps.get('frame_data') and status in ThermalConfig.LOG_FULL_FRAME_ON:
                frame_filename = f"thermal_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{status}_{self.session_id}.json"
                frame_path = os.path.join(ThermalConfig.LOG_DIR, frame_filename)
                
                with open(frame_path, 'w') as frame_file:
                    json.dump({
                        'timestamp': temps['timestamp'],
                        'session_id': self.session_id,
                        'status': status,
                        'alert_triggered': True,
                        'frame_data': temps['frame_data'],
                        'statistics': {
                            'min': temps['min_temp'],
                            'max': temps['max_temp'],
                            'avg': temps['avg_temp'],
                            'std': temps['std_temp'],
                            'gradient': temp_gradient,
                            'stability': stability_index
                        },
                        'metadata': {
                            'sensor_model': 'MLX90640',
                            'valid_pixels': temps['valid_pixels'],
                            'total_pixels': 768,
                            'reading_number': self.reading_count
                        }
                    }, frame_file, indent=2)
                
                logger.info(f"💾 Full frame saved: {frame_filename}")

        except Exception as e:
            logger.error(f"Error in detailed logging: {e}")

    # === AGGREGATED DATA PERSISTENCE (IoT Enhancement) ===
    def save_aggregated_to_csv(self, aggregated_data, power_data=None, alert_level="NORMAL"):
        """Save 1-minute aggregated data with enhanced formatting"""
        try:
            temp_stats = aggregated_data['temperature_stats']
            temp_gradient = temp_stats['max_temp']['value'] - temp_stats['min_temp']['value']
            
            # Data quality assessment
            pixel_consistency = temp_stats['thermal_stability']['pixel_consistency']
            data_quality = "excellent" if pixel_consistency > 0.95 else "good" if pixel_consistency > 0.8 else "fair"
            
            row_data = {
                'timestamp': aggregated_data['timestamp'],
                'session_id': self.session_id,
                'period_minutes': aggregated_data['period_minutes'],
                'sample_count': aggregated_data['sample_count'],
                'min_temp': f"{temp_stats['min_temp']['value']:.2f}",
                'avg_temp': f"{temp_stats['avg_temp']['value']:.2f}",  # This is the 1-minute average!
                'max_temp': f"{temp_stats['max_temp']['value']:.2f}",
                'temp_stability': f"{temp_stats['thermal_stability']['avg_std_dev']:.2f}",
                'temp_gradient': f"{temp_gradient:.2f}",
                'power_active_w': f"{power_data['active_power']:.1f}" if power_data else "N/A",
                'power_voltage_v': f"{power_data['voltage']:.1f}" if power_data else "N/A",
                'alert_level': alert_level,
                'data_quality': data_quality
            }
            
            df = pd.DataFrame([row_data])
            df.to_csv(self.csv_file, mode='a', header=False, index=False)
            
            logger.info(f"💾 1-min avg: {temp_stats['avg_temp']['value']:.1f}°C from {aggregated_data['sample_count']} samples")
            
        except Exception as e:
            logger.error(f"Error saving aggregated data: {e}")

    def create_aws_payload(self, aggregated_data, power_data=None, alert_level="NORMAL"):
        """Create enhanced AWS IoT payload with safety context (Merged)"""
        payload = {
            'device_info': {
                'device_id': 'thermal-camera-01',
                'sensor_model': 'MLX90640',
                'firmware_version': '2.0.0',  # Merged version
                'location': 'office_testbed'
            },
            'thermal_data': aggregated_data,
            'safety_status': {
                'alert_level': alert_level,
                'auto_shutdown_enabled': ThermalConfig.AUTO_SHUTDOWN_CRITICAL,
                'thresholds': {
                    'warning': ThermalConfig.TEMP_WARNING,
                    'critical': ThermalConfig.TEMP_CRITICAL,
                    'emergency': ThermalConfig.TEMP_EMERGENCY
                }
            },
            'metadata': {
                'data_quality': 'good' if aggregated_data['sample_count'] > 10 else 'limited',
                'aggregation_method': 'statistical_summary',
                'raw_samples': aggregated_data['sample_count'],
                'simulation_mode': self.simulation_mode
            }
        }
        
        # Add power data if available (IoT Enhancement)
        if power_data:
            payload['power_data'] = power_data
            payload['metadata']['power_integration'] = True
        
        return payload

    def display_live_stats(self, temps):
        """Real-time display with safety status (Production Safety)"""
        try:
            status = self.check_temperature_alerts(temps)
        except Exception:
            status = "UNKNOWN"

        status_icon = {
            'NORMAL': '✅',
            'ELEVATED': '📈', 
            'WARNING': '⚠️',
            'CRITICAL': '🔴',
            'EMERGENCY': '🚨',
            'UNKNOWN': '❓'
        }.get(status, '❓')

        # Enhanced live display
        gradient = temps['max_temp'] - temps['min_temp']
        print(f"\r{status_icon} Avg: {temps['avg_temp']:.1f}°C | " +
              f"Range: [{temps['min_temp']:.1f}, {temps['max_temp']:.1f}]°C | " +
              f"∇: {gradient:.1f}°C | " +
              f"Pixels: {temps['valid_pixels']}/768 | " +
              f"Count: {self.reading_count} | " +
              f"{datetime.now().strftime('%H:%M:%S')}", end='', flush=True)

    # === MAIN MONITORING LOOP (Merged Production + IoT) ===
    def run_monitoring(self, duration_minutes=None):
        """Main monitoring loop combining safety checks with IoT publishing"""
        
        # === INITIALIZATION PHASE ===
        logger.info("🚀 Starting production thermal monitoring with IoT integration...")
        
        # Initialize sensor with safety checks (Production Safety)
        if not self._initialize_sensor():
            logger.critical("❌ Sensor initialization failed")
            return False
            
        if not self.sensor_health_check():
            logger.critical("❌ Sensor health check failed")
            return False
            
        if not self.simulation_mode:
            self.warm_up_sensor()
        
        # Initialize IoT connections (IoT Enhancement)
        aws_connected = self.aws_publisher.connect()
        testbed_connected = self.testbed.connect_accuenergy()
        
        logger.info(f"🌐 AWS IoT: {'✅ Connected' if aws_connected else '❌ Disabled'}")
        logger.info(f"⚡ Testbed: {'✅ Connected' if testbed_connected else '❌ Disabled'}")
        logger.info(f"📊 Data aggregation: {ThermalConfig.AGGREGATION_INTERVAL}s intervals")
        logger.info(f"🔄 Read interval: {ThermalConfig.READ_INTERVAL}s")
        
        # === MONITORING LOOP ===
        start_time = time.time()
        last_log_time = 0
        highest_alert_level = "NORMAL"
        
        try:
            while self.running:
                current_time = time.time()
                
                # === READ SENSOR DATA (Production Safety) ===
                thermal_reading = self.read_thermal_frame()
                if not thermal_reading:
                    self.error_count += 1
                    if self.error_count > 10:
                        logger.error("❌ Multiple consecutive read failures - aborting")
                        break
                    time.sleep(ThermalConfig.READ_INTERVAL)
                    continue
                
                self.reading_count += 1
                self.error_count = 0
                
                # === SAFETY CHECKS (Production Safety) ===
                alert_level = self.check_temperature_alerts(thermal_reading)
                if alert_level in ['CRITICAL', 'EMERGENCY']:
                    highest_alert_level = alert_level
                
                # Emergency shutdown check
                if not self.running:  # check_temperature_alerts may have triggered shutdown
                    break
                
                # === ADD TO AGGREGATION BUFFER (IoT Enhancement) ===
                self.data_aggregator.add_reading(thermal_reading)
                
                # === LIVE DISPLAY (Production Safety) ===
                self.display_live_stats(thermal_reading)
                
                # === DETAILED LOGGING (Production Safety) ===
                if current_time - last_log_time >= max(10.0, ThermalConfig.READ_INTERVAL * 5):
                    print()  # New line after live stats
                    self.log_detailed_frame_data(thermal_reading, alert_level)
                    last_log_time = current_time
                
                # === 1-MINUTE AGGREGATION AND IOT PUBLISHING (IoT Enhancement) ===
                if self.data_aggregator.should_aggregate():
                    aggregated_data = self.data_aggregator.get_aggregated_data()
                    
                    if aggregated_data:
                        # Read power data from testbed if available
                        power_data = None
                        if testbed_connected:
                            power_data = self.testbed.read_power_data()
                            if not power_data:
                                logger.warning("⚡ Power meter read failed")
                        
                        # Save aggregated data to CSV
                        self.save_aggregated_to_csv(aggregated_data, power_data, highest_alert_level)
                        
                        # Publish to AWS IoT Core
                        if aws_connected:
                            aws_payload = self.create_aws_payload(aggregated_data, power_data, highest_alert_level)
                            success = self.aws_publisher.publish_data(aws_payload, highest_alert_level)
                            if not success:
                                logger.warning("📡 AWS IoT publish failed")
                        
                        # Log 1-minute summary
                        avg_temp = aggregated_data['temperature_stats']['avg_temp']['value']
                        sample_count = aggregated_data['sample_count']
                        logger.info(f"📊 1-min summary: {avg_temp:.1f}°C avg from {sample_count} samples (alert: {highest_alert_level})")
                        
                        # Reset alert level for next period
                        highest_alert_level = "NORMAL"
                
                # === DURATION CHECK ===
                if duration_minutes and (current_time - start_time) > (duration_minutes * 60):
                    print(f"\n⏰ Monitoring completed after {duration_minutes} minutes")
                    break
                
                # === WAIT FOR NEXT READING ===
                time.sleep(ThermalConfig.READ_INTERVAL)
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Monitoring stopped by user after {self.reading_count} readings")
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Unexpected error in monitoring loop: {e}")
        finally:
            self.cleanup()
            
        # === SESSION SUMMARY ===
        total_time = time.time() - start_time
        if total_time > 0:
            logger.info(f"📊 Session summary: {self.reading_count} readings in {total_time:.1f}s")
            logger.info(f"📈 Average rate: {self.reading_count/total_time:.2f} readings/second")
        
        return True

    def cleanup(self):
        """Comprehensive cleanup for graceful shutdown (IoT Enhancement)"""
        logger.info("🧹 Performing system cleanup...")
        
        # Process any remaining aggregated data
        if self.data_aggregator.readings_buffer:
            final_data = self.data_aggregator.get_aggregated_data()
            if final_data:
                logger.info("💾 Processing final aggregated data...")
                power_data = None
                if self.testbed.connection_healthy:
                    power_data = self.testbed.read_power_data()
                
                self.save_aggregated_to_csv(final_data, power_data, "SHUTDOWN")
                
                if self.aws_publisher.is_connected:
                    aws_payload = self.create_aws_payload(final_data, power_data, "SHUTDOWN")
                    self.aws_publisher.publish_data(aws_payload, "SHUTDOWN")
        
        # Disconnect from external services
        self.aws_publisher.disconnect()
        
        if self.testbed.accuenergy_client:
            try:
                self.testbed.accuenergy_client.close()
                logger.info("🔌 Disconnected from Accuenergy meter")
            except Exception as e:
                logger.error(f"Error disconnecting from power meter: {e}")
        
        logger.info("✅ Cleanup completed")


# ====== CALIBRATION HELPER (Production Safety) ======
def calibrate_sensor_interactive(simulation_mode=False):
    """Interactive sensor calibration mode (Production Safety)"""
    print("🔧 MLX90640 Calibration Mode")
    try:
        reference_temp = float(input("Reference temperature (°C): "))
        monitor = ProductionThermalMonitor(simulation_mode=simulation_mode)
        
        if not monitor._initialize_sensor():
            print("❌ Sensor initialization failed")
            return
            
        if not monitor.sensor_health_check():
            print("❌ Sensor health check failed")
            return
        
        print("📊 Taking calibration readings...")
        readings = []
        
        for i in range(10):
            thermal_data = monitor.read_thermal_frame()
            if thermal_data:
                readings.append(thermal_data['avg_temp'])
                print(f"Reading {i+1}/10: {thermal_data['avg_temp']:.2f}°C")
            time.sleep(2)

        if readings:
            measured_avg = float(np.mean(readings))
            offset = reference_temp - measured_avg
            std_dev = float(np.std(readings))
            
            print(f"\n📊 Calibration Results:")
            print(f"Measured average: {measured_avg:.2f}°C ± {std_dev:.2f}°C")
            print(f"Reference: {reference_temp:.2f}°C")
            print(f"Suggested offset: {offset:.2f}°C")
            print(f"\n🔧 Update ThermalConfig.AMBIENT_OFFSET = {offset:.2f}")
            
            if std_dev > 2.0:
                print("⚠️ Warning: High measurement variation - check sensor stability")
        else:
            print("❌ No valid calibration readings collected")

    except ValueError:
        print("❌ Invalid reference temperature")
    except Exception as e:
        print(f"❌ Calibration failed: {e}")


# ====== CONFIGURATION FILE GENERATORS (IoT Enhancement) ======
def create_sample_aws_config():
    """Generate sample AWS IoT configuration file"""
    sample_config = {
        "endpoint": "your-iot-endpoint.iot.us-east-1.amazonaws.com",
        "cert_path": "/home/pi/thermal_project/certs/device-certificate.pem.crt",
        "key_path": "/home/pi/thermal_project/certs/private.pem.key", 
        "ca_path": "/home/pi/thermal_project/certs/Amazon-root-CA-1.pem",
        "client_id": "thermal-camera-01",
        "topic": "thermal/data",
        "location": "office_testbed",
        "region": "us-east-1"
    }
    
    config_path = f"{ThermalConfig.AWS_CONFIG_FILE}.sample"
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print(f"📄 Sample AWS IoT config created: {config_path}")
    print(f"Copy to {ThermalConfig.AWS_CONFIG_FILE} and update with your actual values")

def create_sample_testbed_config():
    """Generate sample testbed configuration file"""
    sample_config = {
        "accuenergy": {
            "enabled": True,
            "host": "192.168.1.100",
            "port": 502,
            "timeout": 3,
            "unit_id": 1,
            "registers": {
                "active_power": 1000,
                "reactive_power": 1002,
                "voltage": 1004,
                "current": 1006
            }
        },
        "teltonika": {
            "enabled": false,
            "host": "192.168.1.101", 
            "api_endpoint": "/api/data",
            "username": "admin",
            "password": "your_password"
        }
    }
    
    config_path = f"{ThermalConfig.TESTBED_CONFIG_FILE}.sample"
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print(f"📄 Sample testbed config created: {config_path}")
    print(f"Copy to {ThermalConfig.TESTBED_CONFIG_FILE} and update with your actual values")


# ====== COMMAND LINE INTERFACE (Enhanced) ======
def main():
    parser = argparse.ArgumentParser(
        description='Production-Safe IoT Thermal Monitor for MLX90640',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Interactive menu
  %(prog)s --simulation              # Run in simulation mode  
  %(prog)s --duration 30             # Monitor for 30 minutes
  %(prog)s --calibrate               # Calibration mode
  %(prog)s --create-config           # Generate sample config files
        """
    )
    
    parser.add_argument('--simulation', action='store_true',
                       help='Run in simulation mode (no hardware required)')
    parser.add_argument('--duration', type=float, metavar='MINUTES',
                       help='Monitor duration in minutes (unlimited if not specified)')
    parser.add_argument('--calibrate', action='store_true',
                       help='Run interactive calibration mode')
    parser.add_argument('--create-config', action='store_true',
                       help='Create sample configuration files')
    parser.add_argument('--refresh-rate', type=float, metavar='HZ',
                       help='Override default refresh rate (Hz)')
    
    args = parser.parse_args()
    
    # Handle command line options
    if args.create_config:
        create_sample_aws_config()
        create_sample_testbed_config()
        return
    
    if args.calibrate:
        calibrate_sensor_interactive(simulation_mode=args.simulation)
        return
    
    # Override refresh rate if specified
    if args.refresh_rate:
        ThermalConfig.REFRESH_RATE_HZ = args.refresh_rate
        ThermalConfig.READ_INTERVAL = 1.0 / args.refresh_rate
        logger.info(f"🔄 Custom refresh rate: {args.refresh_rate}Hz")
    
    # Direct monitoring mode
    if args.simulation is not None or args.duration is not None:
        monitor = ProductionThermalMonitor(simulation_mode=args.simulation)
        monitor.run_monitoring(duration_minutes=args.duration)
        return
    
    # === INTERACTIVE MENU MODE ===
    print("🌡️  === Production Thermal Monitor with IoT Integration ===")
    print("1. Start Monitoring (Real Hardware)")
    print("2. Start Monitoring (Simulation Mode)")  
    print("3. Calibrate Sensor")
    print("4. Test AWS IoT Connection")
    print("5. Test Testbed Integration")
    print("6. Create Sample Config Files")
    print("7. Exit")

    try:
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == "1":
            if not _HW_AVAILABLE:
                print("❌ Hardware libraries not available. Use simulation mode or install dependencies.")
                return
            duration = input("Duration in minutes (press Enter for unlimited): ").strip()
            duration = float(duration) if duration else None
            monitor = ProductionThermalMonitor(simulation_mode=False)
            monitor.run_monitoring(duration_minutes=duration)

        elif choice == "2":
            duration = input("Duration in minutes (press Enter for unlimited): ").strip()
            duration = float(duration) if duration else None
            monitor = ProductionThermalMonitor(simulation_mode=True)
            monitor.run_monitoring(duration_minutes=duration)

        elif choice == "3":
            mock = input("Use simulation mode for calibration? (y/n): ").strip().lower() == 'y'
            calibrate_sensor_interactive(simulation_mode=mock)

        elif choice == "4":
            # Test AWS IoT connection
            publisher = AWSIoTPublisher()
            if publisher.connect():
                print("✅ AWS IoT Core connection successful")
                test_payload = {
                    'test': True,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'device_id': 'thermal-camera-01'
                }
                publisher.publish_data(test_payload)
                publisher.disconnect()
            else:
                print("❌ AWS IoT Core connection failed")

        elif choice == "5":
            # Test testbed integration  
            testbed = TestbedIntegration()
            if testbed.connect_accuenergy():
                print("✅ Accuenergy meter connection successful")
                power_data = testbed.read_power_data()
                if power_data:
                    print(f"⚡ Power reading: {power_data['active_power']}W, {power_data['voltage']}V")
                else:
                    print("⚠️ Power data read failed")
            else:
                print("❌ Testbed connection failed")

        elif choice == "6":
            create_sample_aws_config()
            create_sample_testbed_config()

        elif choice == "7":
            print("👋 Exiting")
            return

        else:
            print("❌ Invalid choice")

    except KeyboardInterrupt:
        print("\n👋 Exiting")
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")


if __name__ == '__main__':
    main()