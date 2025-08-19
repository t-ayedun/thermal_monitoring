#!/usr/bin/env python3
"""
Enhanced MLX90640 Thermal Monitor with AWS IoT Core Integration
Supports 10-minute granularity data aggregation and MQTT publishing
"""

import time
import json
import logging
import statistics
from datetime import datetime, timezone
from collections import deque
import signal
import sys
import os
from pathlib import Path

# Core sensor and data processing
import board
import busio
import adafruit_mlx90640
import numpy as np
import pandas as pd

# MQTT and AWS IoT Core integration
import paho.mqtt.client as mqtt
import ssl
from awscrt import io, mqtt as aws_mqtt, auth, http
from awsiot import mqtt_connection_builder

class ThermalDataAggregator:
    """Handles data collection and aggregation for thermal readings"""
    
    def __init__(self, aggregation_interval=600):
        self.aggregation_interval = aggregation_interval  # seconds
        self.readings_buffer = deque()
        self.last_aggregation_time = time.time()
        
    def add_reading(self, min_temp, avg_temp, max_temp, timestamp):
        """Add a new temperature reading to the buffer"""
        self.readings_buffer.append({
            'min_temp': min_temp,
            'avg_temp': avg_temp,
            'max_temp': max_temp,
            'timestamp': timestamp
        })
        
        # Remove readings older than aggregation interval
        current_time = time.time()
        cutoff_time = current_time - self.aggregation_interval
        
        while self.readings_buffer and self.readings_buffer[0]['timestamp'] < cutoff_time:
            self.readings_buffer.popleft()
    
    def should_aggregate(self):
        """Check if it's time to create an aggregated reading"""
        current_time = time.time()
        return (current_time - self.last_aggregation_time) >= self.aggregation_interval
    
    def get_aggregated_data(self):
        """Calculate aggregated statistics from buffered readings"""
        if not self.readings_buffer:
            return None
            
        # Extract temperature arrays
        min_temps = [reading['min_temp'] for reading in self.readings_buffer]
        avg_temps = [reading['avg_temp'] for reading in self.readings_buffer]
        max_temps = [reading['max_temp'] for reading in self.readings_buffer]
        
        # Calculate aggregated statistics
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
                }
            }
        }
        
        self.last_aggregation_time = time.time()
        return aggregated

class AWSIoTPublisher:
    """Handles AWS IoT Core MQTT communication"""
    
    def __init__(self, config_file="aws_iot_config.json"):
        self.config = self.load_config(config_file)
        self.connection = None
        self.is_connected = False
        
    def load_config(self, config_file):
        """Load AWS IoT configuration from file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            required_fields = ['endpoint', 'cert_path', 'key_path', 'ca_path', 'client_id', 'topic']
            missing_fields = [field for field in required_fields if field not in config]
            
            if missing_fields:
                raise ValueError(f"Missing required config fields: {missing_fields}")
                
            return config
        except FileNotFoundError:
            logging.warning(f"AWS IoT config file not found: {config_file}")
            return None
        except Exception as e:
            logging.error(f"Error loading AWS IoT config: {e}")
            return None
    
    def connect(self):
        """Establish connection to AWS IoT Core"""
        if not self.config:
            logging.warning("AWS IoT not configured - skipping cloud integration")
            return False
            
        try:
            # Create event loop for AWS SDK
            event_loop_group = io.EventLoopGroup(1)
            host_resolver = io.DefaultHostResolver(event_loop_group)
            client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)
            
            # Build MQTT connection
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
            
            # Connect
            connect_future = self.connection.connect()
            connect_future.result()
            self.is_connected = True
            
            logging.info(f"Successfully connected to AWS IoT Core: {self.config['endpoint']}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to connect to AWS IoT Core: {e}")
            self.is_connected = False
            return False
    
    def publish_data(self, data):
        """Publish thermal data to AWS IoT Core"""
        if not self.is_connected or not self.connection:
            logging.warning("Not connected to AWS IoT Core - data not published")
            return False
            
        try:
            # Add device metadata
            message = {
                'device_id': self.config['client_id'],
                'sensor_type': 'MLX90640',
                'location': self.config.get('location', 'unknown'),
                'data': data
            }
            
            # Publish to AWS IoT
            publish_future = self.connection.publish(
                topic=self.config['topic'],
                payload=json.dumps(message),
                qos=aws_mqtt.QoS.AT_LEAST_ONCE
            )
            publish_future.result()
            
            logging.info(f"Published data to AWS IoT topic: {self.config['topic']}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to publish to AWS IoT Core: {e}")
            return False
    
    def disconnect(self):
        """Clean disconnection from AWS IoT Core"""
        if self.connection and self.is_connected:
            try:
                disconnect_future = self.connection.disconnect()
                disconnect_future.result()
                logging.info("Disconnected from AWS IoT Core")
            except Exception as e:
                logging.error(f"Error during AWS IoT disconnect: {e}")
            finally:
                self.is_connected = False

class TestbedIntegration:
    """Integration with Accuenergy meter and Teltonika gateway"""
    
    def __init__(self, config_file="testbed_config.json"):
        self.config = self.load_config(config_file)
        self.accuenergy_client = None
        self.teltonika_client = None
        
    def load_config(self, config_file):
        """Load testbed configuration"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning(f"Testbed config not found: {config_file}")
            return {
                'accuenergy': {'enabled': False},
                'teltonika': {'enabled': False}
            }
    
    def connect_accuenergy(self):
        """Connect to Accuenergy meter via Modbus/TCP"""
        if not self.config.get('accuenergy', {}).get('enabled', False):
            return False
            
        try:
            from pymodbus.client.sync import ModbusTcpClient
            
            meter_config = self.config['accuenergy']
            self.accuenergy_client = ModbusTcpClient(
                host=meter_config['host'],
                port=meter_config.get('port', 502),
                timeout=meter_config.get('timeout', 3)
            )
            
            if self.accuenergy_client.connect():
                logging.info("Connected to Accuenergy meter")
                return True
            else:
                logging.error("Failed to connect to Accuenergy meter")
                return False
                
        except Exception as e:
            logging.error(f"Accuenergy connection error: {e}")
            return False
    
    def read_power_data(self):
        """Read power consumption data from Accuenergy meter"""
        if not self.accuenergy_client or not self.accuenergy_client.is_socket_open():
            return None
            
        try:
            # Read power registers (adjust addresses based on your meter model)
            result = self.accuenergy_client.read_holding_registers(
                address=1000,  # Adjust based on Accuenergy register map
                count=4
            )
            
            if result.isError():
                logging.error("Error reading Accuenergy registers")
                return None
                
            return {
                'active_power': result.registers[0],
                'reactive_power': result.registers[1],
                'voltage': result.registers[2],
                'current': result.registers[3],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error reading power data: {e}")
            return None

class EnhancedThermalMonitor:
    """Main thermal monitoring system with AWS IoT and testbed integration"""
    
    def __init__(self, simulation_mode=False):
        self.simulation_mode = simulation_mode
        self.setup_logging()
        
        # Initialize components
        self.data_aggregator = ThermalDataAggregator(aggregation_interval=60)  # 1 minute
        self.aws_publisher = AWSIoTPublisher()
        self.testbed = TestbedIntegration()
        
        # Sensor setup
        self.mlx = None
        self.running = True
        
        # Data storage
        self.csv_file = f"thermal_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.init_csv_file()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def setup_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('thermal_monitor.log'),
                logging.StreamHandler()
            ]
        )
        
    def init_csv_file(self):
        """Initialize CSV file with headers"""
        headers = [
            'timestamp', 'min_temp', 'avg_temp', 'max_temp', 
            'sample_count', 'std_dev', 'power_data_available'
        ]
        
        df = pd.DataFrame(columns=headers)
        df.to_csv(self.csv_file, index=False)
        logging.info(f"Initialized CSV file: {self.csv_file}")
    
    def initialize_sensor(self):
        """Initialize MLX90640 thermal sensor"""
        if self.simulation_mode:
            logging.info("Running in SIMULATION MODE - no hardware required")
            return True
            
        try:
            i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
            self.mlx = adafruit_mlx90640.MLX90640(i2c)
            self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
            
            logging.info("MLX90640 thermal sensor initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize thermal sensor: {e}")
            return False
    
    def read_thermal_data(self):
        """Read and process thermal sensor data"""
        if self.simulation_mode:
            # Generate realistic simulation data
            base_temp = 25.0 + np.random.normal(0, 2)
            thermal_data = np.random.normal(base_temp, 5, 768)  # 32x24 pixels
        else:
            try:
                thermal_data = np.zeros((24 * 32,))
                self.mlx.getFrame(thermal_data)
            except Exception as e:
                logging.error(f"Error reading thermal data: {e}")
                return None
        
        # Calculate statistics
        min_temp = float(np.min(thermal_data))
        avg_temp = float(np.mean(thermal_data))
        max_temp = float(np.max(thermal_data))
        std_dev = float(np.std(thermal_data))
        
        timestamp = time.time()
        
        return {
            'min_temp': min_temp,
            'avg_temp': avg_temp,
            'max_temp': max_temp,
            'std_dev': std_dev,
            'timestamp': timestamp,
            'iso_timestamp': datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
        }
    
    def save_to_csv(self, data, power_data=None):
        """Save aggregated data to CSV file"""
        try:
            row_data = {
                'timestamp': data['timestamp'],
                'min_temp': data['temperature_stats']['min_temp']['value'],
                'avg_temp': data['temperature_stats']['avg_temp']['value'],
                'max_temp': data['temperature_stats']['max_temp']['value'],
                'sample_count': data['sample_count'],
                'std_dev': data['temperature_stats']['avg_temp']['std_dev'],
                'power_data_available': power_data is not None
            }
            
            df = pd.DataFrame([row_data])
            df.to_csv(self.csv_file, mode='a', header=False, index=False)
            
        except Exception as e:
            logging.error(f"Error saving to CSV: {e}")
    
    def create_aws_payload(self, thermal_data, power_data=None):
        """Create standardized payload for AWS IoT Core"""
        payload = {
            'device_info': {
                'device_id': 'thermal-camera-01',
                'sensor_model': 'MLX90640',
                'firmware_version': '1.0.0',
                'location': 'office_testbed'
            },
            'thermal_data': thermal_data,
            'metadata': {
                'data_quality': 'good' if thermal_data['sample_count'] > 10 else 'limited',
                'aggregation_method': 'statistical_summary',
                'raw_samples': thermal_data['sample_count']
            }
        }
        
        # Add power data if available
        if power_data:
            payload['power_data'] = power_data
            payload['metadata']['power_integration'] = True
        
        return payload
    
    def signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully"""
        logging.info(f"Received signal {sig}, shutting down gracefully...")
        self.running = False
    
    def run_monitoring_loop(self):
        """Main monitoring loop with 1-minute aggregation"""
        if not self.initialize_sensor():
            logging.error("Failed to initialize sensor - exiting")
            return
        
        # Connect to AWS IoT Core
        aws_connected = self.aws_publisher.connect()
        if aws_connected:
            logging.info("AWS IoT Core integration active")
        else:
            logging.warning("Continuing without AWS IoT Core integration")
        
        # Connect to testbed infrastructure
        testbed_connected = self.testbed.connect_accuenergy()
        if testbed_connected:
            logging.info("Testbed integration active")
        
        logging.info("Starting thermal monitoring with 10-minute aggregation...")
        
        try:
            while self.running:
                # Read thermal sensor (every 2 seconds as per refresh rate)
                thermal_reading = self.read_thermal_data()
                if thermal_reading:
                    self.data_aggregator.add_reading(
                        thermal_reading['min_temp'],
                        thermal_reading['avg_temp'],
                        thermal_reading['max_temp'],
                        thermal_reading['timestamp']
                    )
                    
                    # Log individual reading (optional, for debugging)
                    logging.debug(f"Thermal reading: Avg={thermal_reading['avg_temp']:.2f}°C, "
                                f"Range=[{thermal_reading['min_temp']:.2f}, {thermal_reading['max_temp']:.2f}]")
                
                # Check if it's time for 10-minute aggregation
                if self.data_aggregator.should_aggregate():
                    aggregated_data = self.data_aggregator.get_aggregated_data()
                    
                    if aggregated_data:
                        # Read power data from testbed if available
                        power_data = None
                        if testbed_connected:
                            power_data = self.testbed.read_power_data()
                        
                        # Save to local CSV
                        self.save_to_csv(aggregated_data, power_data)
                        
                        # Publish to AWS IoT Core
                        if aws_connected:
                            aws_payload = self.create_aws_payload(aggregated_data, power_data)
                            self.aws_publisher.publish_data(aws_payload)
                        
                        # Log aggregated results
                        avg_temp = aggregated_data['temperature_stats']['avg_temp']['value']
                        sample_count = aggregated_data['sample_count']
                        logging.info(f"10-minute summary: Avg temp={avg_temp:.2f}°C "
                                   f"(from {sample_count} samples)")
                
                # Wait before next reading (1-second interval)
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            logging.info("Monitoring stopped by user")
        except Exception as e:
            logging.error(f"Unexpected error in monitoring loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean shutdown procedures"""
        logging.info("Performing cleanup...")
        
        # Process any remaining data
        if self.data_aggregator.readings_buffer:
            final_data = self.data_aggregator.get_aggregated_data()
            if final_data:
                self.save_to_csv(final_data)
                if self.aws_publisher.is_connected:
                    aws_payload = self.create_aws_payload(final_data)
                    self.aws_publisher.publish_data(aws_payload)
        
        # Disconnect from AWS IoT Core
        self.aws_publisher.disconnect()
        
        logging.info("Cleanup completed")

def create_sample_aws_config():
    """Create sample AWS IoT configuration file"""
    sample_config = {
        "endpoint": "your-iot-endpoint.amazonaws.com",
        "cert_path": "/home/pi/thermal_project/certs/device-certificate.pem.crt",
        "key_path": "/home/pi/thermal_project/certs/private.pem.key",
        "ca_path": "/home/pi/thermal_project/certs/Amazon-root-CA-1.pem",
        "client_id": "thermal-camera-01",
        "topic": "thermal/data",
        "location": "office_testbed"
    }
    
    with open("aws_iot_config_sample.json", 'w') as f:
        json.dump(sample_config, indent=2, fp=f)
    
    print("Sample AWS IoT config created: aws_iot_config_sample.json")
    print("Copy to aws_iot_config.json and update with your actual values")

def create_sample_testbed_config():
    """Create sample testbed configuration file"""
    sample_config = {
        "accuenergy": {
            "enabled": True,
            "host": "192.168.1.100",
            "port": 502,
            "timeout": 3,
            "unit_id": 1
        },
        "teltonika": {
            "enabled": True,
            "host": "192.168.1.101",
            "api_endpoint": "/api/data",
            "username": "admin",
            "password": "your_password"
        }
    }
    
    with open("testbed_config_sample.json", 'w') as f:
        json.dump(sample_config, indent=2, fp=f)
    
    print("Sample testbed config created: testbed_config_sample.json")
    print("Copy to testbed_config.json and update with your actual values")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Thermal Monitor with AWS IoT Integration')
    parser.add_argument('--simulation', action='store_true', 
                       help='Run in simulation mode (no hardware required)')
    parser.add_argument('--create-config', action='store_true',
                       help='Create sample configuration files')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_aws_config()
        create_sample_testbed_config()
        sys.exit(0)
    
    # Run the monitoring system
    monitor = EnhancedThermalMonitor(simulation_mode=args.simulation)
    monitor.run_monitoring_loop()