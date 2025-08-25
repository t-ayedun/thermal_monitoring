#!/usr/bin/env python3
"""
MLX90640 Temperature Monitoring System with MQTT Integration
Live demo script for transformer temperature monitoring
"""

import time
import board
import busio
import json
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import paho.mqtt.client as mqtt
import threading
import queue

# MLX90640 imports
try:
    import adafruit_mlx90640
    MLX_AVAILABLE = True
except ImportError:
    print("MLX90640 library not available - using simulation mode")
    MLX_AVAILABLE = False

class TemperatureMonitor:
    def __init__(self, mqtt_broker="localhost", mqtt_port=1883):
        # MQTT Configuration
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_disconnect = self.on_mqtt_disconnect
        
        # Data storage
        self.temperature_data = []
        self.data_queue = queue.Queue()
        self.running = True
        
        # MLX90640 setup
        if MLX_AVAILABLE:
            self.i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
            self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
            self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
        
        # CSV file setup
        self.csv_filename = f"transformer_temps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.init_csv_file()
        
        # Temperature thresholds (Celsius)
        self.temp_warning = 70.0
        self.temp_critical = 85.0
        
    def init_csv_file(self):
        """Initialize CSV file with headers"""
        with open(self.csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'min_temp', 'max_temp', 'avg_temp', 
                         'hotspot_x', 'hotspot_y', 'status', 'raw_data']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            print(f"✅ Connected to MQTT broker at {self.mqtt_broker}")
        else:
            print(f"❌ Failed to connect to MQTT broker. Code: {rc}")
    
    def on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        print(f"⚠️  Disconnected from MQTT broker")
    
    def connect_mqtt(self):
        """Connect to MQTT broker"""
        try:
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
            return True
        except Exception as e:
            print(f"MQTT connection error: {e}")
            return False
    
    def get_temperature_frame(self):
        """Get temperature data from MLX90640 or simulate if not available"""
        if MLX_AVAILABLE:
            frame = [0] * 768  # 32x24 pixels
            try:
                self.mlx.getFrame(frame)
                return np.array(frame)
            except Exception as e:
                print(f"Sensor read error: {e}")
                return self.simulate_temperature_frame()
        else:
            return self.simulate_temperature_frame()
    
    def simulate_temperature_frame(self):
        """Simulate temperature data for demo purposes"""
        # Create realistic transformer temperature simulation
        base_temp = 45.0  # Ambient temperature
        hotspot_temp = 75.0 + np.random.normal(0, 5)  # Variable hotspot
        
        # Create 32x24 temperature array
        temps = np.full((24, 32), base_temp)
        
        # Add hotspots (simulate transformer components)
        # Main hotspot (transformer core area)
        center_y, center_x = 12, 16
        for y in range(max(0, center_y-3), min(24, center_y+3)):
            for x in range(max(0, center_x-4), min(32, center_x+4)):
                distance = np.sqrt((y-center_y)**2 + (x-center_x)**2)
                if distance < 4:
                    temps[y, x] = hotspot_temp - distance * 3
        
        # Secondary hotspot (cooling system)
        temps[5:8, 8:12] = base_temp + np.random.normal(15, 2)
        
        # Add noise
        temps += np.random.normal(0, 0.5, temps.shape)
        
        return temps.flatten()
    
    def analyze_temperature_data(self, frame_data):
        """Analyze temperature frame and extract key metrics"""
        temps = np.array(frame_data)
        
        min_temp = np.min(temps)
        max_temp = np.max(temps)
        avg_temp = np.mean(temps)
        
        # Find hotspot location
        hotspot_idx = np.argmax(temps)
        hotspot_y = hotspot_idx // 32
        hotspot_x = hotspot_idx % 32
        
        # Determine status
        if max_temp >= self.temp_critical:
            status = "CRITICAL"
        elif max_temp >= self.temp_warning:
            status = "WARNING"
        else:
            status = "NORMAL"
        
        return {
            'min_temp': round(min_temp, 2),
            'max_temp': round(max_temp, 2),
            'avg_temp': round(avg_temp, 2),
            'hotspot_x': hotspot_x,
            'hotspot_y': hotspot_y,
            'status': status,
            'raw_data': temps.tolist()
        }
    
    def publish_mqtt_data(self, analysis):
        """Publish temperature data to MQTT topics"""
        timestamp = datetime.now().isoformat()
        
        # Main temperature topic (for dashboard)
        main_payload = {
            'timestamp': timestamp,
            'device_id': 'transformer_thermal_01',
            'location': 'Main Transformer',
            'temperatures': {
                'min': analysis['min_temp'],
                'max': analysis['max_temp'],
                'avg': analysis['avg_temp']
            },
            'hotspot': {
                'x': analysis['hotspot_x'],
                'y': analysis['hotspot_y'],
                'temp': analysis['max_temp']
            },
            'status': analysis['status']
        }
        
        # Publish to different topics
        self.mqtt_client.publish("transformer/temperature/main", 
                               json.dumps(main_payload, default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else x))
        
        # Alert topic (only for warnings/critical)
        if analysis['status'] != 'NORMAL':
            alert_payload = {
                'timestamp': timestamp,
                'alert_level': analysis['status'],
                'temperature': analysis['max_temp'],
                'threshold_exceeded': self.temp_warning if analysis['status'] == 'WARNING' else self.temp_critical,
                'location': f"Position ({analysis['hotspot_x']}, {analysis['hotspot_y']})"
            }
            self.mqtt_client.publish("transformer/alerts", 
                                   json.dumps(alert_payload))
        
        # AWS IoT Core compatible format
        aws_payload = {
            'deviceId': 'thermal_sensor_01',
            'timestamp': int(time.time() * 1000),  # AWS expects milliseconds
            'temperature': analysis['max_temp'],
            'status': analysis['status'],
            'location': 'transformer_yard_a'
        }
        self.mqtt_client.publish("aws/transformer/telemetry", 
                               json.dumps(aws_payload))
    
    def save_to_csv(self, analysis):
        """Save analysis data to CSV file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.csv_filename, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'min_temp', 'max_temp', 'avg_temp', 
                         'hotspot_x', 'hotspot_y', 'status', 'raw_data']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writerow({
                'timestamp': timestamp,
                'min_temp': analysis['min_temp'],
                'max_temp': analysis['max_temp'],
                'avg_temp': analysis['avg_temp'],
                'hotspot_x': analysis['hotspot_x'],
                'hotspot_y': analysis['hotspot_y'],
                'status': analysis['status'],
                'raw_data': json.dumps(analysis['raw_data'])
            })
    
    def data_collection_loop(self):
        """Main data collection loop running in separate thread"""
        print("🌡️  Starting temperature monitoring...")
        
        while self.running:
            try:
                # Get temperature frame
                frame_data = self.get_temperature_frame()
                
                # Analyze data
                analysis = self.analyze_temperature_data(frame_data)
                
                # Add to queue for live display
                self.data_queue.put(analysis)
                
                # Publish to MQTT
                self.publish_mqtt_data(analysis)
                
                # Save to CSV
                self.save_to_csv(analysis)
                
                # Print live data
                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"[{timestamp}] Temp: {analysis['min_temp']}°C - {analysis['max_temp']}°C | "
                      f"Avg: {analysis['avg_temp']}°C | Status: {analysis['status']}")
                
                time.sleep(10)  # 10-second intervals for demo
                
            except KeyboardInterrupt:
                print("\n🛑 Monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in data collection: {e}")
                time.sleep(5)
    
    def start_monitoring(self):
        """Start the temperature monitoring system"""
        print("🚀 Initializing Temperature Monitoring System...")
        
        # Connect to MQTT
        if self.connect_mqtt():
            print("📡 MQTT connection established")
        else:
            print("⚠️  MQTT connection failed - continuing without MQTT")
        
        # Start data collection in separate thread
        data_thread = threading.Thread(target=self.data_collection_loop)
        data_thread.daemon = True
        data_thread.start()
        
        return data_thread
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.running = False
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()

def main():
    """Main function for demo"""
    print("=" * 60)
    print("🌡️  TRANSFORMER TEMPERATURE MONITORING SYSTEM")
    print("=" * 60)
    
    # Initialize monitor (change MQTT broker as needed)
    monitor = TemperatureMonitor(mqtt_broker="localhost")  # Use "test.mosquitto.org" for public broker
    
    try:
        # Start monitoring
        data_thread = monitor.start_monitoring()
        
        print("\n📊 System Status:")
        print("   • Temperature monitoring: Active")
        print("   • MQTT publishing: Active")
        print("   • Data logging: Active")
        print("   • Press Ctrl+C to stop")
        print("\n" + "─" * 60)
        
        # Keep main thread alive
        data_thread.join()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down monitoring system...")
        monitor.stop_monitoring()
        print("✅ System stopped successfully")

if __name__ == "__main__":
    main()