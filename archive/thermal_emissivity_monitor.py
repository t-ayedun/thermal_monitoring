# thermal_monitor_with_emissivity.py - Thermal monitor with emissivity correction
import time
import json
import csv
import numpy as np
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Optional, Dict, List

# Import the emissivity system
from emissivity_calibration import EmissivityDatabase, EmissivityCorrector, MLX90640EmissivityCalibrator
from config_manager import ThermalConfig
from mock_mlx90640 import MockMLX90640

class EmissivityAwareThermalMonitor:
    """Enhanced thermal monitor with emissivity correction capabilities"""
    
    def __init__(self, config_file: str = 'thermal_config.yaml', mock_mode: Optional[bool] = None):
        # Load configuration
        self.config = ThermalConfig(config_file)
        
        # Initialize emissivity system
        self.emissivity_db = EmissivityDatabase()
        self.emissivity_corrector = EmissivityCorrector()
        
        # Determine sensor mode
        if mock_mode is None:
            mock_mode = self.config.get('sensor.type') == 'mock'
        self.mock_mode = mock_mode
        
        # Current material being monitored
        self.current_material = self.config.get('monitoring.target_material', 'transformer_oil')
        self.current_emissivity = self.emissivity_db.get_emissivity(self.current_material)
        
        # Initialize system
        self.setup_logging()
        self.temperature_buffer = []
        self.initialize_sensor()
        
        # Performance tracking
        self.performance_stats = {
            'total_readings': 0,
            'corrected_readings': 0,
            'emissivity_corrections_applied': 0
        }
        
        self.logger.info(f"🎯 Target material: {self.current_material} (ε = {self.current_emissivity})")
    
    def setup_logging(self):
        """Setup logging system"""
        log_path = self.config.get_file_path('log')
        log_path.parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('EmissivityThermalMonitor')
    
    def initialize_sensor(self):
        """Initialize thermal sensor"""
        if self.mock_mode:
            self.sensor = MockMLX90640()
            scenario = self.config.get('sensor.mock_scenario', 'normal')
            self.sensor.set_scenario(scenario)
            self.logger.info(f"🔧 Mock sensor initialized with scenario: {scenario}")
        else:
            try:
                import board
                import busio
                import adafruit_mlx90640
                
                i2c_freq = self.config.get('monitoring.i2c_frequency', 400000)
                i2c = busio.I2C(board.SCL, board.SDA, frequency=i2c_freq)
                self.sensor = adafruit_mlx90640.MLX90640(i2c)
                
                refresh_setting = self.config.get('sensor.refresh_rate_setting', 'REFRESH_1_HZ')
                refresh_rate = getattr(adafruit_mlx90640.RefreshRate, refresh_setting)
                self.sensor.refresh_rate = refresh_rate
                
                self.logger.info("🌡️ Real MLX90640 sensor initialized")
                self.warm_up_sensor()
                
            except ImportError as e:
                self.logger.error(f"MLX90640 libraries not found: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Failed to initialize MLX90640: {e}")
                raise
    
    def warm_up_sensor(self):
        """Warm up sensor"""
        if self.mock_mode:
            return
        
        warm_up_readings = self.config.get('monitoring.warm_up_readings', 5)
        self.logger.info(f"🔥 Warming up sensor with {warm_up_readings} readings...")
        
        for i in range(warm_up_readings):
            try:
                frame = [0] * 768
                self.sensor.getFrame(frame)
                time.sleep(2)
            except Exception as e:
                self.logger.warning(f"Warm-up reading {i+1} failed: {e}")
        
        self.logger.info("✅ Sensor warm-up completed")
    
    def set_target_material(self, material: str) -> bool:
        """Set the target material for emissivity correction"""
        if material in self.emissivity_db.EMISSIVITY_TABLE:
            self.current_material = material
            self.current_emissivity = self.emissivity_db.get_emissivity(material)
            self.logger.info(f"🎯 Target material set to: {material} (ε = {self.current_emissivity})")
            
            # Update config
            self.config.set('monitoring.target_material', material)
            return True
        else:
            # Try to find similar materials
            similar = self.emissivity_db.find_similar_materials(material)
            if similar:
                self.logger.warning(f"❌ Material '{material}' not found. Similar: {similar[:3]}")
            else:
                self.logger.warning(f"❌ Material '{material}' not found in database")
            return False
    
    def read_thermal_frame_with_correction(self) -> Optional[Dict]:
        """Read thermal frame and apply emissivity correction"""
        try:
            # Get raw frame
            if self.mock_mode:
                frame = self.sensor.getFrame()
            else:
                frame = [0] * 768
                self.sensor.getFrame(frame)
            
            # Validate data
            valid_frame = self.validate_temperature_data(frame)
            if not valid_frame:
                return None
            
            # Apply emissivity correction
            corrected_frame = self.apply_emissivity_correction(valid_frame)
            
            # Calculate statistics
            raw_stats = self.calculate_frame_statistics(valid_frame, 'raw')
            corrected_stats = self.calculate_frame_statistics(corrected_frame, 'corrected')
            
            # Combine results
            result = {
                'timestamp': datetime.now().isoformat(),
                'material': self.current_material,
                'emissivity': self.current_emissivity,
                'raw': raw_stats,
                'corrected': corrected_stats,
                'correction_applied': True,
                'valid_pixels': len(valid_frame)
            }
            
            self.performance_stats['corrected_readings'] += 1
            self.performance_stats['emissivity_corrections_applied'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error reading thermal frame: {e}")
            return None
    
    def validate_temperature_data(self, frame: List[float]) -> Optional[List[float]]:
        """Validate temperature data"""
        min_valid = self.config.get('temperature_thresholds.min_valid', -10.0)
        max_valid = self.config.get('temperature_thresholds.max_valid', 150.0)
        min_valid_percent = self.config.get('data_quality.min_valid_pixels_percent', 78)
        
        valid_frame = []
        for temp in frame:
            if min_valid <= temp <= max_valid:
                valid_frame.append(temp)
        
        # Check if we have enough valid data
        valid_percentage = (len(valid_frame) / len(frame)) * 100
        if valid_percentage < min_valid_percent:
            self.logger.warning(f"Only {valid_percentage:.1f}% valid pixels")
            return None
        
        return valid_frame
    
    def apply_emissivity_correction(self, frame: List[float]) -> List[float]:
        """Apply emissivity correction to thermal frame"""
        corrected_frame = []
        
        for temp in frame:
            # Apply simple emissivity correction
            corrected_temp = self.emissivity_corrector.simple_emissivity_correction(
                measured_temp=temp,
                actual_emissivity=self.current_emissivity,
                assumed_emissivity=0.95  # MLX90640 default assumption
            )
            corrected_frame.append(corrected_temp)
        
        return corrected_frame
    
    def calculate_frame_statistics(self, frame: List[float], data_type: str) -> Dict:
        """Calculate statistics for a thermal frame"""
        return {
            'min_temp': float(min(frame)),
            'max_temp': float(max(frame)),
            'avg_temp': float(np.mean(frame)),
            'std_temp': float(np.std(frame)),
            'median_temp': float(np.median(frame)),
            'data_type': data_type
        }
    
    def check_temperature_alerts(self, temps: Dict) -> str:
        """Check temperature alerts using corrected data"""
        # Use corrected temperatures for alert logic
        corrected_data = temps['corrected']
        max_temp = corrected_data['max_temp']
        avg_temp = corrected_data['avg_temp']
        
        # Get thresholds
        temp_warning = self.config.get('temperature_thresholds.warning', 65.0)
        temp_critical = self.config.get('temperature_thresholds.critical', 85.0)
        temp_emergency = self.config.get('temperature_thresholds.emergency', 105.0)
        temp_elevated = self.config.get('temperature_thresholds.elevated', 45.0)
        
        # Multi-level alert system based on corrected temperatures
        if max_temp > temp_emergency:
            self.logger.critical(f"🚨 EMERGENCY: Corrected max temp {max_temp:.1f}°C - IMMEDIATE ACTION REQUIRED!")
            return "EMERGENCY"
        elif max_temp > temp_critical:
            self.logger.critical(f"🔴 CRITICAL: Corrected max temp {max_temp:.1f}°C, avg {avg_temp:.1f}°C")
            return "CRITICAL"
        elif max_temp > temp_warning:
            self.logger.warning(f"⚠️ WARNING: Corrected max temp {max_temp:.1f}°C, avg {avg_temp:.1f}°C")
            return "WARNING"
        elif avg_temp > temp_elevated:
            self.logger.info(f"📈 ELEVATED: Corrected max temp {max_temp:.1f}°C, avg {avg_temp:.1f}°C")
            return "ELEVATED"
        else:
            self.logger.info(f"✅ NORMAL: Corrected max temp {max_temp:.1f}°C, avg {avg_temp:.1f}°C")
            return "NORMAL"
    
    def log_data_with_emissivity(self, temps: Dict):
        """Enhanced logging with emissivity correction data"""
        try:
            data_file = self.config.get_file_path('data')
            file_exists = data_file.exists()
            
            with open(data_file, 'a', newline='') as csvfile:
                fieldnames = [
                    'timestamp', 'material', 'emissivity',
                    'raw_min', 'raw_max', 'raw_avg', 'raw_std',
                    'corrected_min', 'corrected_max', 'corrected_avg', 'corrected_std',
                    'correction_delta', 'valid_pixels', 'status'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                    self.logger.info(f"📝 Created thermal data log with emissivity tracking")
                
                status = self.check_temperature_alerts(temps)
                raw_data = temps['raw']
                corrected_data = temps['corrected']
                
                # Calculate correction delta
                correction_delta = corrected_data['max_temp'] - raw_data['max_temp']
                
                writer.writerow({
                    'timestamp': temps['timestamp'],
                    'material': temps['material'],
                    'emissivity': f"{temps['emissivity']:.3f}",
                    'raw_min': f"{raw_data['min_temp']:.2f}",
                    'raw_max': f"{raw_data['max_temp']:.2f}",
                    'raw_avg': f"{raw_data['avg_temp']:.2f}",
                    'raw_std': f"{raw_data['std_temp']:.2f}",
                    'corrected_min': f"{corrected_data['min_temp']:.2f}",
                    'corrected_max': f"{corrected_data['max_temp']:.2f}",
                    'corrected_avg': f"{corrected_data['avg_temp']:.2f}",
                    'corrected_std': f"{corrected_data['std_temp']:.2f}",
                    'correction_delta': f"{correction_delta:.2f}",
                    'valid_pixels': temps['valid_pixels'],
                    'status': status
                })
                
        except Exception as e:
            self.logger.error(f"Error logging data: {e}")
    
    def display_live_stats_with_emissivity(self, temps: Dict):
        """Enhanced live display showing both raw and corrected temperatures"""
        status = self.check_temperature_alerts(temps)
        status_icon = {
            'NORMAL': '✅',
            'ELEVATED': '📈', 
            'WARNING': '⚠️',
            'CRITICAL': '🔴',
            'EMERGENCY': '🚨'
        }.get(status, '❓')
        
        raw_data = temps['raw']
        corrected_data = temps['corrected']
        correction_delta = corrected_data['max_temp'] - raw_data['max_temp']
        
        print(f"\r{status_icon} {temps['material']} (ε={temps['emissivity']:.2f}) | " +
              f"Raw: {raw_data['max_temp']:.1f}°C | " +
              f"Corrected: {corrected_data['max_temp']:.1f}°C | " +
              f"Δ: {correction_delta:+.1f}°C | " +
              f"Time: {datetime.now().strftime('%H:%M:%S')}", end='', flush=True)
    
    def run_emissivity_aware_monitoring(self, duration_minutes: Optional[float] = None):
        """Main monitoring loop with emissivity correction"""
        self.logger.info("🚀 Starting emissivity-aware thermal monitoring...")
        self.logger.info(f"🎯 Target material: {self.current_material} (emissivity: {self.current_emissivity})")
        self.logger.info(f"📊 Data logging interval: {self.config.get('monitoring.log_interval')} seconds")
        
        last_log_time = 0
        start_time = time.time()
        reading_count = 0
        error_count = 0
        
        try:
            while True:
                current_time = time.time()
                
                # Read thermal data with emissivity correction
                temps = self.read_thermal_frame_with_correction()
                if temps:
                    reading_count += 1
                    error_count = 0
                    
                    # Display live stats
                    self.display_live_stats_with_emissivity(temps)
                    
                    # Log to file at specified interval
                    log_interval = self.config.get('monitoring.log_interval', 300)
                    if current_time - last_log_time >= log_interval:
                        print()  # New line after live stats
                        self.log_data_with_emissivity(temps)
                        last_log_time = current_time
                        
                        # Show correction summary
                        raw_max = temps['raw']['max_temp']
                        corrected_max = temps['corrected']['max_temp']
                        delta = corrected_max - raw_max
                        self.logger.info(f"📊 Correction: {raw_max:.1f}°C → {corrected_max:.1f}°C (Δ{delta:+.1f}°C)")
                else:
                    error_count += 1
                    max_errors = self.config.get('monitoring.max_consecutive_errors', 10)
                    if error_count > max_errors:
                        self.logger.error("Multiple consecutive sensor reading failures")
                        break
                
                # Exit condition for testing
                if duration_minutes and (current_time - start_time) > (duration_minutes * 60):
                    print(f"\n⏰ Monitoring completed after {duration_minutes} minutes")
                    break
                
                # Sleep based on refresh rate
                refresh_rate = self.config.get('monitoring.refresh_rate', 1.0)
                time.sleep(1.0 / refresh_rate)
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Monitoring stopped by user after {reading_count} readings")
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            print(f"\n💥 Error during monitoring: {e}")
            self.logger.error(f"Monitoring error: {e}")
        
        # Final summary
        total_time = time.time() - start_time
        avg_rate = reading_count / total_time if total_time > 0 else 0
        
        print(f"\n📊 SESSION SUMMARY")
        print(f"   Duration: {total_time:.1f} seconds")
        print(f"   Total readings: {reading_count}")
        print(f"   Average rate: {avg_rate:.1f} readings/second")
        print(f"   Emissivity corrections: {self.performance_stats['emissivity_corrections_applied']}")
        print(f"   Target material: {self.current_material} (ε = {self.current_emissivity})")
    
    def calibration_mode(self):
        """Run calibration mode for emissivity correction"""
        calibrator = MLX90640EmissivityCalibrator(self)
        
        print("\n🎯 EMISSIVITY CALIBRATION MODE")
        print("=" * 50)
        print("Choose calibration type:")
        print("1. Quick test with black tape")
        print("2. Full guided calibration")
        print("3. Material database lookup")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            calibrator.quick_calibration_test()
        elif choice == "2":
            calibration_results = calibrator.guided_emissivity_calibration()
            if calibration_results:
                print(f"\n✅ Calibration completed for {len(calibration_results)} materials")
        elif choice == "3":
            self.material_database_interface()
        else:
            print("❌ Invalid choice")
    
    def material_database_interface(self):
        """Interactive material database interface"""
        print(f"\n📚 MATERIAL DATABASE INTERFACE")
        print("=" * 40)
        
        while True:
            print(f"\nOptions:")
            print(f"1. Search materials")
            print(f"2. Show transformer materials")
            print(f"3. Set target material")
            print(f"4. Show current material")
            print(f"5. Exit")
            
            choice = input("Enter choice (1-5): ").strip()
            
            if choice == "1":
                search_term = input("Enter search term: ").strip()
                if search_term:
                    matches = self.emissivity_db.find_similar_materials(search_term)
                    if matches:
                        print(f"\n🔍 Found {len(matches)} matches:")
                        for i, material in enumerate(matches[:10], 1):
                            emissivity = self.emissivity_db.get_emissivity(material)
                            print(f"   {i:2d}. {material}: ε = {emissivity}")
                    else:
                        print("❌ No matches found")
            
            elif choice == "2":
                print(f"\n🔌 Transformer-related materials:")
                materials = ['transformer_oil', 'transformer_steel_core', 'copper_windings_new', 
                           'copper_windings_aged', 'aluminum_windings', 'porcelain_insulator']
                for material in materials:
                    emissivity = self.emissivity_db.get_emissivity(material)
                    print(f"   {material}: ε = {emissivity}")
            
            elif choice == "3":
                material = input("Enter material name: ").strip()
                if self.set_target_material(material):
                    print(f"✅ Target material set to: {material}")
                else:
                    print(f"❌ Material not found")
            
            elif choice == "4":
                print(f"Current material: {self.current_material} (ε = {self.current_emissivity})")
            
            elif choice == "5":
                break
            
            else:
                print("❌ Invalid choice")
    
    def accuracy_comparison_mode(self):
        """Mode for comparing MLX90640 accuracy with reference sensor"""
        print(f"\n🎯 ACCURACY COMPARISON MODE")
        print("=" * 40)
        print("This mode helps you compare MLX90640 readings with a reference sensor")
        print("(e.g., thermocouple, RTD, contact thermometer)")
        
        # Get reference sensor reading
        try:
            ref_temp = float(input("\nEnter reference sensor reading (°C): "))
        except ValueError:
            print("❌ Invalid temperature")
            return
        
        # Take MLX90640 readings
        print(f"\nTaking MLX90640 measurements...")
        readings = []
        
        for i in range(10):
            temps = self.read_thermal_frame_with_correction()
            if temps:
                # Use the corrected max temperature
                reading = temps['corrected']['max_temp']
                readings.append(reading)
                print(f"Reading {i+1}/10: {reading:.2f}°C")
            time.sleep(1)
        
        if not readings:
            print("❌ No valid readings obtained")
            return
        
        # Analysis
        mlx_avg = np.mean(readings)
        mlx_std = np.std(readings)
        difference = mlx_avg - ref_temp
        percent_error = (difference / ref_temp) * 100 if ref_temp != 0 else 0
        
        print(f"\n📊 ACCURACY ANALYSIS:")
        print(f"   Reference temperature: {ref_temp:.2f}°C")
        print(f"   MLX90640 average: {mlx_avg:.2f}°C")
        print(f"   Standard deviation: {mlx_std:.2f}°C")
        print(f"   Difference: {difference:+.2f}°C")
        print(f"   Percent error: {percent_error:+.1f}%")
        print(f"   Material: {self.current_material} (ε = {self.current_emissivity})")
        
        # Interpretation
        if abs(difference) < 2.0:
            print(f"✅ EXCELLENT: Very good agreement")
        elif abs(difference) < 5.0:
            print(f"✅ GOOD: Acceptable accuracy for most applications")
        elif abs(difference) < 10.0:
            print(f"⚠️ FAIR: Consider recalibration or emissivity adjustment")
        else:
            print(f"❌ POOR: Check setup, emissivity, or sensor calibration")
        
        # Suggestions
        if abs(difference) > 3.0:
            print(f"\n💡 SUGGESTIONS:")
            print(f"   • Verify target material emissivity")
            print(f"   • Check for reflected radiation sources")
            print(f"   • Ensure thermal equilibrium")
            print(f"   • Consider ambient temperature correction")

def main():
    """Enhanced main function with emissivity features"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MLX90640 Thermal Monitor with Emissivity Correction')
    parser.add_argument('--config', default='thermal_config.yaml', help='Configuration file')
    parser.add_argument('--mock', action='store_true', help='Use mock sensor')
    parser.add_argument('--material', help='Set target material')
    parser.add_argument('--duration', type=float, help='Monitoring duration in minutes')
    parser.add_argument('--mode', choices=['monitor', 'calibrate', 'accuracy', 'materials'], 
                       default='monitor', help='Operation mode')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = EmissivityAwareThermalMonitor(config_file=args.config, mock_mode=args.mock)
    
    # Set target material if specified
    if args.material:
        if not monitor.set_target_material(args.material):
            print(f"❌ Failed to set material: {args.material}")
            return
    
    # Run specified mode
    if args.mode == 'monitor':
        monitor.run_emissivity_aware_monitoring(duration_minutes=args.duration)
    elif args.mode == 'calibrate':
        monitor.calibration_mode()
    elif args.mode == 'accuracy':
        monitor.accuracy_comparison_mode()
    elif args.mode == 'materials':
        monitor.material_database_interface()
    
    # Show data file location
    data_file = monitor.config.get_file_path('data')
    if data_file.exists():
        print(f"\n📄 Data saved to: {data_file}")

if __name__ == "__main__":
    main()