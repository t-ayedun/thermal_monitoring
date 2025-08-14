# setup_emissivity_system.py - Easy setup for MLX90640 emissivity system
import os
import sys
import json
from pathlib import Path

def create_directory_structure():
    """Create necessary directories"""
    directories = ['logs', 'data', 'backups', 'calibration']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def create_sample_config():
    """Create a sample configuration file"""
    config_content = """# MLX90640 Thermal Monitoring Configuration
monitoring:
  refresh_rate: 1.0        # Hz - readings per second
  log_interval: 300        # seconds - how often to log to file
  smoothing_window: 3      # number of readings to average
  target_material: "transformer_oil"  # default material to monitor
  warm_up_readings: 5      # sensor warm-up cycles
  max_consecutive_errors: 10
  i2c_frequency: 400000    # Hz for MLX90640

temperature_thresholds:
  warning: 65.0           # °C - Transformer loading concern
  critical: 85.0          # °C - Immediate attention required  
  emergency: 105.0        # °C - Emergency shutdown recommended
  elevated: 45.0          # °C - Above normal but acceptable
  min_valid: -10.0        # °C - Reject readings below this
  max_valid: 150.0        # °C - Reject readings above this

calibration:
  ambient_offset: 0.0          # °C - Calibration offset
  emissivity_correction: 1.0   # Multiplier for emissivity
  stability_threshold: 0.5     # °C std dev for stable readings
  calibration_samples: 10      # Number of readings for calibration

data_quality:
  min_valid_pixels_percent: 78    # % of pixels that must be valid
  max_invalid_readings_percent: 6  # % of invalid readings allowed

file_paths:
  data_file: "thermal_data.csv"
  log_file: "thermal_monitor.log"
  config_file: "thermal_config.yaml"
  logs_directory: "logs"
  data_directory: "data"
  backup_directory: "backups"

alerts:
  email_enabled: false
  sound_alerts: true
  log_full_frame_on_critical: true

sensor:
  type: "MLX90640"         # or "mock" for testing
  mock_scenario: "normal"   # for testing: normal, warning, critical
  refresh_rate_setting: "REFRESH_1_HZ"
"""
    
    with open('thermal_config.yaml', 'w') as f:
        f.write(config_content)
    
    print("📋 Created thermal_config.yaml")

def create_quick_start_script():
    """Create a quick start script"""
    script_content = '''#!/usr/bin/env python3
# quick_start.py - Quick start for MLX90640 emissivity monitoring

from thermal_monitor_with_emissivity import EmissivityAwareThermalMonitor

def main():
    print("🔥 MLX90640 Emissivity-Aware Thermal Monitor")
    print("=" * 50)
    
    # Choose mode
    print("\\nSelect mode:")
    print("1. Start monitoring (transformer oil)")
    print("2. Start monitoring (custom material)")
    print("3. Calibration mode")
    print("4. Accuracy comparison")
    print("5. Material database")
    
    choice = input("\\nEnter choice (1-5): ").strip()
    
    # Initialize monitor (use mock mode if no real sensor)
    try:
        monitor = EmissivityAwareThermalMonitor(mock_mode=False)
        print("✅ Real MLX90640 sensor initialized")
    except:
        print("🔧 Using mock sensor (no real MLX90640 detected)")
        monitor = EmissivityAwareThermalMonitor(mock_mode=True)
    
    if choice == "1":
        # Monitor transformer oil
        monitor.set_target_material('transformer_oil')
        monitor.run_emissivity_aware_monitoring()
    
    elif choice == "2":
        # Custom material monitoring
        material = input("Enter material name: ").strip()
        if monitor.set_target_material(material):
            monitor.run_emissivity_aware_monitoring()
        else:
            print("❌ Material not found")
    
    elif choice == "3":
        # Calibration mode
        monitor.calibration_mode()
    
    elif choice == "4":
        # Accuracy comparison
        monitor.accuracy_comparison_mode()
    
    elif choice == "5":
        # Material database
        monitor.material_database_interface()
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
'''
    
    with open('quick_start.py', 'w') as f:
        f.write(script_content)
    
    os.chmod('quick_start.py', 0o755)
    print("🚀 Created quick_start.py")

def create_calibration_guide():
    """Create a calibration guide"""
    guide_content = """# MLX90640 Emissivity Calibration Guide

## Quick Setup for Transformer Monitoring

### 1. Basic Accuracy Test
```bash
python quick_start.py
# Choose option 4 (Accuracy comparison)
```

### 2. Materials for Your Transformer Setup

**High Emissivity (Easy to measure accurately):**
- Transformer oil: ε = 0.94
- Black electrical tape: ε = 0.95 (good calibration reference)
- Aged copper windings: ε = 0.74

**Low Emissivity (Needs correction):**
- New copper windings: ε = 0.04
- Polished aluminum: ε = 0.05
- Stainless steel: ε = 0.17

### 3. Quick Calibration Steps

1. **Get a reference thermometer** (contact sensor)
   - Thermocouple probe
   - RTD sensor
   - Digital infrared thermometer (for comparison)

2. **Choose calibration targets:**
   - Black electrical tape (high emissivity reference)
   - Your actual transformer components

3. **Calibration procedure:**
   ```bash
   python thermal_monitor_with_emissivity.py --mode calibrate
   ```

### 4. Understanding Emissivity Effects

**High emissivity materials (ε > 0.8):**
- MLX90640 reads close to actual temperature
- Minimal correction needed
- Examples: Oil, painted surfaces, oxidized metals

**Low emissivity materials (ε < 0.3):**
- MLX90640 reads LOWER than actual temperature
- Significant correction needed
- Examples: Polished metals, new copper

**Correction formula:**
```
Actual_Temp = Measured_Temp × (Actual_Emissivity / 0.95)
```

### 5. Practical Tips

**For transformer monitoring:**
- Focus on oil temperature (high emissivity, accurate)
- Be careful with new copper windings (low emissivity)
- Use aged/oxidized metal surfaces when possible
- Consider applying high-emissivity tape for reference points

**Environmental factors:**
- Avoid strong infrared reflections
- Account for ambient temperature
- Ensure thermal equilibrium before measuring

### 6. Accuracy Expectations

**With emissivity correction:**
- High emissivity materials: ±2°C
- Low emissivity materials: ±5°C (after correction)
- Without correction on low emissivity: -10°C to -30°C error

### 7. Troubleshooting

**MLX90640 reads too low:**
- Check material emissivity (likely low emissivity)
- Apply emissivity correction
- Verify no cold reflections

**Inconsistent readings:**
- Check thermal stability
- Reduce ambient air movement
- Increase measurement distance slightly

**Large discrepancy with reference:**
- Verify reference sensor accuracy
- Check field of view overlap
- Consider reflected radiation
"""
    
    with open('CALIBRATION_GUIDE.md', 'w') as f:
        f.write(guide_content)
    
    print("📖 Created CALIBRATION_GUIDE.md")

def main():
    """Main setup function"""
    print("🔧 MLX90640 Emissivity System Setup")
    print("=" * 40)
    
    # Check if files already exist
    files_to_create = [
        'thermal_config.yaml',
        'quick_start.py', 
        'CALIBRATION_GUIDE.md'
    ]
    
    existing_files = [f for f in files_to_create if os.path.exists(f)]
    
    if existing_files:
        print(f"⚠️ The following files already exist:")
        for f in existing_files:
            print(f"   - {f}")
        
        overwrite = input("\nOverwrite existing files? (y/N): ").lower().strip()
        if not overwrite.startswith('y'):
            print("❌ Setup cancelled")
            return
    
    # Create directory structure
    print("\n📁 Creating directories...")
    create_directory_structure()
    
    # Create configuration
    print("\n📋 Creating configuration...")
    create_sample_config()
    
    # Create quick start script
    print("\n🚀 Creating quick start script...")
    create_quick_start_script()
    
    # Create calibration guide
    print("\n📖 Creating calibration guide...")
    create_calibration_guide()
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Review thermal_config.yaml and adjust settings")
    print("2. Run: python quick_start.py")
    print("3. Read CALIBRATION_GUIDE.md for detailed instructions")
    print("4. Start with calibration mode to verify accuracy")
    
    print(f"\n📦 Required Python packages:")
    print(f"   pip install numpy pyyaml")
    print(f"   pip install adafruit-circuitpython-mlx90640  # for real sensor")

if __name__ == "__main__":
    main()