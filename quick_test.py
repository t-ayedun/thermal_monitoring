print("🔧 Starting quick test...")

try:
    print("1. Testing Python basics...")
    import sys
    print(f"   Python version: {sys.version}")
    
    print("2. Testing imports...")
    import numpy as np
    print("   ✅ numpy imported")
    
    print("3. Testing mock sensor...")
    from mock_mlx90640 import MockMLX90640
    sensor = MockMLX90640()
    print("   ✅ Mock sensor created")
    
    print("4. Testing sensor reading...")
    frame = sensor.getFrame()
    print(f"   ✅ Got {len(frame)} temperature readings")
    print(f"   Temperature range: {min(frame):.1f}°C to {max(frame):.1f}°C")
    
    print("✅ All tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
