import numpy as np
import time
import random

class MockMLX90640:
    def __init__(self):
        self.width = 32
        self.height = 24
        self.ambient_temp = 25.0  # Base temperature in Celsius
        self.last_reading_time = time.time()
    
    def getFrame(self):
        """Simulate thermal camera frame with realistic temperature distribution"""
        # Create base temperature field
        frame = np.full((self.height, self.width), self.ambient_temp)
        
        # Add some hot spots (simulate heat sources like transformers)
        hot_spots = random.randint(1, 3)
        for _ in range(hot_spots):
            x = random.randint(5, self.width-5)
            y = random.randint(5, self.height-5)
            intensity = random.uniform(15, 40)  # Temperature increase
            
            # Create Gaussian hot spot
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    if 0 <= y+dy < self.height and 0 <= x+dx < self.width:
                        distance = np.sqrt(dx**2 + dy**2)
                        temp_increase = intensity * np.exp(-distance**2 / 3)
                        frame[y+dy, x+dx] += temp_increase
        
        # Add random noise (sensor noise)
        frame += np.random.normal(0, 0.3, frame.shape)
        
        # Simulate time-based temperature drift
        time_factor = (time.time() - self.last_reading_time) * 0.1
        frame += np.sin(time_factor) * 2
        
        return frame.flatten()  # MLX90640 returns flat array of 768 values

# Simple test function
if __name__ == "__main__":
    print("Testing MockMLX90640...")
    sensor = MockMLX90640()
    frame = sensor.getFrame()
    print(f"Frame size: {len(frame)}")
    print(f"Temperature range: {min(frame):.1f}°C to {max(frame):.1f}°C")
    print("✅ MockMLX90640 working!")