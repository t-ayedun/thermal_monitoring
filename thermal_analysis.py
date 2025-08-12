# thermal_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_thermal_log(csv_file):
    """Analyze thermal monitoring log"""
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Generate report
    print(f"Monitoring period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Temperature range: {df['min_temp'].min():.1f}°C to {df['max_temp'].max():.1f}°C")
    print(f"Average temperature: {df['avg_temp'].mean():.1f}°C")
    
    # Plot temperature trends
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['min_temp'], label='Min Temperature', alpha=0.7)
    plt.plot(df['timestamp'], df['max_temp'], label='Max Temperature', alpha=0.7)
    plt.plot(df['timestamp'], df['avg_temp'], label='Average Temperature', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.title('Thermal Camera Temperature Monitoring')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('thermal_analysis.png', dpi=300)
    plt.show()

# Test with mock data
if __name__ == "__main__":
    # Run the monitor to generate test data first
    monitor = ThermalMonitor(mock_mode=False)
    monitor.run_monitoring(duration_minutes=2)
    
    # Then analyze the results
    analyze_thermal_log("thermal_data.csv")