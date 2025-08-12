import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_thermal_log(csv_file):
    """Analyze thermal monitoring log"""
    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found.")
        return

    df = pd.read_csv(csv_file)
    if df.empty:
        print("❌ Error: CSV file is empty.")
        return

    # Ensure timestamp column exists
    if 'timestamp' not in df.columns:
        print("❌ Error: 'timestamp' column missing in CSV.")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Drop any rows with invalid timestamps
    df = df.dropna(subset=['timestamp'])

    # Basic stats
    print(f"📅 Monitoring period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"🌡 Temperature range: {df['min_temp'].min():.1f}°C to {df['max_temp'].max():.1f}°C")
    print(f"📊 Average temperature: {df['avg_temp'].mean():.1f}°C")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['min_temp'], label='Min Temp (°C)', alpha=0.7)
    plt.plot(df['timestamp'], df['max_temp'], label='Max Temp (°C)', alpha=0.7)
    plt.plot(df['timestamp'], df['avg_temp'], label='Avg Temp (°C)', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.title('Thermal Camera Temperature Monitoring')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('thermal_analysis.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # Default to analyzing the main operational data file
    analyze_thermal_log("thermal_data.csv")
