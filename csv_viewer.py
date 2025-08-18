#!/usr/bin/env python3
"""
CSV Data Viewer and File Manager for Thermal Monitor
Simple tool to view, analyze, and manage thermal monitoring CSV files
"""

import os
import glob
import pandas as pd
from datetime import datetime, timedelta
import argparse


def list_thermal_files(directory="."):
    """List all thermal monitoring files with details"""
    print(f"📁 Thermal monitoring files in {os.path.abspath(directory)}:\n")
    
    # Aggregated files (1-minute summaries)
    agg_files = glob.glob(os.path.join(directory, "thermal_aggregated_*.csv"))
    if agg_files:
        print("📊 AGGREGATED DATA FILES (1-minute summaries):")
        for file in sorted(agg_files):
            size_kb = os.path.getsize(file) / 1024
            mod_time = datetime.fromtimestamp(os.path.getmtime(file))
            print(f"  • {os.path.basename(file)} ({size_kb:.1f}KB) - {mod_time.strftime('%Y-%m-%d %H:%M')}")
    
    # Detailed files (individual readings)
    detailed_files = glob.glob(os.path.join(directory, "thermal_detailed_*.csv"))
    if detailed_files:
        print("\n🔍 DETAILED DATA FILES (individual readings):")
        for file in sorted(detailed_files):
            size_kb = os.path.getsize(file) / 1024
            mod_time = datetime.fromtimestamp(os.path.getmtime(file))
            print(f"  • {os.path.basename(file)} ({size_kb:.1f}KB) - {mod_time.strftime('%Y-%m-%d %H:%M')}")
    
    # Log files
    log_files = glob.glob(os.path.join(directory, "logs/*.log"))
    if log_files:
        print("\n📋 LOG FILES:")
        for file in sorted(log_files):
            size_kb = os.path.getsize(file) / 1024
            mod_time = datetime.fromtimestamp(os.path.getmtime(file))
            print(f"  • {os.path.basename(file)} ({size_kb:.1f}KB) - {mod_time.strftime('%Y-%m-%d %H:%M')}")
    
    # Frame dumps
    frame_files = glob.glob(os.path.join(directory, "logs/thermal_frame_*.json"))
    if frame_files:
        print(f"\n🚨 ALERT FRAME DUMPS ({len(frame_files)} files):")
        for file in sorted(frame_files)[-5:]:  # Show last 5
            mod_time = datetime.fromtimestamp(os.path.getmtime(file))
            print(f"  • {os.path.basename(file)} - {mod_time.strftime('%Y-%m-%d %H:%M')}")
        if len(frame_files) > 5:
            print(f"  ... and {len(frame_files) - 5} more")


def view_latest_data(file_pattern, num_rows=10):
    """View latest data from CSV files"""
    files = glob.glob(file_pattern)
    if not files:
        print(f"❌ No files found matching: {file_pattern}")
        return
    
    latest_file = max(files, key=os.path.getmtime)
    print(f"📊 Latest data from: {os.path.basename(latest_file)}\n")
    
    try:
        df = pd.read_csv(latest_file)
        
        if len(df) == 0:
            print("📝 File is empty (headers only)")
            return
        
        # Show basic info
        print(f"📈 Total records: {len(df)}")
        if 'timestamp' in df.columns:
            first_time = pd.to_datetime(df['timestamp'].iloc[0])
            last_time = pd.to_datetime(df['timestamp'].iloc[-1])
            duration = last_time - first_time
            print(f"⏱️ Time span: {first_time.strftime('%H:%M')} to {last_time.strftime('%H:%M')} ({duration})")
        
        # Show temperature summary if available
        if 'avg_temp' in df.columns:
            avg_temp_series = pd.to_numeric(df['avg_temp'], errors='coerce')
            print(f"🌡️ Temperature range: {avg_temp_series.min():.1f}°C to {avg_temp_series.max():.1f}°C")
            print(f"🌡️ Overall average: {avg_temp_series.mean():.1f}°C")
        
        print(f"\n📋 Last {min(num_rows, len(df))} rows:")
        
        # Format display based on file type
        if 'period_minutes' in df.columns:
            # Aggregated data - show key columns
            display_cols = ['timestamp', 'avg_temp', 'max_temp', 'sample_count', 'alert_level']
            display_cols = [col for col in display_cols if col in df.columns]
        else:
            # Detailed data - show key columns
            display_cols = ['timestamp', 'avg_temp', 'max_temp', 'alert_level', 'reading_number']
            display_cols = [col for col in display_cols if col in df.columns]
        
        # Show the data
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        print(df[display_cols].tail(num_rows).to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")


def analyze_session_data(session_id=None):
    """Analyze data from a specific session"""
    if session_id is None:
        # Find most recent session
        agg_files = glob.glob("thermal_aggregated_*.csv")
        if not agg_files:
            print("❌ No aggregated files found")
            return
        
        latest_file = max(agg_files, key=os.path.getmtime)
        session_id = latest_file.split('_')[2].replace('.csv', '')
        print(f"📊 Analyzing latest session: {session_id}")
    
    # Load aggregated data
    agg_file = f"thermal_aggregated_{session_id}.csv"
    if not os.path.exists(agg_file):
        print(f"❌ Aggregated file not found: {agg_file}")
        return
    
    try:
        df_agg = pd.read_csv(agg_file)
        
        if len(df_agg) == 0:
            print("📝 No data in aggregated file yet")
            return
        
        print(f"\n📊 SESSION ANALYSIS: {session_id}")
        print("=" * 50)
        
        # Basic stats
        avg_temps = pd.to_numeric(df_agg['avg_temp'], errors='coerce')
        max_temps = pd.to_numeric(df_agg['max_temp'], errors='coerce')
        
        print(f"📈 Total 1-minute periods: {len(df_agg)}")
        print(f"🌡️ Temperature summary:")
        print(f"   Average: {avg_temps.mean():.1f}°C")
        print(f"   Range: {avg_temps.min():.1f}°C to {max_temps.max():.1f}°C")
        print(f"   Std Dev: {avg_temps.std():.1f}°C")
        
        # Alert analysis
        if 'alert_level' in df_agg.columns:
            alert_counts = df_agg['alert_level'].value_counts()
            print(f"\n🚨 Alert Summary:")
            for alert, count in alert_counts.items():
                print(f"   {alert}: {count} periods")
        
        # Data quality
        if 'sample_count' in df_agg.columns:
            sample_counts = pd.to_numeric(df_agg['sample_count'], errors='coerce')
            print(f"\n📊 Data Quality:")
            print(f"   Avg samples per minute: {sample_counts.mean():.1f}")
            print(f"   Min samples: {sample_counts.min()}")
            print(f"   Max samples: {sample_counts.max()}")
        
        # Show trend
        if len(df_agg) > 1:
            first_temp = avg_temps.iloc[0]
            last_temp = avg_temps.iloc[-1]
            trend = "📈 Rising" if last_temp > first_temp + 1 else "📉 Falling" if last_temp < first_temp - 1 else "➡️ Stable"
            print(f"\n🔄 Temperature Trend: {trend}")
            print(f"   Start: {first_temp:.1f}°C → End: {last_temp:.1f}°C")
        
    except Exception as e:
        print(f"❌ Error analyzing session data: {e}")


def cleanup_old_files(days_to_keep=7):
    """Clean up files older than specified days"""
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    
    patterns = [
        "thermal_aggregated_*.csv",
        "thermal_detailed_*.csv", 
        "logs/thermal_frame_*.json"
    ]
    
    deleted_count = 0
    
    for pattern in patterns:
        files = glob.glob(pattern)
        for file in files:
            mod_time = datetime.fromtimestamp(os.path.getmtime(file))
            if mod_time < cutoff_date:
                try:
                    os.remove(file)
                    deleted_count += 1
                    print(f"🗑️ Deleted old file: {os.path.basename(file)}")
                except Exception as e:
                    print(f"❌ Error deleting {file}: {e}")
    
    print(f"✅ Cleanup complete: {deleted_count} files removed")


def export_data_for_analysis(output_file="thermal_export.csv"):
    """Export combined data for external analysis"""
    try:
        # Combine all aggregated files
        agg_files = glob.glob("thermal_aggregated_*.csv")
        
        if not agg_files:
            print("❌ No aggregated files found")
            return
        
        combined_data = []
        for file in sorted(agg_files):
            df = pd.read_csv(file)
            combined_data.append(df)
        
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            combined_df.to_csv(output_file, index=False)
            print(f"📊 Exported {len(combined_df)} records to {output_file}")
            
            # Show summary
            if 'avg_temp' in combined_df.columns:
                avg_temps = pd.to_numeric(combined_df['avg_temp'], errors='coerce')
                print(f"🌡️ Overall average: {avg_temps.mean():.1f}°C")
                print(f"📈 Range: {avg_temps.min():.1f}°C to {avg_temps.max():.1f}°C")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Thermal Monitor CSV Viewer and Manager')
    parser.add_argument('--list', action='store_true', help='List all thermal files')
    parser.add_argument('--latest', action='store_true', help='View latest aggregated data')
    parser.add_argument('--detailed', action='store_true', help='View latest detailed data')
    parser.add_argument('--analyze', type=str, nargs='?', const='latest', 
                       help='Analyze session data (session_id or "latest")')
    parser.add_argument('--cleanup', type=int, metavar='DAYS', 
                       help='Delete files older than N days')
    parser.add_argument('--export', type=str, nargs='?', const='thermal_export.csv',
                       help='Export all data to single CSV file')
    parser.add_argument('--rows', type=int, default=10, help='Number of rows to display')
    
    args = parser.parse_args()
    
    if args.list:
        list_thermal_files()
    elif args.latest:
        view_latest_data("thermal_aggregated_*.csv", args.rows)
    elif args.detailed:
        view_latest_data("thermal_detailed_*.csv", args.rows)
    elif args.analyze:
        session_id = None if args.analyze == 'latest' else args.analyze
        analyze_session_data(session_id)
    elif args.cleanup is not None:
        cleanup_old_files(args.cleanup)
    elif args.export:
        export_data_for_analysis(args.export)
    else:
        # Interactive mode
        print("🌡️ Thermal Monitor CSV Manager")
        print("1. List all files")
        print("2. View latest aggregated data")
        print("3. View latest detailed data") 
        print("4. Analyze latest session")
        print("5. Cleanup old files")
        print("6. Export all data")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            list_thermal_files()
        elif choice == "2":
            view_latest_data("thermal_aggregated_*.csv", 10)
        elif choice == "3":
            view_latest_data("thermal_detailed_*.csv", 10)
        elif choice == "4":
            analyze_session_data()
        elif choice == "5":
            days = input("Delete files older than how many days? (default: 7): ").strip()
            days = int(days) if days else 7
            cleanup_old_files(days)
        elif choice == "6":
            filename = input("Export filename (default: thermal_export.csv): ").strip()
            filename = filename if filename else "thermal_export.csv"
            export_data_for_analysis(filename)
        else:
            print("❌ Invalid choice")


if __name__ == '__main__':
    main()