# emissivity_calibration.py - Comprehensive emissivity correction for MLX90640
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

class EmissivityDatabase:
    """Database of emissivity values for common materials"""
    
    # Comprehensive emissivity table (at ~25°C unless noted)
    EMISSIVITY_TABLE = {
        # Metals (generally low emissivity)
        'aluminum_polished': 0.05,
        'aluminum_oxidized': 0.25,
        'aluminum_rough': 0.18,
        'brass_polished': 0.03,
        'brass_oxidized': 0.61,
        'copper_polished': 0.04,
        'copper_oxidized': 0.78,
        'steel_polished': 0.07,
        'steel_oxidized': 0.79,
        'stainless_steel_polished': 0.17,
        'stainless_steel_oxidized': 0.85,
        'iron_cast': 0.81,
        'iron_rust': 0.69,
        'zinc_polished': 0.02,
        'zinc_oxidized': 0.11,
        
        # Transformer materials (key for your application)
        'transformer_oil': 0.94,
        'transformer_steel_core': 0.80,
        'copper_windings_new': 0.04,
        'copper_windings_aged': 0.74,
        'aluminum_windings': 0.22,
        'porcelain_insulator': 0.92,
        'epoxy_resin': 0.94,
        'fiberglass_insulation': 0.75,
        'paper_insulation': 0.95,
        'mineral_oil': 0.95,
        
        # Common materials for calibration
        'electrical_tape_black': 0.95,
        'paint_black': 0.98,
        'paint_white': 0.91,
        'paint_aluminum': 0.45,
        'concrete': 0.94,
        'brick': 0.93,
        'wood': 0.85,
        'glass': 0.92,
        'plastic_black': 0.95,
        'plastic_white': 0.84,
        'rubber': 0.93,
        'water': 0.98,
        'ice': 0.97,
        'snow': 0.85,
        'soil_dry': 0.92,
        'asphalt': 0.95,
        
        # Biological materials
        'human_skin': 0.98,
        'vegetation': 0.94,
        'leaves_green': 0.95,
        
        # Building materials
        'drywall': 0.90,
        'ceramic_tile': 0.94,
        'marble': 0.95,
        'granite': 0.45,
        
        # High-emissivity reference materials (good for calibration)
        'carbon_black': 0.96,
        'lamp_black': 0.96,
        'graphite': 0.98,
        'anodized_aluminum_black': 0.98
    }
    
    @classmethod
    def get_emissivity(cls, material: str) -> float:
        """Get emissivity value for a material"""
        material_key = material.lower().replace(' ', '_').replace('-', '_')
        return cls.EMISSIVITY_TABLE.get(material_key, 0.95)  # Default to high emissivity
    
    @classmethod
    def list_materials(cls) -> List[str]:
        """List all available materials"""
        return sorted(cls.EMISSIVITY_TABLE.keys())
    
    @classmethod
    def find_similar_materials(cls, search_term: str) -> List[str]:
        """Find materials matching search term"""
        search_lower = search_term.lower()
        matches = []
        for material in cls.EMISSIVITY_TABLE.keys():
            if search_lower in material:
                matches.append(material)
        return sorted(matches)

class EmissivityCorrector:
    """Handles emissivity corrections for thermal measurements"""
    
    def __init__(self):
        self.logger = logging.getLogger('EmissivityCorrector')
        self.ambient_temperature = 25.0  # °C - Will be updated
        self.calibration_data = {}
        
    def stefan_boltzmann_correction(self, measured_temp: float, 
                                  measured_emissivity: float,
                                  actual_emissivity: float,
                                  ambient_temp: float = None) -> float:
        """
        Apply Stefan-Boltzmann law correction for emissivity
        
        The MLX90640 assumes emissivity = 0.95 by default
        We need to correct for actual material emissivity
        """
        if ambient_temp is None:
            ambient_temp = self.ambient_temperature
        
        # Convert to Kelvin
        T_measured_K = measured_temp + 273.15
        T_ambient_K = ambient_temp + 273.15
        
        # Stefan-Boltzmann correction
        # The sensor detects: ε_actual * σ * T_actual^4 + (1-ε_actual) * ε_ambient * σ * T_ambient^4
        # But calculates assuming: ε_assumed * σ * T_measured^4
        
        # Simplified correction (assuming ambient reflection is minimal)
        correction_factor = actual_emissivity / measured_emissivity
        
        # Apply correction
        T_corrected_K = T_measured_K * (correction_factor ** 0.25)
        
        return T_corrected_K - 273.15
    
    def simple_emissivity_correction(self, measured_temp: float, 
                                   actual_emissivity: float,
                                   assumed_emissivity: float = 0.95) -> float:
        """
        Simplified emissivity correction for MLX90640
        
        The MLX90640 is calibrated assuming emissivity = 0.95
        This provides a linear approximation for small corrections
        """
        # Linear approximation correction
        correction_factor = actual_emissivity / assumed_emissivity
        corrected_temp = measured_temp * correction_factor
        
        return corrected_temp
    
    def advanced_correction(self, measured_temp: float,
                          actual_emissivity: float,
                          ambient_temp: float,
                          reflected_temp: float = None) -> float:
        """
        Advanced correction accounting for reflected radiation
        """
        if reflected_temp is None:
            reflected_temp = ambient_temp
        
        # Convert to Kelvin
        T_meas_K = measured_temp + 273.15
        T_amb_K = ambient_temp + 273.15
        T_refl_K = reflected_temp + 273.15
        
        # Account for reflected radiation
        # T_object^4 = (T_measured^4 - (1-ε) * T_reflected^4) / ε
        T_obj_K_4th = (T_meas_K**4 - (1 - actual_emissivity) * T_refl_K**4) / actual_emissivity
        T_corrected_K = T_obj_K_4th ** 0.25
        
        return T_corrected_K - 273.15

class MLX90640EmissivityCalibrator:
    """Calibration system for MLX90640 with emissivity correction"""
    
    def __init__(self, thermal_monitor):
        self.monitor = thermal_monitor
        self.corrector = EmissivityCorrector()
        self.db = EmissivityDatabase()
        self.logger = logging.getLogger('MLX90640Calibrator')
        
        # Calibration targets - materials with known emissivity
        self.calibration_targets = [
            {'name': 'electrical_tape_black', 'emissivity': 0.95, 'description': 'Black electrical tape (high emissivity reference)'},
            {'name': 'aluminum_polished', 'emissivity': 0.05, 'description': 'Polished aluminum (low emissivity reference)'},
            {'name': 'transformer_oil', 'emissivity': 0.94, 'description': 'Transformer oil (your application)'},
            {'name': 'copper_windings_aged', 'emissivity': 0.74, 'description': 'Aged copper windings'},
            {'name': 'paint_black', 'emissivity': 0.98, 'description': 'Black paint (near-blackbody reference)'}
        ]
    
    def guided_emissivity_calibration(self) -> Dict[str, float]:
        """
        Interactive calibration process using known materials
        """
        print("\n🎯 MLX90640 EMISSIVITY CALIBRATION")
        print("=" * 50)
        print("This process will help you calibrate your thermal camera")
        print("for accurate measurements on different materials.")
        print("\nYou'll need:")
        print("- A contact thermometer (thermocouple, RTD, etc.)")
        print("- Objects made of known materials")
        print("- Stable temperature environment")
        
        calibration_results = {}
        
        for target in self.calibration_targets:
            print(f"\n📋 CALIBRATION TARGET: {target['name'].upper()}")
            print(f"Description: {target['description']}")
            print(f"Expected emissivity: {target['emissivity']}")
            
            proceed = input("\nDo you have this material available? (y/n): ").lower().strip()
            if not proceed.startswith('y'):
                print("⏭️ Skipping this target...")
                continue
            
            # Get reference temperature
            ref_temp = self.get_reference_temperature(target['name'])
            if ref_temp is None:
                continue
            
            # Take thermal measurements
            thermal_readings = self.take_thermal_measurements(target['name'])
            if not thermal_readings:
                continue
            
            # Calculate correction
            avg_thermal = np.mean(thermal_readings)
            correction = self.calculate_emissivity_correction(
                measured_temp=avg_thermal,
                reference_temp=ref_temp,
                expected_emissivity=target['emissivity']
            )
            
            calibration_results[target['name']] = {
                'reference_temp': ref_temp,
                'thermal_readings': thermal_readings,
                'avg_thermal': avg_thermal,
                'correction_factor': correction,
                'expected_emissivity': target['emissivity']
            }
            
            print(f"✅ Calibration complete for {target['name']}")
            print(f"   Reference: {ref_temp:.2f}°C")
            print(f"   Thermal reading: {avg_thermal:.2f}°C")
            print(f"   Correction factor: {correction:.3f}")
        
        # Save calibration data
        self.save_calibration_data(calibration_results)
        
        return calibration_results
    
    def get_reference_temperature(self, material_name: str) -> Optional[float]:
        """Get reference temperature from contact measurement"""
        print(f"\n🌡️ REFERENCE MEASUREMENT for {material_name}")
        print("1. Place your contact thermometer on the target material")
        print("2. Wait for temperature to stabilize")
        print("3. Record the temperature")
        
        try:
            ref_temp = float(input("Enter reference temperature (°C): "))
            
            # Sanity check
            if ref_temp < -50 or ref_temp > 200:
                print("⚠️ Temperature seems out of normal range")
                confirm = input("Continue anyway? (y/n): ").lower().strip()
                if not confirm.startswith('y'):
                    return None
            
            return ref_temp
            
        except ValueError:
            print("❌ Invalid temperature value")
            return None
    
    def take_thermal_measurements(self, material_name: str) -> List[float]:
        """Take multiple thermal measurements for averaging"""
        print(f"\n📸 THERMAL MEASUREMENTS for {material_name}")
        print("1. Point the MLX90640 at the target material")
        print("2. Ensure the target fills the field of view")
        print("3. Avoid reflections from other heat sources")
        
        input("Press Enter when ready to start measurements...")
        
        readings = []
        num_readings = 10
        
        print(f"Taking {num_readings} readings...")
        
        for i in range(num_readings):
            # Take thermal reading
            temps = self.monitor.read_thermal_frame()
            if temps:
                # Use max temperature (hotspot) for calibration
                readings.append(temps['max_temp'])
                print(f"Reading {i+1}/{num_readings}: {temps['max_temp']:.2f}°C")
            else:
                print(f"Reading {i+1}/{num_readings}: FAILED")
            
            time.sleep(1)  # Wait between readings
        
        if len(readings) < num_readings // 2:
            print("❌ Too many failed readings")
            return []
        
        # Remove outliers (simple method)
        readings_array = np.array(readings)
        mean_temp = np.mean(readings_array)
        std_temp = np.std(readings_array)
        
        # Keep readings within 2 standard deviations
        filtered_readings = readings_array[np.abs(readings_array - mean_temp) < 2 * std_temp]
        
        print(f"📊 Statistics:")
        print(f"   Mean: {np.mean(filtered_readings):.2f}°C")
        print(f"   Std Dev: {np.std(filtered_readings):.2f}°C")
        print(f"   Valid readings: {len(filtered_readings)}/{len(readings)}")
        
        return filtered_readings.tolist()
    
    def calculate_emissivity_correction(self, measured_temp: float, 
                                      reference_temp: float,
                                      expected_emissivity: float) -> float:
        """Calculate emissivity correction factor"""
        # Simple linear correction factor
        if measured_temp != 0:
            correction_factor = reference_temp / measured_temp
        else:
            correction_factor = 1.0
        
        return correction_factor
    
    def save_calibration_data(self, calibration_data: Dict) -> None:
        """Save calibration data to file"""
        timestamp = datetime.now().isoformat()
        
        calibration_file = {
            'timestamp': timestamp,
            'mlx90640_calibration': calibration_data,
            'emissivity_database_version': '1.0'
        }
        
        filename = f"mlx90640_emissivity_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(calibration_file, f, indent=2, default=str)
            
            print(f"\n💾 Calibration data saved to: {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save calibration data: {e}")
    
    def apply_material_correction(self, thermal_frame: List[float], 
                                material: str,
                                ambient_temp: float = 25.0) -> List[float]:
        """Apply emissivity correction to entire thermal frame"""
        material_emissivity = self.db.get_emissivity(material)
        
        corrected_frame = []
        for temp in thermal_frame:
            corrected_temp = self.corrector.simple_emissivity_correction(
                measured_temp=temp,
                actual_emissivity=material_emissivity
            )
            corrected_frame.append(corrected_temp)
        
        return corrected_frame
    
    def quick_calibration_test(self) -> None:
        """Quick test using black electrical tape (high emissivity reference)"""
        print("\n🎯 QUICK CALIBRATION TEST")
        print("=" * 40)
        print("This test uses black electrical tape as a reference.")
        print("Black tape has emissivity ≈ 0.95 (very close to blackbody)")
        print("\nSetup:")
        print("1. Apply black electrical tape to a metal surface")
        print("2. Let it reach thermal equilibrium")
        print("3. Measure both tape and bare metal with contact thermometer")
        print("4. Compare with MLX90640 readings")
        
        input("\nPress Enter when ready...")
        
        # Measure tape (high emissivity)
        print("\n📸 Point MLX90640 at the BLACK TAPE")
        input("Press Enter to measure...")
        
        tape_temps = self.monitor.read_thermal_frame()
        if tape_temps:
            tape_reading = tape_temps['max_temp']
            print(f"MLX90640 reading (black tape): {tape_reading:.2f}°C")
        
        # Measure metal (low emissivity)
        print("\n📸 Point MLX90640 at the BARE METAL")
        input("Press Enter to measure...")
        
        metal_temps = self.monitor.read_thermal_frame()
        if metal_temps:
            metal_reading = metal_temps['max_temp']
            print(f"MLX90640 reading (bare metal): {metal_reading:.2f}°C")
        
        # Analysis
        if tape_temps and metal_temps:
            temp_difference = tape_reading - metal_reading
            print(f"\n📊 ANALYSIS:")
            print(f"Temperature difference: {temp_difference:.2f}°C")
            
            if abs(temp_difference) < 2.0:
                print("✅ Good! Small difference suggests similar actual temperatures")
                print("   The MLX90640 is working reasonably well")
            elif temp_difference > 5.0:
                print("⚠️  Large difference suggests emissivity effects")
                print("   Metal appears cooler due to low emissivity")
                print("   Consider emissivity correction for metal measurements")
            else:
                print("🔍 Moderate difference - normal for different materials")

# CLI interface for emissivity calibration
if __name__ == "__main__":
    print("🔧 MLX90640 Emissivity Calibration Tool")
    print("=" * 50)
    
    # Show material database
    db = EmissivityDatabase()
    print(f"\n📚 Material Database: {len(db.EMISSIVITY_TABLE)} materials")
    
    # Search functionality
    search_term = input("\nSearch for material (or press Enter to continue): ").strip()
    if search_term:
        matches = db.find_similar_materials(search_term)
        if matches:
            print(f"\n🔍 Found {len(matches)} matches:")
            for material in matches[:10]:  # Show top 10
                emissivity = db.get_emissivity(material)
                print(f"   {material}: {emissivity}")
        else:
            print("❌ No matches found")
    
    # Show transformer-related materials
    print(f"\n🔌 Transformer Materials:")
    transformer_materials = db.find_similar_materials('transformer')
    transformer_materials.extend(db.find_similar_materials('copper'))
    transformer_materials.extend(db.find_similar_materials('oil'))
    
    for material in sorted(set(transformer_materials)):
        emissivity = db.get_emissivity(material)
        print(f"   {material}: {emissivity}")
    
    print(f"\n💡 For accurate measurements:")
    print(f"   1. Identify your target material")
    print(f"   2. Look up its emissivity value")
    print(f"   3. Apply correction in your thermal monitoring code")
    print(f"   4. Consider calibration with known references")