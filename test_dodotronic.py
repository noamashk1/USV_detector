#!/usr/bin/env python3
"""
Script to test detection of dodotronic ultramic384K_evo microphone on Raspberry Pi
Displays all available audio devices and checks support for different sample rates
"""

import sys

try:
    import sounddevice as sd
except OSError as e:
    if 'PortAudio' in str(e):
        print("=" * 80)
        print("ERROR: PortAudio library not found")
        print("=" * 80)
        print("\nPlease install PortAudio library:")
        print("  sudo apt-get install -y portaudio19-dev python3-pyaudio")
        print("\nThen reinstall sounddevice:")
        print("  pip install --upgrade --force-reinstall sounddevice")
        print("\nSee README_dodotronic.md for detailed installation instructions.")
        print("=" * 80)
        sys.exit(1)
    else:
        raise

def find_dodotronic_device():
    """Search for dodotronic microphone in device list"""
    devices = sd.query_devices()
    dodotronic_devices = []
    
    print("=" * 80)
    print("List of all available audio devices:")
    print("=" * 80)
    
    for i, device in enumerate(devices):
        device_name = device['name'].lower()
        is_input = device['max_input_channels'] > 0
        
        # Search for keywords
        keywords = ['dodotronic', 'ultramic', '384k', '384', 'evo']
        found_keywords = [kw for kw in keywords if kw in device_name]
        
        print(f"\n[{i}] {device['name']}")
        print(f"    Inputs: {device['max_input_channels']}, Outputs: {device['max_output_channels']}")
        print(f"    Default sample rate: {device['default_samplerate']} Hz")
        
        if is_input:
            if found_keywords:
                print(f"    ⭐ Found keywords: {', '.join(found_keywords)}")
                dodotronic_devices.append((i, device, found_keywords))
            else:
                print(f"    (Input device)")
    
    print("\n" + "=" * 80)
    
    return dodotronic_devices, devices

def test_sample_rates(device_index, device_name):
    """Test support for different sample rates"""
    sample_rates = [96000, 192000, 384000]
    supported_rates = []
    
    print(f"\nTesting sample rate support for: {device_name}")
    print("-" * 80)
    
    for sr in sample_rates:
        try:
            sd.check_input_settings(device=device_index, samplerate=sr, channels=1)
            supported_rates.append(sr)
            print(f"✓ {sr} Hz - Supported")
        except Exception as e:
            print(f"✗ {sr} Hz - Not supported ({str(e)[:50]}...)")
    
    return supported_rates

def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("Testing detection of dodotronic ultramic384K_evo microphone")
    print("=" * 80 + "\n")
    
    # Search for devices
    dodotronic_devices, all_devices = find_dodotronic_device()
    
    if not dodotronic_devices:
        print("\n⚠️  Dodotronic microphone not found automatically!")
        print("\nOptions:")
        print("1. Make sure the microphone is connected via USB")
        print("2. Check that the microphone is powered on")
        print("3. Run: lsusb  (to check USB devices)")
        print("4. Run: arecord -l  (to check audio devices)")
        print("\nSelect a device number from the list above for manual testing:")
        
        try:
            choice = input("\nEnter device number (or Enter to skip): ").strip()
            if choice:
                device_index = int(choice)
                if 0 <= device_index < len(all_devices):
                    device = all_devices[device_index]
                    if device['max_input_channels'] > 0:
                        print(f"\nTesting device: {device['name']}")
                        supported = test_sample_rates(device_index, device['name'])
                        if supported:
                            print(f"\n✓ Device supports sample rates: {', '.join(map(str, supported))} Hz")
                        else:
                            print("\n✗ Device does not support any of the tested sample rates")
                    else:
                        print("\n✗ Selected device is not an input device")
                else:
                    print("\n✗ Invalid device number")
        except (ValueError, KeyboardInterrupt):
            print("\n\nCancelled")
            sys.exit(0)
    else:
        print(f"\n✓ Found {len(dodotronic_devices)} possible device(s):")
        print("-" * 80)
        
        for device_index, device, keywords in dodotronic_devices:
            print(f"\n[{device_index}] {device['name']}")
            print(f"    Found keywords: {', '.join(keywords)}")
            
            # Test sample rate support
            supported = test_sample_rates(device_index, device['name'])
            
            if supported:
                print(f"\n✓ Device supports sample rates: {', '.join(map(str, supported))} Hz")
                print(f"\n💡 To use this device, use index: {device_index}")
            else:
                print("\n⚠️  Device does not support the tested sample rates")
        
        # Device recommendation
        if len(dodotronic_devices) == 1:
            recommended_index = dodotronic_devices[0][0]
            recommended_name = dodotronic_devices[0][1]['name']
            print(f"\n" + "=" * 80)
            print(f"Recommendation: Use device [{recommended_index}] - {recommended_name}")
            print("=" * 80)
    
    print("\n" + "=" * 80)
    print("Testing completed")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
