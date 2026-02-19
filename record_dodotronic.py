#!/usr/bin/env python3
"""
Script for recording from dodotronic ultramic384K_evo microphone on Raspberry Pi
Automatically detects the microphone and records at 192kHz
"""

import sys
import os
from datetime import datetime

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

import numpy as np
import scipy.io.wavfile as wav

class DodotronicRecorder:
    def __init__(self, sample_rate=192000):
        """
        Initialize recording
        Args:
            sample_rate: Sample rate (default: 192000 Hz)
        """
        self.sample_rate = sample_rate
        self.device_index = None
        self.device_name = None
        
    def find_dodotronic_device(self):
        """Search and identify dodotronic microphone"""
        devices = sd.query_devices()
        keywords = ['dodotronic', 'ultramic', '384k', '384', 'evo']
        
        print("Searching for dodotronic microphone...")
        
        # Search for device with keywords
        for i, device in enumerate(devices):
            device_name_lower = device['name'].lower()
            is_input = device['max_input_channels'] > 0
            
            if is_input:
                for keyword in keywords:
                    if keyword in device_name_lower:
                        self.device_index = i
                        self.device_name = device['name']
                        print(f"✓ Found microphone: {self.device_name} (index: {i})")
                        # If dodotronic found, try to use 384kHz as default
                        if '384' in device_name_lower or 'ultramic' in device_name_lower:
                            self.sample_rate = 384000
                            print(f"  → Setting default sample rate to 384kHz for this device")
                        return True
        
        # If not found, display list
        print("\n⚠️  Dodotronic microphone not found automatically")
        print("\nAvailable input devices:")
        input_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        
        if not input_devices:
            print("  ✗ No input devices found")
            return False
        
        for i, device in input_devices:
            print(f"  [{i}] {device['name']}")
        
        return False
    
    def check_sample_rate_support(self):
        """Check support for sample rate and find best supported rate"""
        if self.device_index is None:
            return False
        
        # Test current sample rate first
        try:
            sd.check_input_settings(
                device=self.device_index,
                samplerate=self.sample_rate,
                channels=1
            )
            print(f"✓ Device supports {self.sample_rate} Hz")
            return True
        except Exception as e:
            print(f"✗ Device does not support {self.sample_rate} Hz: {e}")
            
            # Try to find the best supported sample rate
            # Test rates in descending order (highest first)
            test_rates = [384000, 192000, 96000, 48000, 44100]
            supported_rates = []
            
            print("\nTesting available sample rates...")
            for test_rate in test_rates:
                try:
                    sd.check_input_settings(
                        device=self.device_index,
                        samplerate=test_rate,
                        channels=1
                    )
                    supported_rates.append(test_rate)
                    print(f"  ✓ {test_rate} Hz - Supported")
                except:
                    print(f"  ✗ {test_rate} Hz - Not supported")
            
            if supported_rates:
                # Use the highest supported rate
                best_rate = max(supported_rates)
                print(f"\n💡 Device supports: {', '.join(map(str, supported_rates))} Hz")
                print(f"   Using highest supported rate: {best_rate} Hz")
                self.sample_rate = best_rate
                return True
            else:
                print("\n✗ No supported sample rates found")
                return False
    
    def record(self, duration, output_file=None):
        """
        Record audio
        Args:
            duration: Recording duration in seconds
            output_file: File path for saving (if None, creates automatic name)
        Returns:
            Audio data (numpy array) and saved file path
        """
        if self.device_index is None:
            if not self.find_dodotronic_device():
                raise RuntimeError("Dodotronic microphone not found")
        
        if not self.check_sample_rate_support():
            raise RuntimeError(f"Device does not support sample rate {self.sample_rate} Hz")
        
        print(f"\nStarting recording...")
        print(f"  Device: {self.device_name}")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Duration: {duration} seconds")
        print(f"  Press Ctrl+C to stop recording early\n")
        
        try:
            # Calculate number of samples
            num_samples = int(self.sample_rate * duration)
            
            # Set recording parameters
            stream_kwargs = {
                'samplerate': self.sample_rate,
                'channels': 1,
                'dtype': 'float32',
                'blocksize': 4096
            }
            
            if self.device_index is not None:
                stream_kwargs['device'] = self.device_index
            
            # Recording
            recording_data = sd.rec(
                frames=num_samples,
                **stream_kwargs
            )
            
            # Wait for recording to finish
            sd.wait()
            
            # Convert to 1D array if needed
            if len(recording_data.shape) > 1:
                recording_data = recording_data[:, 0]
            
            print(f"✓ Recording completed: {len(recording_data)} samples")
            
            # Save file - ensure .wav extension
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"recording_dodotronic_{timestamp}.wav"
            else:
                # Ensure filename has .wav extension
                if not output_file.lower().endswith('.wav'):
                    output_file = output_file + '.wav'
            
            self.save_wav(recording_data, output_file)
            
            return recording_data, output_file
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Recording stopped by user")
            raise
        except Exception as e:
            print(f"\n✗ Recording error: {e}")
            raise
    
    def save_wav(self, audio_data, filename):
        """Save audio data as WAV file"""
        try:
            # Ensure filename has .wav extension
            if not filename.lower().endswith('.wav'):
                filename = filename + '.wav'
            
            # Convert to int16 (standard WAV format)
            # Normalize to values between -1 and 1 before conversion
            audio_normalized = np.clip(audio_data, -1.0, 1.0)
            audio_int16 = (audio_normalized * 32767).astype(np.int16)
            
            # Save as WAV file using scipy.io.wavfile
            wav.write(filename, self.sample_rate, audio_int16)
            
            # Verify file was created
            if not os.path.exists(filename):
                raise IOError(f"File {filename} was not created")
            
            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
            print(f"✓ WAV file saved: {filename}")
            print(f"  Format: WAV (PCM)")
            print(f"  Sample rate: {self.sample_rate} Hz")
            print(f"  Size: {file_size:.2f} MB")
            print(f"  Duration: {len(audio_data) / self.sample_rate:.2f} seconds")
            
        except Exception as e:
            print(f"✗ Error saving WAV file: {e}")
            raise

def main():
    """Main function - interactive recording"""
    print("=" * 80)
    print("Recording from dodotronic ultramic384K_evo microphone")
    print("=" * 80 + "\n")
    
    # Start with 192kHz default, but will auto-adjust if dodotronic found
    recorder = DodotronicRecorder(sample_rate=192000)
    
    # Device detection
    if not recorder.find_dodotronic_device():
        print("\nOptions:")
        print("1. Run: python test_dodotronic.py  (to test devices)")
        print("2. Make sure the microphone is connected and powered on")
        print("3. Try running: arecord -l  (to check audio devices)")
        
        # Option for manual selection
        try:
            choice = input("\nEnter device number manually (or Enter to cancel): ").strip()
            if choice:
                recorder.device_index = int(choice)
                devices = sd.query_devices()
                if 0 <= recorder.device_index < len(devices):
                    recorder.device_name = devices[recorder.device_index]['name']
                    print(f"✓ Selected device: {recorder.device_name}")
                else:
                    print("✗ Invalid device number")
                    sys.exit(1)
            else:
                sys.exit(0)
        except (ValueError, KeyboardInterrupt):
            print("\nCancelled")
            sys.exit(0)
    
    # Check sample rate support
    if not recorder.check_sample_rate_support():
        print("\n✗ Cannot continue - device does not support required sample rate")
        sys.exit(1)
    
    # Get recording parameters
    try:
        duration_input = input("\nEnter recording duration in seconds (default: 10): ").strip()
        duration = float(duration_input) if duration_input else 10.0
        
        if duration <= 0:
            print("✗ Recording duration must be positive")
            sys.exit(1)
        
        filename_input = input("Enter filename to save (or Enter for automatic name): ").strip()
        output_file = filename_input if filename_input else None
        
        # Recording
        audio_data, saved_file = recorder.record(duration, output_file)
        
        print("\n" + "=" * 80)
        print("✓ Recording completed successfully!")
        print(f"  File: {saved_file}")
        print("=" * 80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
