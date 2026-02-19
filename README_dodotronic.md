# Dodotronic Ultramic384K_evo Microphone Setup for Raspberry Pi

This guide explains how to set up and use the dodotronic ultramic384K_evo microphone with Raspberry Pi.

## Prerequisites

### 1. Install System Dependencies

First, install PortAudio library which is required by `sounddevice`:

```bash
# Update package list
sudo apt-get update

# Install PortAudio development libraries
sudo apt-get install -y portaudio19-dev python3-pyaudio

# For Python 3.11+ you might also need:
sudo apt-get install -y python3-dev
```

### 2. Install Python Dependencies

Create a virtual environment (recommended):

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install required packages
pip install sounddevice numpy scipy
```

Or install globally:

```bash
pip3 install sounddevice numpy scipy
```

### 3. Audio Permissions

Add your user to the audio group:

```bash
sudo usermod -a -G audio $USER
```

**Important:** You need to log out and log back in (or reboot) for this to take effect.

## Hardware Setup

1. Connect the dodotronic ultramic384K_evo microphone to Raspberry Pi via USB
2. Ensure the microphone is powered on (if it has a power switch)
3. Verify connection:
   ```bash
   lsusb
   ```
   You should see a USB audio device listed.

## Usage

### Step 1: Test Microphone Detection

Run the test script to verify the microphone is detected:

```bash
python test_dodotronic.py
```

This will:
- List all available audio devices
- Search for the dodotronic microphone
- Test sample rate support (96kHz, 192kHz, 384kHz)

### Step 2: Record Audio

Run the recording script:

```bash
python record_dodotronic.py
```

The script will:
- Automatically detect the dodotronic microphone
- Check sample rate support
- Prompt for recording duration
- Save the recording as a WAV file

## Troubleshooting

### PortAudio Library Not Found

If you see the error:
```
OSError: PortAudio library not found
```

Install PortAudio:
```bash
sudo apt-get install -y portaudio19-dev python3-pyaudio
```

Then reinstall sounddevice:
```bash
pip install --upgrade --force-reinstall sounddevice
```

### Microphone Not Detected

1. Check USB connection:
   ```bash
   lsusb
   ```

2. Check audio devices:
   ```bash
   arecord -l
   ```

3. Check if the device appears in ALSA:
   ```bash
   cat /proc/asound/cards
   ```

### Permission Errors

If you get permission errors:
```bash
sudo usermod -a -G audio $USER
```
Then log out and log back in.

### Sample Rate Not Supported

The microphone supports up to 384kHz, but your system might not. The script will automatically suggest alternative sample rates (96kHz or 192kHz) if 384kHz is not available.

## Files

- `test_dodotronic.py` - Script to test microphone detection and sample rate support
- `record_dodotronic.py` - Script to record audio from the microphone

## Notes

- Default sample rate is 192kHz (can be changed in the code)
- Recordings are saved as WAV files with automatic timestamp naming
- The scripts automatically detect the microphone by searching for keywords: "dodotronic", "ultramic", "384k", "384", "evo"
