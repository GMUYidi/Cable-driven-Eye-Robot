# ESP32 IMU Camera Firmware

This folder contains firmware for the ESP32-S3 camera and BNO055 IMU module.

Main firmware:

```text
IMUcamera.ino
```

The firmware is responsible for:

- Starting the ESP32 camera stream.
- Reading BNO055 IMU data.
- Sending IMU orientation data over serial.
- Joining or creating a Wi-Fi network for camera streaming.

Local Wi-Fi credentials should be stored in `wifi_secrets.h`. That file is
ignored by git and should not be committed.

An example credentials file can be created locally as:

```c
#define WIFI_SSID "your-network"
#define WIFI_PASSWORD "your-password"
```
