#include <Arduino.h>
#include "esp_camera.h"
#include <Wire.h>
#include <WiFi.h>
#include <WebServer.h>

#define CAMERA_ROLE_EXTERNAL_WIFI 1
#define CAMERA_ROLE_HUB_AP 2
#define CAMERA_ROLE_CLIENT_TO_HUB 3
#define CAMERA_ROLE_AUTO_ESP_AP 4

#if __has_include("wifi_secrets.h")
#include "wifi_secrets.h"
#endif

#ifndef CAMERA_WIFI_ROLE
#define CAMERA_WIFI_ROLE CAMERA_ROLE_AUTO_ESP_AP
#endif

#ifndef WIFI_SSID
#define WIFI_SSID ""
#define WIFI_PASSWORD ""
#endif

#ifndef HUB_AP_SSID
#define HUB_AP_SSID "ESP32S3-CAM-NET"
#define HUB_AP_PASSWORD "esp32camera"
#endif

#ifndef HUB_CLIENT_STATIC_IP_LAST_OCTET
#define HUB_CLIENT_STATIC_IP_LAST_OCTET 200
#endif

// Aideepen/Keyestudio/Freenove-style ESP32-S3 CAM, same pinout as CAMERA_MODEL_ESP32S3_EYE.
// Arduino IDE target: ESP32S3 Dev Module
// Important Tools settings for this tested board:
//   Flash Size: 8MB
//   PSRAM: OPI PSRAM / Octal PSRAM
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1

#define XCLK_GPIO_NUM  15
#define SIOD_GPIO_NUM   4
#define SIOC_GPIO_NUM   5

#define Y9_GPIO_NUM    16
#define Y8_GPIO_NUM    17
#define Y7_GPIO_NUM    18
#define Y6_GPIO_NUM    12
#define Y5_GPIO_NUM    10
#define Y4_GPIO_NUM     8
#define Y3_GPIO_NUM     9
#define Y2_GPIO_NUM    11

#define VSYNC_GPIO_NUM  6
#define HREF_GPIO_NUM   7
#define PCLK_GPIO_NUM  13

// BNO055 IMU I2C pins. These do not conflict with the camera pins above.
static const int IMU_SDA_PIN = 1;
static const int IMU_SCL_PIN = 2;

static const uint8_t BNO055_ADDR = 0x28;  // Change to 0x29 if ADR is high.

static const uint8_t BNO055_CHIP_ID_ADDR = 0x00;
static const uint8_t BNO055_GYRO_DATA_X_LSB_ADDR = 0x14;
static const uint8_t BNO055_EULER_H_LSB_ADDR = 0x1A;
static const uint8_t BNO055_LINEAR_ACCEL_DATA_X_LSB_ADDR = 0x28;
static const uint8_t BNO055_OPR_MODE_ADDR = 0x3D;
static const uint8_t BNO055_PWR_MODE_ADDR = 0x3E;
static const uint8_t BNO055_SYS_TRIGGER_ADDR = 0x3F;
static const uint8_t BNO055_UNIT_SEL_ADDR = 0x3B;

static const uint8_t CONFIGMODE = 0x00;
static const uint8_t NDOF_MODE = 0x0C;
static const uint8_t NORMAL_POWER_MODE = 0x00;

static const uint32_t SERIAL_BAUD = 921600;
static const unsigned long IMU_INTERVAL_MS = 50;  // 20 Hz, matching IMUtest.
static const uint32_t WIFI_CONNECT_TIMEOUT_MS = 15000;
static const uint32_t HUB_SCAN_DELAY_MIN_MS = 1200;
static const uint32_t HUB_SCAN_DELAY_SPREAD_MS = 6000;
static const char *AP_SSID_PREFIX = "ESP32S3-CAM";
static const char *AP_PASSWORD = "esp32camera";

// Stability-first defaults. After QVGA is stable, use /set?size=VGA&quality=12.
static const framesize_t START_FRAME_SIZE = FRAMESIZE_QVGA;
static const int START_JPEG_QUALITY = 15;
static const uint16_t STREAM_DELAY_MS = 20;
static const size_t STREAM_CHUNK_SIZE = 1460;

WebServer server(80);

bool softApMode = false;
bool cameraReady = false;
framesize_t currentFrameSize = START_FRAME_SIZE;
int currentJpegQuality = START_JPEG_QUALITY;
uint32_t totalFrames = 0;
uint32_t failedFrames = 0;
String deviceSuffix;
String deviceHostname;
String softApSsid;
String activeNetworkRole = "unknown";

bool imuReady = false;
unsigned long lastIMURead = 0;
uint32_t imuReadCount = 0;
uint32_t imuReadErrors = 0;

bool hasYawOffset = false;
float yawOffset = 0.0f;

bool hasLastEuler = false;
bool hasLastEulerRate = false;
unsigned long lastEulerMicros = 0;
float lastYaw = 0.0f;
float lastPitch = 0.0f;
float lastRoll = 0.0f;
float lastYawRate = 0.0f;
float lastPitchRate = 0.0f;
float lastRollRate = 0.0f;

bool hasLatestIMU = false;
unsigned long latestIMUMillis = 0;
float latestYaw = 0.0f;
float latestRoll = 0.0f;
float latestPitch = 0.0f;
float latestYawRate = 0.0f;
float latestPitchRate = 0.0f;
float latestRollRate = 0.0f;
float latestYawAccel = 0.0f;
float latestPitchAccel = 0.0f;
float latestRollAccel = 0.0f;
float latestLinAccX = 0.0f;
float latestLinAccY = 0.0f;
float latestLinAccZ = 0.0f;
float latestGyroX = 0.0f;
float latestGyroY = 0.0f;
float latestGyroZ = 0.0f;

struct FrameSizeOption {
  const char *name;
  framesize_t value;
};

static const FrameSizeOption FRAME_SIZES[] = {
  {"QQVGA", FRAMESIZE_QQVGA},
  {"QVGA", FRAMESIZE_QVGA},
  {"VGA", FRAMESIZE_VGA},
  {"SVGA", FRAMESIZE_SVGA},
};

const char *frameSizeName(framesize_t value) {
  for (const FrameSizeOption &option : FRAME_SIZES) {
    if (option.value == value) {
      return option.name;
    }
  }
  return "UNKNOWN";
}

IPAddress cameraIp() {
  return softApMode ? WiFi.softAPIP() : WiFi.localIP();
}

String makeDeviceSuffix() {
  uint64_t mac = ESP.getEfuseMac();
  char suffix[7];
  snprintf(suffix, sizeof(suffix), "%06X", (uint32_t)(mac & 0xFFFFFF));
  return String(suffix);
}

void initDeviceIdentity() {
  deviceSuffix = makeDeviceSuffix();
  deviceHostname = "esp32s3-cam-" + deviceSuffix;
  softApSsid = String(AP_SSID_PREFIX) + "-" + deviceSuffix;
}

bool writeReg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(BNO055_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool readRegs(uint8_t reg, uint8_t *buffer, uint8_t len) {
  Wire.beginTransmission(BNO055_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  uint8_t count = Wire.requestFrom(BNO055_ADDR, len);
  if (count != len) {
    return false;
  }

  for (uint8_t i = 0; i < len; i++) {
    buffer[i] = Wire.read();
  }
  return true;
}

int16_t readS16LE(uint8_t *buf, int idx) {
  return (int16_t)((buf[idx + 1] << 8) | buf[idx]);
}

float normalizeAngleDelta(float deltaDeg) {
  while (deltaDeg > 180.0f) {
    deltaDeg -= 360.0f;
  }
  while (deltaDeg <= -180.0f) {
    deltaDeg += 360.0f;
  }
  return deltaDeg;
}

void appendIMUCsv(String &text) {
  text += "IMU,";
  text += String(latestYaw, 2);
  text += ",";
  text += String(latestRoll, 2);
  text += ",";
  text += String(latestPitch, 2);
  text += ",";
  text += String(latestYawRate, 3);
  text += ",";
  text += String(latestPitchRate, 3);
  text += ",";
  text += String(latestRollRate, 3);
  text += ",";
  text += String(latestYawAccel, 3);
  text += ",";
  text += String(latestPitchAccel, 3);
  text += ",";
  text += String(latestRollAccel, 3);
  text += ",";
  text += String(latestLinAccX, 3);
  text += ",";
  text += String(latestLinAccY, 3);
  text += ",";
  text += String(latestLinAccZ, 3);
  text += ",";
  text += String(latestGyroX, 3);
  text += ",";
  text += String(latestGyroY, 3);
  text += ",";
  text += String(latestGyroZ, 3);
  text += "\n";
}

void printLatestIMUCsv() {
  Serial.print("IMU,");
  Serial.print(latestYaw, 2);
  Serial.print(",");
  Serial.print(latestRoll, 2);
  Serial.print(",");
  Serial.print(latestPitch, 2);
  Serial.print(",");
  Serial.print(latestYawRate, 3);
  Serial.print(",");
  Serial.print(latestPitchRate, 3);
  Serial.print(",");
  Serial.print(latestRollRate, 3);
  Serial.print(",");
  Serial.print(latestYawAccel, 3);
  Serial.print(",");
  Serial.print(latestPitchAccel, 3);
  Serial.print(",");
  Serial.print(latestRollAccel, 3);
  Serial.print(",");
  Serial.print(latestLinAccX, 3);
  Serial.print(",");
  Serial.print(latestLinAccY, 3);
  Serial.print(",");
  Serial.print(latestLinAccZ, 3);
  Serial.print(",");
  Serial.print(latestGyroX, 3);
  Serial.print(",");
  Serial.print(latestGyroY, 3);
  Serial.print(",");
  Serial.println(latestGyroZ, 3);
}

bool initBNO055() {
  Wire.begin(IMU_SDA_PIN, IMU_SCL_PIN);
  Wire.setClock(100000);
  delay(50);

  uint8_t chipId = 0;
  if (!readRegs(BNO055_CHIP_ID_ADDR, &chipId, 1)) {
    Serial.println("WARN Failed to read BNO055 chip ID; camera will continue without IMU.");
    return false;
  }

  Serial.print("BNO055 Chip ID = 0x");
  Serial.println(chipId, HEX);

  if (chipId != 0xA0) {
    Serial.println("WARN BNO055 chip ID is not 0xA0; camera will continue without IMU.");
    return false;
  }

  if (!writeReg(BNO055_OPR_MODE_ADDR, CONFIGMODE)) return false;
  delay(50);

  if (!writeReg(BNO055_PWR_MODE_ADDR, NORMAL_POWER_MODE)) return false;
  delay(10);

  // 0x00: Euler angle deg, gyro deg/s, acceleration m/s^2.
  if (!writeReg(BNO055_UNIT_SEL_ADDR, 0x00)) return false;
  delay(10);

  if (!writeReg(BNO055_SYS_TRIGGER_ADDR, 0x00)) return false;
  delay(10);

  if (!writeReg(BNO055_OPR_MODE_ADDR, NDOF_MODE)) return false;
  delay(200);

  return true;
}

void readAndPrintIMU() {
  uint8_t eulerData[6];
  uint8_t gyroData[6];
  uint8_t linearAccelData[6];

  bool okEuler = readRegs(BNO055_EULER_H_LSB_ADDR, eulerData, 6);
  bool okGyro = readRegs(BNO055_GYRO_DATA_X_LSB_ADDR, gyroData, 6);
  bool okLinearAccel = readRegs(BNO055_LINEAR_ACCEL_DATA_X_LSB_ADDR, linearAccelData, 6);

  if (!okEuler || !okGyro || !okLinearAccel) {
    imuReadErrors++;
    Serial.println("READ_ERROR");
    return;
  }

  int16_t headingRaw = readS16LE(eulerData, 0);
  int16_t rollRaw = readS16LE(eulerData, 2);
  int16_t pitchRaw = readS16LE(eulerData, 4);

  int16_t gyroXRaw = readS16LE(gyroData, 0);
  int16_t gyroYRaw = readS16LE(gyroData, 2);
  int16_t gyroZRaw = readS16LE(gyroData, 4);

  int16_t linAccXRaw = readS16LE(linearAccelData, 0);
  int16_t linAccYRaw = readS16LE(linearAccelData, 2);
  int16_t linAccZRaw = readS16LE(linearAccelData, 4);

  float yaw = headingRaw / 16.0f;
  float roll = rollRaw / 16.0f;
  float pitch = pitchRaw / 16.0f;

  if (!hasYawOffset) {
    yawOffset = yaw;
    hasYawOffset = true;
  }
  yaw = normalizeAngleDelta(yaw - yawOffset);

  float gyroX = gyroXRaw / 16.0f;
  float gyroY = gyroYRaw / 16.0f;
  float gyroZ = gyroZRaw / 16.0f;

  float linAccX = linAccXRaw / 100.0f;
  float linAccY = linAccYRaw / 100.0f;
  float linAccZ = linAccZRaw / 100.0f;

  unsigned long nowMicros = micros();
  float yawRate = 0.0f;
  float pitchRate = 0.0f;
  float rollRate = 0.0f;
  float yawAccel = 0.0f;
  float pitchAccel = 0.0f;
  float rollAccel = 0.0f;
  bool computedEulerRate = false;

  if (hasLastEuler) {
    float dt = (nowMicros - lastEulerMicros) / 1000000.0f;
    if (dt > 0.000001f) {
      yawRate = normalizeAngleDelta(yaw - lastYaw) / dt;
      pitchRate = (pitch - lastPitch) / dt;
      rollRate = normalizeAngleDelta(roll - lastRoll) / dt;
      computedEulerRate = true;

      if (hasLastEulerRate) {
        yawAccel = (yawRate - lastYawRate) / dt;
        pitchAccel = (pitchRate - lastPitchRate) / dt;
        rollAccel = (rollRate - lastRollRate) / dt;
      }
    }
  }

  lastYaw = yaw;
  lastPitch = pitch;
  lastRoll = roll;
  lastYawRate = yawRate;
  lastPitchRate = pitchRate;
  lastRollRate = rollRate;
  lastEulerMicros = nowMicros;
  hasLastEuler = true;
  hasLastEulerRate = computedEulerRate;

  latestYaw = yaw;
  latestRoll = roll;
  latestPitch = pitch;
  latestYawRate = yawRate;
  latestPitchRate = pitchRate;
  latestRollRate = rollRate;
  latestYawAccel = yawAccel;
  latestPitchAccel = pitchAccel;
  latestRollAccel = rollAccel;
  latestLinAccX = linAccX;
  latestLinAccY = linAccY;
  latestLinAccZ = linAccZ;
  latestGyroX = gyroX;
  latestGyroY = gyroY;
  latestGyroZ = gyroZ;
  latestIMUMillis = millis();
  hasLatestIMU = true;
  imuReadCount++;

  printLatestIMUCsv();
}

void serviceIMU() {
  if (!imuReady) {
    return;
  }

  unsigned long now = millis();
  if (now - lastIMURead >= IMU_INTERVAL_MS) {
    lastIMURead = now;
    readAndPrintIMU();
  }
}

void delayWithIMU(uint32_t ms) {
  uint32_t start = millis();
  while (millis() - start < ms) {
    serviceIMU();
    delay(2);
  }
}

bool writeClientAll(WiFiClient &client, const uint8_t *buffer, size_t length) {
  size_t offset = 0;
  uint16_t noProgressCount = 0;

  while (offset < length && client.connected()) {
    size_t remaining = length - offset;
    size_t chunk = remaining < STREAM_CHUNK_SIZE ? remaining : STREAM_CHUNK_SIZE;
    size_t written = client.write(buffer + offset, chunk);

    if (written == 0) {
      noProgressCount++;
      if (noProgressCount > 200) {
        return false;
      }
      serviceIMU();
      delay(1);
      continue;
    }

    noProgressCount = 0;
    offset += written;
    serviceIMU();
    delay(0);
  }

  return offset == length;
}

void appendStatus(String &text) {
  text += "ready=yes\n";
  text += "ip=" + cameraIp().toString() + "\n";
  text += "mode=" + String(softApMode ? "soft_ap" : "wifi_sta") + "\n";
  text += "network_role=" + activeNetworkRole + "\n";
  text += "device=" + deviceHostname + "\n";
  text += "mac_suffix=" + deviceSuffix + "\n";
  text += "camera_ready=" + String(cameraReady ? "yes" : "no") + "\n";
  text += "stream=http://" + cameraIp().toString() + "/stream\n";
  text += "capture=http://" + cameraIp().toString() + "/capture\n";
  text += "psram_found=" + String(psramFound() ? "yes" : "no") + "\n";
  text += "psram_size=" + String((uint32_t)ESP.getPsramSize()) + "\n";
  text += "free_psram=" + String((uint32_t)ESP.getFreePsram()) + "\n";
  text += "heap_size=" + String((uint32_t)ESP.getHeapSize()) + "\n";
  text += "free_heap=" + String((uint32_t)ESP.getFreeHeap()) + "\n";
  text += "frame_size=" + String(frameSizeName(currentFrameSize)) + "\n";
  text += "jpeg_quality=" + String(currentJpegQuality) + "\n";
  text += "total_frames=" + String(totalFrames) + "\n";
  text += "failed_frames=" + String(failedFrames) + "\n";
  text += "imu_ready=" + String(imuReady ? "yes" : "no") + "\n";
  text += "imu_samples=" + String(imuReadCount) + "\n";
  text += "imu_read_errors=" + String(imuReadErrors) + "\n";
  if (hasLatestIMU) {
    text += "imu_age_ms=" + String((uint32_t)(millis() - latestIMUMillis)) + "\n";
    text += "imu_yaw=" + String(latestYaw, 2) + "\n";
    text += "imu_roll=" + String(latestRoll, 2) + "\n";
    text += "imu_pitch=" + String(latestPitch, 2) + "\n";
    text += "imu_yaw_rate=" + String(latestYawRate, 3) + "\n";
    text += "imu_pitch_rate=" + String(latestPitchRate, 3) + "\n";
    text += "imu_roll_rate=" + String(latestRollRate, 3) + "\n";
  }
}

void handleRoot() {
  String html;
  html.reserve(1800);
  html += "<!doctype html><html><head>";
  html += "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">";
  html += "<title>ESP32-S3 CAM Test</title>";
  html += "<style>body{font-family:Arial,sans-serif;margin:20px;background:#101418;color:#f4f7fb}";
  html += "a{color:#6ee7ff}img{max-width:100%;height:auto;border:1px solid #3a4652}";
  html += "code{background:#1b232c;padding:2px 5px;border-radius:4px}</style>";
  html += "</head><body>";
  html += "<h2>ESP32-S3 CAM Test</h2>";
  html += "<p>Device: <code>";
  html += deviceHostname;
  html += "</code></p>";
  html += "<p>Role: <code>";
  html += activeNetworkRole;
  html += "</code></p>";
  html += "<p><code>http://";
  html += cameraIp().toString();
  html += "</code></p>";
  html += "<p><a href=\"/capture\" target=\"_blank\">capture</a> ";
  html += "<a href=\"/stream\" target=\"_blank\">stream</a> ";
  html += "<a href=\"/status\" target=\"_blank\">status</a> ";
  html += "<a href=\"/imu\" target=\"_blank\">imu</a></p>";
  html += "<p>";
  html += "<a href=\"/set?size=QVGA&quality=15\">QVGA</a> ";
  html += "<a href=\"/set?size=VGA&quality=12\">VGA</a> ";
  html += "<a href=\"/set?size=SVGA&quality=14\">SVGA</a>";
  html += "</p>";
  if (cameraReady) {
    html += "<img src=\"/stream\">";
  } else {
    html += "<p><strong>Camera is not available.</strong> IMU serial output and <a href=\"/imu\" target=\"_blank\">/imu</a> are still running.</p>";
  }
  html += "</body></html>";
  server.send(200, "text/html", html);
}

void handleStatus() {
  String status;
  status.reserve(1100);
  appendStatus(status);
  server.send(200, "text/plain", status);
}

void handleIMU() {
  String response;
  response.reserve(450);
  response += "ready=" + String(imuReady ? "yes" : "no") + "\n";
  response += "samples=" + String(imuReadCount) + "\n";
  response += "read_errors=" + String(imuReadErrors) + "\n";

  if (!hasLatestIMU) {
    response += "latest=none\n";
    server.send(imuReady ? 200 : 503, "text/plain", response);
    return;
  }

  response += "age_ms=" + String((uint32_t)(millis() - latestIMUMillis)) + "\n";
  response += "format=IMU,yaw,roll,pitch,yaw_rate,pitch_rate,roll_rate,yaw_accel,pitch_accel,roll_accel,linacc_x,linacc_y,linacc_z,gyro_x,gyro_y,gyro_z\n";
  appendIMUCsv(response);
  server.send(200, "text/plain", response);
}

void handleSet() {
  if (!cameraReady) {
    server.send(503, "text/plain", "camera is not available; IMU is still running");
    return;
  }

  sensor_t *sensor = esp_camera_sensor_get();
  if (!sensor) {
    server.send(500, "text/plain", "camera sensor is not available");
    return;
  }

  String response;
  response.reserve(300);

  if (server.hasArg("size")) {
    String requested = server.arg("size");
    requested.toUpperCase();

    bool found = false;
    for (const FrameSizeOption &option : FRAME_SIZES) {
      if (requested == option.name) {
        int rc = sensor->set_framesize(sensor, option.value);
        if (rc == 0) {
          currentFrameSize = option.value;
          response += "frame_size=" + String(option.name) + "\n";
        } else {
          response += "frame_size_error=" + String(rc) + "\n";
        }
        found = true;
        break;
      }
    }

    if (!found) {
      response += "frame_size_error=unknown size; use QQVGA,QVGA,VGA,SVGA\n";
    }
  }

  if (server.hasArg("quality")) {
    int quality = server.arg("quality").toInt();
    if (quality >= 4 && quality <= 30) {
      int rc = sensor->set_quality(sensor, quality);
      if (rc == 0) {
        currentJpegQuality = quality;
        response += "jpeg_quality=" + String(quality) + "\n";
      } else {
        response += "jpeg_quality_error=" + String(rc) + "\n";
      }
    } else {
      response += "jpeg_quality_error=use 4..30, higher number is lower quality/faster\n";
    }
  }

  if (response.length() == 0) {
    response = "nothing changed; try /set?size=VGA&quality=12\n";
  }

  response += "\n";
  appendStatus(response);
  server.send(200, "text/plain", response);
}

void handleCapture() {
  if (!cameraReady) {
    server.send(503, "text/plain", "camera is not available; IMU is still running");
    return;
  }

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    failedFrames++;
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }

  totalFrames++;
  WiFiClient client = server.client();
  client.printf(
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: image/jpeg\r\n"
    "Content-Length: %u\r\n"
    "Cache-Control: no-store\r\n"
    "Connection: close\r\n"
    "\r\n",
    (unsigned int)fb->len
  );
  client.write(fb->buf, fb->len);
  client.flush();

  esp_camera_fb_return(fb);
  client.stop();
}

void handleStream() {
  if (!cameraReady) {
    server.send(503, "text/plain", "camera is not available; IMU is still running");
    return;
  }

  WiFiClient client = server.client();
  client.setNoDelay(true);

  client.print("HTTP/1.1 200 OK\r\n");
  client.print("Content-Type: multipart/x-mixed-replace; boundary=frame\r\n");
  client.print("Cache-Control: no-cache\r\n");
  client.print("Connection: close\r\n\r\n");

  uint32_t windowStartMs = millis();
  uint32_t windowFrames = 0;
  uint32_t windowBytes = 0;

  while (client.connected()) {
    serviceIMU();

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      failedFrames++;
      Serial.println("Camera capture failed");
      delayWithIMU(50);
      continue;
    }

    totalFrames++;
    windowFrames++;
    windowBytes += fb->len;

    client.print("--frame\r\n");
    client.print("Content-Type: image/jpeg\r\n");
    client.printf("Content-Length: %u\r\n\r\n", (unsigned int)fb->len);
    bool ok = writeClientAll(client, fb->buf, fb->len);
    client.print("\r\n");

    esp_camera_fb_return(fb);

    if (!ok) {
      break;
    }

    uint32_t now = millis();
    if (now - windowStartMs >= 1000) {
      float seconds = (now - windowStartMs) / 1000.0f;
      float fps = windowFrames / seconds;
      uint32_t avgBytes = windowFrames > 0 ? windowBytes / windowFrames : 0;
      Serial.printf(
        "Stream: %.1f fps, avg=%u bytes, size=%s, q=%d, free_heap=%u, free_psram=%u\n",
        fps,
        (unsigned int)avgBytes,
        frameSizeName(currentFrameSize),
        currentJpegQuality,
        (unsigned int)ESP.getFreeHeap(),
        (unsigned int)ESP.getFreePsram()
      );
      windowStartMs = now;
      windowFrames = 0;
      windowBytes = 0;
    }

    delayWithIMU(STREAM_DELAY_MS);
  }

  client.stop();
  Serial.println("Stream client disconnected");
}

bool initCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = START_FRAME_SIZE;
  config.jpeg_quality = START_JPEG_QUALITY;
  config.grab_mode = CAMERA_GRAB_LATEST;

  if (psramFound()) {
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.fb_count = 2;
    Serial.println("PSRAM found, using PSRAM frame buffers");
  } else {
    config.frame_size = FRAMESIZE_QQVGA;
    currentFrameSize = FRAMESIZE_QQVGA;
    config.jpeg_quality = 20;
    currentJpegQuality = 20;
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.fb_count = 1;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    Serial.println("PSRAM not found, falling back to QQVGA DRAM mode");
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }

  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor) {
    sensor->set_vflip(sensor, 1);
    sensor->set_hmirror(sensor, 0);
    sensor->set_quality(sensor, currentJpegQuality);
  }

  Serial.println("Camera init success");
  return true;
}

void printMemoryInfo() {
  Serial.printf("Chip model: %s\n", ESP.getChipModel());
  Serial.printf("Device hostname: %s\n", deviceHostname.c_str());
  Serial.printf("MAC suffix: %s\n", deviceSuffix.c_str());
  Serial.printf("CPU frequency: %u MHz\n", (unsigned int)ESP.getCpuFreqMHz());
  Serial.printf("Flash size: %u bytes\n", (unsigned int)ESP.getFlashChipSize());
  Serial.printf("PSRAM found: %s\n", psramFound() ? "YES" : "NO");
  Serial.printf("PSRAM size: %u bytes\n", (unsigned int)ESP.getPsramSize());
  Serial.printf("Free PSRAM: %u bytes\n", (unsigned int)ESP.getFreePsram());
  Serial.printf("Heap size: %u bytes\n", (unsigned int)ESP.getHeapSize());
  Serial.printf("Free heap: %u bytes\n", (unsigned int)ESP.getFreeHeap());
}

void startSoftApNetwork(const String &ssid, const char *password, const char *roleName) {
  activeNetworkRole = roleName;
  softApMode = true;
  WiFi.mode(WIFI_AP);
  WiFi.setSleep(false);

  IPAddress apIp(192, 168, 4, 1);
  IPAddress subnet(255, 255, 255, 0);
  WiFi.softAPConfig(apIp, apIp, subnet);

  bool ok = WiFi.softAP(ssid.c_str(), password, 1, 0, 4);
  Serial.println();
  Serial.println(ok ? "SoftAP started" : "SoftAP start failed");
  Serial.printf("Role: %s\n", activeNetworkRole.c_str());
  Serial.printf("SSID: %s\n", ssid.c_str());
  Serial.printf("Password: %s\n", password);
  Serial.printf("Open: http://%s\n", WiFi.softAPIP().toString().c_str());
  Serial.println("Laptop should connect to this Wi-Fi network.");
}

bool connectToWifiNetwork(const char *ssid, const char *password, const char *roleName, bool useHubStaticIp) {
  activeNetworkRole = roleName;
  softApMode = false;
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setHostname(deviceHostname.c_str());

  if (useHubStaticIp) {
    IPAddress localIp(192, 168, 4, HUB_CLIENT_STATIC_IP_LAST_OCTET);
    IPAddress gateway(192, 168, 4, 1);
    IPAddress subnet(255, 255, 255, 0);
    WiFi.config(localIp, gateway, subnet, gateway);
    Serial.printf("Using static client IP: %s\n", localIp.toString().c_str());
  }

  WiFi.begin(ssid, password);

  Serial.printf("Connecting to WiFi: %s", ssid);
  uint32_t startMs = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startMs < WIFI_CONNECT_TIMEOUT_MS) {
    delayWithIMU(500);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.printf("WiFi failed, status=%d\n", WiFi.status());
    WiFi.disconnect(true);
    delayWithIMU(300);
    return false;
  }

  Serial.println("WiFi connected");
  Serial.printf("Role: %s\n", activeNetworkRole.c_str());
  Serial.printf("Open: http://%s\n", WiFi.localIP().toString().c_str());
  return true;
}

bool hubApVisible() {
  softApMode = false;
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setHostname(deviceHostname.c_str());

  Serial.printf("Scanning for ESP camera hub: %s\n", HUB_AP_SSID);
  int networkCount = WiFi.scanNetworks();
  bool found = false;
  int32_t bestRssi = -127;

  for (int i = 0; i < networkCount; i++) {
    if (WiFi.SSID(i) == String(HUB_AP_SSID)) {
      found = true;
      bestRssi = max(bestRssi, WiFi.RSSI(i));
    }
  }

  WiFi.scanDelete();

  if (found) {
    Serial.printf("Hub found, RSSI=%d dBm\n", (int)bestRssi);
  } else {
    Serial.println("Hub not found");
  }

  return found;
}

void connectAutoEspAp() {
  if (hubApVisible()) {
    if (connectToWifiNetwork(HUB_AP_SSID, HUB_AP_PASSWORD, "client_to_esp_hub", true)) {
      return;
    }
  }

  uint32_t macDelay = (uint32_t)(ESP.getEfuseMac() % HUB_SCAN_DELAY_SPREAD_MS);
  uint32_t waitMs = HUB_SCAN_DELAY_MIN_MS + macDelay;
  Serial.printf("No usable hub yet. Waiting %u ms before claiming hub role.\n", (unsigned int)waitMs);
  delayWithIMU(waitMs);

  if (hubApVisible()) {
    if (connectToWifiNetwork(HUB_AP_SSID, HUB_AP_PASSWORD, "client_to_esp_hub", true)) {
      return;
    }
  }

  startSoftApNetwork(String(HUB_AP_SSID), HUB_AP_PASSWORD, "hub_ap");
  Serial.println("Hub camera stream is normally http://192.168.4.1/stream");
}

void connectWifiOrStartAp() {
#if CAMERA_WIFI_ROLE == CAMERA_ROLE_HUB_AP
  startSoftApNetwork(String(HUB_AP_SSID), HUB_AP_PASSWORD, "hub_ap");
  Serial.println("Hub camera stream is normally http://192.168.4.1/stream");
#elif CAMERA_WIFI_ROLE == CAMERA_ROLE_CLIENT_TO_HUB
  if (!connectToWifiNetwork(HUB_AP_SSID, HUB_AP_PASSWORD, "client_to_esp_hub", true)) {
    startSoftApNetwork(softApSsid, AP_PASSWORD, "fallback_ap");
  }
#elif CAMERA_WIFI_ROLE == CAMERA_ROLE_EXTERNAL_WIFI
  if (strlen(WIFI_SSID) == 0 || !connectToWifiNetwork(WIFI_SSID, WIFI_PASSWORD, "external_wifi", false)) {
    startSoftApNetwork(softApSsid, AP_PASSWORD, "fallback_ap");
  }
#else
  connectAutoEspAp();
#endif
}

void startServer() {
  server.on("/", HTTP_GET, handleRoot);
  server.on("/status", HTTP_GET, handleStatus);
  server.on("/imu", HTTP_GET, handleIMU);
  server.on("/set", HTTP_GET, handleSet);
  server.on("/capture", HTTP_GET, handleCapture);
  server.on("/stream", HTTP_GET, handleStream);
  server.on("/favicon.ico", HTTP_GET, []() {
    server.send(204);
  });
  server.onNotFound([]() {
    server.send(404, "text/plain", "Open /, /stream, /capture, /status, or /set?size=VGA&quality=12");
  });
  server.begin();
  Serial.println("HTTP camera server started");
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(1500);
  Serial.println();
  Serial.println("ESP32-S3 camera diagnostic stream starting");

  initDeviceIdentity();
  printMemoryInfo();

  Serial.printf("IMU I2C pins: SDA=%d SCL=%d\n", IMU_SDA_PIN, IMU_SCL_PIN);
  imuReady = initBNO055();
  if (imuReady) {
    Serial.println("READY imu_camera");
    Serial.printf("MODE IMU 20Hz baud=%lu\n", SERIAL_BAUD);
    Serial.println("Yaw output is zeroed by subtracting the first valid yaw sample.");
    Serial.println("Output format: IMU,yaw,roll,pitch,yaw_rate,pitch_rate,roll_rate,yaw_accel,pitch_accel,roll_accel,linacc_x,linacc_y,linacc_z,gyro_x,gyro_y,gyro_z");
  } else {
    Serial.println("WARN IMU unavailable; camera stream and web server will still run.");
  }

  cameraReady = initCamera();
  if (!cameraReady) {
    Serial.println("WARN Camera init failed; IMU serial output will continue.");
  }

  connectWifiOrStartAp();
  startServer();

  Serial.printf("Status: http://%s/status\n", cameraIp().toString().c_str());
  if (cameraReady) {
    Serial.printf("Stream: http://%s/stream\n", cameraIp().toString().c_str());
  } else {
    Serial.println("Stream: unavailable because camera init failed");
  }
  Serial.printf("IMU: http://%s/imu\n", cameraIp().toString().c_str());
}

void loop() {
  serviceIMU();
  server.handleClient();
}
