// Two BNO085 IMUs streaming accelerometer + gyroscope at 200 Hz each.
//
//   sensor 0 = jaw  (chin)  on Wire1: SDA D18, SCL D19
//   sensor 1 = head (still) on Wire : SDA D21, SCL D22
//
// Output, one line per sensor per sample (921600 baud):
//   t_us,sensor,gx,gy,gz,ax,ay,az,qw,qx,qy,qz
// gyro in rad/s, accel in m/s^2, t_us = ESP32 micros() when the sample was read.
// The quaternion columns are always 0 (not used). Lines starting with '#' are
// status messages and are ignored by the backend.
//
// Why not the SparkFun library: its SH2 driver is a singleton, so only one sensor
// could be open at a time. This file talks SHTP over I2C directly, so each sensor
// has its own state and both stream continuously.

#include <Arduino.h>
#include <Wire.h>

constexpr int JAW_SDA = 18, JAW_SCL = 19;
constexpr int HEAD_SDA = 21, HEAD_SCL = 22;
constexpr int LED_PIN = 2;
constexpr uint32_t REPORT_INTERVAL_US = 5000;  // 200 Hz
constexpr uint32_t STALL_MS = 1000;            // no samples for this long -> re-initialise the sensor
constexpr size_t MAX_READ = 100;               // bytes per I2C transaction (Wire buffer is 128)

constexpr uint8_t CH_EXEC = 1, CH_CONTROL = 2, CH_SENSOR = 3;
constexpr uint8_t REPORT_ACCEL = 0x01, REPORT_GYRO = 0x02;
constexpr uint8_t REPORT_BASE_TS = 0xFB, REPORT_TS_REBASE = 0xFA;

struct Bno {
  const char *name;
  uint8_t id;  // value written in the sensor column
  TwoWire *bus;
  uint8_t addr = 0;
  uint8_t seqOut[8] = {0};
  uint8_t payload[256];
  float a[3] = {0, 0, 0};
  bool haveAccel = false;
  uint32_t lastSampleMs = 0;
  uint32_t lastTryMs = 0;
  bool up = false;

  // ---- low level ---------------------------------------------------------
  bool sendPacket(uint8_t channel, const uint8_t *data, uint16_t len) {
    uint16_t total = len + 4;
    bus->beginTransmission(addr);
    bus->write((uint8_t)(total & 0xFF));
    bus->write((uint8_t)(total >> 8));
    bus->write(channel);
    bus->write(seqOut[channel]++);
    bus->write(data, len);
    return bus->endTransmission() == 0;
  }

  // Reads one SHTP packet. Returns payload length (0 = nothing waiting).
  // The payload is stored in `payload` (truncated if oversized; the rest is drained).
  int readPacket(uint8_t &channel) {
    if (bus->requestFrom((uint16_t)addr, (uint8_t)4) != 4) return 0;
    uint8_t h0 = bus->read(), h1 = bus->read(), h2 = bus->read();
    bus->read();
    uint16_t len = ((uint16_t)h1 << 8 | h0) & 0x7FFF;
    channel = h2;
    if (len <= 4 || len == 0x7FFF) return 0;
    size_t need = len - 4, got = 0;
    while (got < need) {
      size_t want = min(need - got + 4, MAX_READ);
      if (bus->requestFrom((uint16_t)addr, (uint8_t)want) != (int)want) return 0;
      for (int i = 0; i < 4; i++) bus->read();  // every transaction repeats the header
      for (size_t i = 0; i < want - 4; i++) {
        uint8_t b = bus->read();
        if (got < sizeof(payload)) payload[got] = b;
        got++;
      }
    }
    return (int)min(need, sizeof(payload));
  }

  bool enableReport(uint8_t reportId) {
    uint8_t cmd[17] = {0xFD, reportId, 0, 0, 0};
    for (int i = 0; i < 4; i++) cmd[5 + i] = (REPORT_INTERVAL_US >> (8 * i)) & 0xFF;  // report interval (us)
    for (int i = 9; i < 17; i++) cmd[i] = 0;                                          // batch interval, sensor specific
    return sendPacket(CH_CONTROL, cmd, sizeof(cmd));
  }

  bool enableReports() { return enableReport(REPORT_ACCEL) && enableReport(REPORT_GYRO); }

  void drain(uint32_t quietMs) {
    uint32_t quietSince = millis();
    uint8_t ch;
    while (millis() - quietSince < quietMs) {
      if (readPacket(ch) > 0) quietSince = millis();
      else delay(2);
    }
  }

  // ---- lifecycle ---------------------------------------------------------
  bool begin() {
    lastTryMs = millis();
    up = false;
    for (uint8_t candidate : {(uint8_t)0x4A, (uint8_t)0x4B}) {
      addr = candidate;
      bus->beginTransmission(addr);
      if (bus->endTransmission() != 0) continue;
      const uint8_t softReset = 1;
      memset(seqOut, 0, sizeof(seqOut));
      sendPacket(CH_EXEC, &softReset, 1);
      delay(300);
      drain(150);
      if (enableReports()) {
        up = true;
        haveAccel = false;
        lastSampleMs = millis();
        Serial.printf("# %s IMU ready at 0x%02X\n", name, addr);
        return true;
      }
    }
    Serial.printf("# %s IMU not found\n", name);
    return false;
  }

  // Reads waiting packets (at most a few) and prints a line per gyro report.
  // Returns true if anything was read.
  bool poll() {
    if (!up) {
      if (millis() - lastTryMs > 2000) begin();
      return false;
    }
    bool any = false;
    for (int n = 0; n < 4; n++) {
      uint8_t ch;
      int len = readPacket(ch);
      if (len <= 0) break;
      any = true;
      uint32_t now = micros();
      if (ch == CH_EXEC) {
        if (payload[0] == 0x01) enableReports();  // sensor reset itself: turn the reports back on
      } else if (ch == CH_SENSOR) {
        parseSensorReports(len, now);
      }
    }
    if (millis() - lastSampleMs > STALL_MS) {
      Serial.printf("# %s IMU stalled, re-initialising\n", name);
      begin();
    }
    return any;
  }

  void parseSensorReports(int len, uint32_t tUs) {
    int i = 0;
    while (i < len) {
      uint8_t id = payload[i];
      if (id == REPORT_BASE_TS || id == REPORT_TS_REBASE) {
        i += 5;
      } else if ((id == REPORT_ACCEL || id == REPORT_GYRO) && i + 10 <= len) {
        int16_t x = (int16_t)(payload[i + 4] | payload[i + 5] << 8);
        int16_t y = (int16_t)(payload[i + 6] | payload[i + 7] << 8);
        int16_t z = (int16_t)(payload[i + 8] | payload[i + 9] << 8);
        if (id == REPORT_ACCEL) {
          a[0] = x / 256.0f; a[1] = y / 256.0f; a[2] = z / 256.0f;  // Q8, m/s^2
          haveAccel = true;
        } else if (haveAccel) {
          emit(tUs, x / 512.0f, y / 512.0f, z / 512.0f);            // Q9, rad/s
        }
        i += 10;
      } else {
        break;  // unknown report: skip the rest of this packet
      }
    }
  }

  void emit(uint32_t tUs, float gx, float gy, float gz) {
    char line[128];
    int n = snprintf(line, sizeof(line), "%lu,%u,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,0,0,0,0\n", (unsigned long)tUs, id, gx, gy,
                     gz, a[0], a[1], a[2]);
    Serial.write((const uint8_t *)line, n);
    lastSampleMs = millis();
  }
};

Bno jaw{"jaw", 0, &Wire1};
Bno head{"head", 1, &Wire};

void setup() {
  Serial.setTxBufferSize(8192);  // 2 x 200 lines/s must never block on the UART
  Serial.begin(921600);
  pinMode(LED_PIN, OUTPUT);
  Wire1.begin(JAW_SDA, JAW_SCL);
  Wire1.setClock(400000);
  Wire1.setBufferSize(128);
  Wire.begin(HEAD_SDA, HEAD_SCL);
  Wire.setClock(400000);
  Wire.setBufferSize(128);
  delay(500);
  jaw.begin();
  head.begin();
}

void loop() {
  bool a = jaw.poll();
  bool b = head.poll();
  digitalWrite(LED_PIN, jaw.up && head.up ? HIGH : LOW);
  if (!a && !b) delayMicroseconds(300);
}
