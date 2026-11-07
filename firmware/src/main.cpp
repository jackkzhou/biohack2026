// Two BNO085 IMUs streaming accelerometer + gyroscope at 100 Hz each.
//
//   sensor 0 = jaw  (chin)  on Wire1: SDA D18, SCL D19
//   sensor 1 = head (still) on Wire : SDA D21, SCL D22
//
// Output, one line per sensor per sample (921600 baud):
//   t_us,sensor,gx,gy,gz,ax,ay,az,qw,qx,qy,qz
// gyro in rad/s, accel in m/s^2, t_us = ESP32 micros() when the sample was read.
// qw,qx,qy,qz is the BNO085's on-chip Game Rotation Vector (gyro+accel fusion, no
// magnetometer), latest value at the time of the gyro sample; 0,0,0,0 until the first one arrives. Lines starting with '#' are
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
constexpr uint32_t JAW_CLOCK_HZ = 400000;
// The head sensor is on longer wires: at 400 kHz it dropped packets and reset itself, 100 kHz is stable.
constexpr uint32_t HEAD_CLOCK_HZ = 100000;
constexpr uint32_t REPORT_INTERVAL_US = 10000;  // 100 Hz
constexpr uint32_t STALL_MS = 1000;            // no samples for this long -> re-initialise the sensor
constexpr size_t MAX_READ = 100;               // bytes per I2C transaction (Wire buffer is 128)

constexpr uint8_t CH_EXEC = 1, CH_CONTROL = 2, CH_SENSOR = 3;
constexpr uint8_t REPORT_ACCEL = 0x01, REPORT_GYRO = 0x02, REPORT_GAME_ROT = 0x08;
constexpr uint8_t REPORT_BASE_TS = 0xFB, REPORT_TS_REBASE = 0xFA;

SemaphoreHandle_t serialMux;
// Only one I2C transaction at a time across both buses: with both buses switching at once
// the head sensor (long wires) dropped packets and reset itself.
SemaphoreHandle_t i2cMux;
struct I2cLock {
  I2cLock() { xSemaphoreTake(i2cMux, portMAX_DELAY); }
  ~I2cLock() { xSemaphoreGive(i2cMux); }
};

// One line at a time on the UART, from either sensor task.
static void writeLine(const char *line, size_t n) {
  xSemaphoreTake(serialMux, portMAX_DELAY);
  Serial.write((const uint8_t *)line, n);
  xSemaphoreGive(serialMux);
}

#define LOGF(...)                                \
  do {                                           \
    char m_[96];                                 \
    int n_ = snprintf(m_, sizeof(m_), __VA_ARGS__); \
    writeLine(m_, n_);                           \
  } while (0)

struct Bno {
  const char *name;
  uint8_t id;  // value written in the sensor column
  TwoWire *bus;
  uint8_t addr = 0;
  uint8_t seqOut[8] = {0};
  uint8_t payload[256];
  float a[3] = {0, 0, 0};
  float q[4] = {0, 0, 0, 0};  // qw qx qy qz
  bool haveAccel = false;
  uint32_t lastSampleMs = 0;
  uint32_t lastTryMs = 0;
  volatile bool up = false;

  // ---- low level ---------------------------------------------------------
  bool sendPacket(uint8_t channel, const uint8_t *data, uint16_t len) {
    I2cLock lock;
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
    I2cLock lock;
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

  bool enableReports() { return enableReport(REPORT_ACCEL) && enableReport(REPORT_GYRO) && enableReport(REPORT_GAME_ROT); }

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
        q[0] = q[1] = q[2] = q[3] = 0;
        lastSampleMs = millis();
        LOGF("# %s IMU ready at 0x%02X\n", name, addr);
        return true;
      }
    }
    LOGF("# %s IMU not found\n", name);
    return false;
  }

  // Reads waiting packets (at most a few) and prints a line per gyro report.
  // Returns true if anything was read.
  bool poll() {
    if (!up) {
      if (millis() - lastTryMs > 1000) begin();
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
      LOGF("# %s IMU stalled, re-initialising\n", name);
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
      } else if (id == REPORT_GAME_ROT && i + 12 <= len) {
        // Q14 unit quaternion, wire order i,j,k,real
        float qi = (int16_t)(payload[i + 4] | payload[i + 5] << 8) / 16384.0f;
        float qj = (int16_t)(payload[i + 6] | payload[i + 7] << 8) / 16384.0f;
        float qk = (int16_t)(payload[i + 8] | payload[i + 9] << 8) / 16384.0f;
        float qr = (int16_t)(payload[i + 10] | payload[i + 11] << 8) / 16384.0f;
        q[0] = qr; q[1] = qi; q[2] = qj; q[3] = qk;
        i += 12;
      } else {
        break;  // unknown report: skip the rest of this packet
      }
    }
  }

  void emit(uint32_t tUs, float gx, float gy, float gz) {
    char line[160];
    int n = snprintf(line, sizeof(line), "%lu,%u,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.5f,%.5f,%.5f,%.5f\n", (unsigned long)tUs, id, gx,
                     gy, gz, a[0], a[1], a[2], q[0], q[1], q[2], q[3]);
    writeLine(line, n);
    lastSampleMs = millis();
  }
};

Bno jaw{"jaw", 0, &Wire1};
Bno head{"head", 1, &Wire};

// Each sensor has its own I2C bus, so each gets its own task (and core): a slow or
// restarting sensor can no longer delay the other one.
static void sensorTask(void *arg) {
  Bno *b = (Bno *)arg;
  b->begin();
  for (;;) {
    if (!b->poll()) vTaskDelay(1);  // nothing waiting: yield (also feeds the watchdog)
  }
}

void setup() {
  Serial.setTxBufferSize(8192);  // 2 x 100 lines/s must never block on the UART
  Serial.begin(921600);
  serialMux = xSemaphoreCreateMutex();
  i2cMux = xSemaphoreCreateMutex();
  pinMode(LED_PIN, OUTPUT);
  Wire1.begin(JAW_SDA, JAW_SCL);
  Wire1.setClock(JAW_CLOCK_HZ);
  Wire1.setBufferSize(128);
  Wire.begin(HEAD_SDA, HEAD_SCL);
  Wire.setClock(HEAD_CLOCK_HZ);
  Wire.setBufferSize(128);
  delay(500);
  xTaskCreatePinnedToCore(sensorTask, "jaw", 6144, &jaw, 2, nullptr, 0);
  xTaskCreatePinnedToCore(sensorTask, "head", 6144, &head, 2, nullptr, 1);
}

// The laptop can drive the onboard LED to show sensor orientation: "O<m>\n" with
//   m=1 slow blink (neither sensor oriented right), 2 fast blink (one), 3 solid (both),
//   0 = back to normal. Without a fresh command for 2 s it falls back to normal, where
//   the LED means "both IMUs are streaming".
static uint8_t orientMode = 0;
static uint32_t orientAt = 0;

static void pollCommands() {
  static char cmd[8];
  static size_t n = 0;
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\n') {
      if (n == 2 && cmd[0] == 'O' && cmd[1] >= '0' && cmd[1] <= '3') {
        orientMode = cmd[1] - '0';
        orientAt = millis();
      }
      n = 0;
    } else if (ch != '\r' && n < sizeof(cmd)) {
      cmd[n++] = ch;
    }
  }
}

void loop() {
  pollCommands();
  uint32_t now = millis();
  if (orientMode && now - orientAt > 2000) orientMode = 0;
  bool on;
  switch (orientMode) {
    case 1: on = (now / 700) % 2 == 0; break;
    case 2: on = (now / 120) % 2 == 0; break;
    case 3: on = true; break;
    default: on = jaw.up && head.up;
  }
  digitalWrite(LED_PIN, on ? HIGH : LOW);
  delay(20);
}
