#include <Arduino.h>
#include <Wire.h>
#include <SparkFun_BNO08x_Arduino_Library.h>

// The BNO08x driver (SH2) is a singleton, so only one sensor can be open at a
// time. Every print interval we open each sensor in turn, grab one sample, and
// close it again.

constexpr int SDA1_PIN = 21;  // sensor 1 bus (Wire)
constexpr int SCL1_PIN = 22;
constexpr int SDA2_PIN = 18;  // sensor 2 bus (Wire1)
constexpr int SCL2_PIN = 19;
constexpr int BTN_PIN = 0;
constexpr int LED_PIN = 2;
constexpr uint32_t DEBOUNCE_MS = 50;
constexpr uint32_t PRINT_INTERVAL_MS = 3000;
constexpr uint16_t REPORT_INTERVAL_MS = 10;     // 100 Hz
constexpr uint32_t SAMPLE_TIMEOUT_MS = 1500;

BNO08x imu;
bool driverOpen = false;

struct Sample {
  bool ok = false;
  float ax = 0, ay = 0, az = 0, gx = 0, gy = 0, gz = 0;
};

bool printing = true;
bool lastReading = HIGH;
bool stableState = HIGH;
uint32_t lastChange = 0;
uint32_t lastPrint = 0;

static void pollButton() {
  bool reading = digitalRead(BTN_PIN);
  if (reading != lastReading) lastChange = millis();
  lastReading = reading;
  if (millis() - lastChange > DEBOUNCE_MS && reading != stableState) {
    stableState = reading;
    if (stableState == LOW) {
      printing = !printing;
      digitalWrite(LED_PIN, printing ? HIGH : LOW);
    }
  }
}

// Returns true and fills `s` with one accel + gyro reading from the sensor on `bus`.
static bool readSensor(TwoWire &bus, Sample &s) {
  s.ok = false;
  if (driverOpen) { sh2_close(); driverOpen = false; }  // release the previous sensor
  bool up = imu.begin(0x4A, bus, -1, -1) || imu.begin(0x4B, bus, -1, -1);
  if (!up) return false;
  driverOpen = true;
  if (!imu.enableAccelerometer(REPORT_INTERVAL_MS)) return false;
  if (!imu.enableGyro(REPORT_INTERVAL_MS)) return false;

  bool gotA = false, gotG = false;
  uint32_t start = millis();
  while (!(gotA && gotG) && millis() - start < SAMPLE_TIMEOUT_MS) {
    if (imu.getSensorEvent()) {
      uint8_t id = imu.getSensorEventID();
      if (id == SENSOR_REPORTID_ACCELEROMETER) {
        s.ax = imu.getAccelX(); s.ay = imu.getAccelY(); s.az = imu.getAccelZ();
        gotA = true;
      } else if (id == SENSOR_REPORTID_GYROSCOPE_CALIBRATED) {
        s.gx = imu.getGyroX(); s.gy = imu.getGyroY(); s.gz = imu.getGyroZ();
        gotG = true;
      }
    }
    pollButton();
    delay(1);
  }
  s.ok = gotA && gotG;
  return s.ok;
}

static void printSample(const char *name, const Sample &s) {
  if (s.ok) {
    Serial.printf("%s Accel (m/s^2): %.3f, %.3f, %.3f | Gyro (rad/s): %.3f, %.3f, %.3f\n",
                  name, s.ax, s.ay, s.az, s.gx, s.gy, s.gz);
  } else {
    Serial.printf("%s: BNO085 not found\n", name);
  }
}

void setup() {
  Serial.begin(921600);
  pinMode(BTN_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);
  delay(500);

  Wire.begin(SDA1_PIN, SCL1_PIN);
  Wire.setClock(400000);
  Wire1.begin(SDA2_PIN, SCL2_PIN);
  Wire1.setClock(400000);
}

void loop() {
  pollButton();
  if (printing && millis() - lastPrint >= PRINT_INTERVAL_MS) {
    lastPrint = millis();
    Sample s1, s2;
    readSensor(Wire, s1);
    readSensor(Wire1, s2);
    if (printing) {  // button may have been pressed while sampling
      printSample("IMU1", s1);
      printSample("IMU2", s2);
    }
  }
}
