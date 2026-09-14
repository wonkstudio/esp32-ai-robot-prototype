#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ESP32Servo.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_SDA 21
#define OLED_SCL 22
#define SERVO_YAW_PIN 13
#define SERVO_PITCH_PIN 14

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
Servo servoYaw;
Servo servoPitch;

float currentYaw = 0.0f;
float currentPitch = 0.0f;
int currentFace = 0;
String lineBuffer;

float clampFloat(float value, float low, float high) {
  return max(low, min(high, value));
}

void drawFace(int face) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  if (face == 0) { // idle
    display.fillRoundRect(22, 24, 28, 22, 7, SSD1306_WHITE);
    display.fillRoundRect(78, 24, 28, 22, 7, SSD1306_WHITE);
  } else if (face == 1 || face == 2) { // happy / big smile
    display.drawLine(22, 36, 31, 27, SSD1306_WHITE);
    display.drawLine(31, 27, 42, 27, SSD1306_WHITE);
    display.drawLine(42, 27, 51, 36, SSD1306_WHITE);
    display.drawLine(77, 36, 86, 27, SSD1306_WHITE);
    display.drawLine(86, 27, 97, 27, SSD1306_WHITE);
    display.drawLine(97, 27, 106, 36, SSD1306_WHITE);
    if (face == 2) {
      display.fillRoundRect(44, 43, 40, 15, 7, SSD1306_WHITE);
      display.fillRoundRect(49, 46, 30, 8, 4, SSD1306_BLACK);
    }
  } else if (face == 3) { // surprised
    display.drawCircle(36, 31, 13, SSD1306_WHITE);
    display.drawCircle(92, 31, 13, SSD1306_WHITE);
    display.fillCircle(36, 31, 4, SSD1306_WHITE);
    display.fillCircle(92, 31, 4, SSD1306_WHITE);
    display.drawCircle(64, 53, 5, SSD1306_WHITE);
  } else { // angry
    display.drawLine(20, 20, 51, 31, SSD1306_WHITE);
    display.drawLine(77, 31, 108, 20, SSD1306_WHITE);
    display.fillRoundRect(26, 34, 20, 11, 4, SSD1306_WHITE);
    display.fillRoundRect(82, 34, 20, 11, 4, SSD1306_WHITE);
  }
  display.display();
}

void applyState(float yaw, float pitch, int face) {
  currentYaw = clampFloat(yaw, -60.0f, 60.0f);
  currentPitch = clampFloat(pitch, -25.0f, 25.0f);
  currentFace = constrain(face, 0, 4);

  // Signed simulation angle 0 deg maps to the SG90 center at 90 deg.
  servoYaw.write((int)round(90.0f + currentYaw));
  servoPitch.write((int)round(90.0f + currentPitch));
  drawFace(currentFace);
}

void printState() {
  Serial.print("STATE,");
  Serial.print(currentYaw, 3);
  Serial.print(",");
  Serial.print(currentPitch, 3);
  Serial.print(",");
  Serial.println(currentFace);
}

void handleLine(String line) {
  line.trim();
  if (line == "PING") {
    Serial.println("PONG");
    return;
  }
  if (line == "GET") {
    printState();
    return;
  }
  if (line.startsWith("SET,")) {
    int comma1 = line.indexOf(',', 4);
    int comma2 = line.indexOf(',', comma1 + 1);
    if (comma1 < 0 || comma2 < 0) {
      Serial.println("ERR,BAD_SET");
      return;
    }
    float yaw = line.substring(4, comma1).toFloat();
    float pitch = line.substring(comma1 + 1, comma2).toFloat();
    int face = line.substring(comma2 + 1).toInt();
    applyState(yaw, pitch, face);
    printState();
    return;
  }
  Serial.println("ERR,UNKNOWN_COMMAND");
}

void setup() {
  Serial.begin(115200);
  Wire.begin(OLED_SDA, OLED_SCL);
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);

  servoYaw.setPeriodHertz(50);
  servoPitch.setPeriodHertz(50);
  servoYaw.attach(SERVO_YAW_PIN, 500, 2400);
  servoPitch.attach(SERVO_PITCH_PIN, 500, 2400);
  applyState(0.0f, 0.0f, 0);
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      handleLine(lineBuffer);
      lineBuffer = "";
    } else if (c != '\r') {
      lineBuffer += c;
      if (lineBuffer.length() > 120) {
        lineBuffer = "";
        Serial.println("ERR,LINE_TOO_LONG");
      }
    }
  }
}
