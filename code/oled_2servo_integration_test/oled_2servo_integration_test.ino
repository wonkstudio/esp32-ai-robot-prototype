#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ESP32Servo.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

#define OLED_SDA 21
#define OLED_SCL 22

#define SERVO1_PIN 13
#define SERVO2_PIN 14

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

Servo servo1;
Servo servo2;

void drawIdleFace() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(1);
  display.setCursor(50, 0);
  display.println("IDLE");

  display.fillRoundRect(22, 24, 28, 24, 8, SSD1306_WHITE);
  display.fillRoundRect(78, 24, 28, 24, 8, SSD1306_WHITE);

  display.display();
}

void drawHappyFace() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(1);
  display.setCursor(45, 0);
  display.println("HAPPY");

  display.drawLine(22, 36, 30, 28, SSD1306_WHITE);
  display.drawLine(30, 28, 42, 28, SSD1306_WHITE);
  display.drawLine(42, 28, 50, 36, SSD1306_WHITE);

  display.drawLine(78, 36, 86, 28, SSD1306_WHITE);
  display.drawLine(86, 28, 98, 28, SSD1306_WHITE);
  display.drawLine(98, 28, 106, 36, SSD1306_WHITE);

  display.fillRoundRect(44, 43, 40, 15, 7, SSD1306_WHITE);
  display.fillRoundRect(49, 46, 30, 8, 4, SSD1306_BLACK);

  display.display();
}

void drawAngryFace() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(1);
  display.setCursor(47, 0);
  display.println("ANGRY");

  display.drawLine(20, 22, 52, 32, SSD1306_WHITE);
  display.drawLine(20, 23, 52, 33, SSD1306_WHITE);
  display.drawLine(76, 32, 108, 22, SSD1306_WHITE);
  display.drawLine(76, 33, 108, 23, SSD1306_WHITE);

  display.fillRoundRect(26, 35, 20, 11, 4, SSD1306_WHITE);
  display.fillRoundRect(82, 35, 20, 11, 4, SSD1306_WHITE);

  display.drawLine(45, 56, 55, 51, SSD1306_WHITE);
  display.drawLine(55, 51, 73, 51, SSD1306_WHITE);
  display.drawLine(73, 51, 83, 56, SSD1306_WHITE);

  display.display();
}

void bothCenter() {
  servo1.write(90);
  servo2.write(90);
  delay(500);
}

void bothWaveHappy() {
  servo1.write(60);
  servo2.write(120);
  delay(350);

  servo1.write(120);
  servo2.write(60);
  delay(350);

  servo1.write(60);
  servo2.write(120);
  delay(350);

  servo1.write(120);
  servo2.write(60);
  delay(350);

  bothCenter();
}

void bothAngryMove() {
  servo1.write(45);
  servo2.write(45);
  delay(180);

  servo1.write(135);
  servo2.write(135);
  delay(180);

  servo1.write(45);
  servo2.write(45);
  delay(180);

  servo1.write(135);
  servo2.write(135);
  delay(180);

  bothCenter();
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin(OLED_SDA, OLED_SCL);

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED not found");
    while (true) {
      delay(1000);
    }
  }

  servo1.setPeriodHertz(50);
  servo2.setPeriodHertz(50);

  servo1.attach(SERVO1_PIN, 500, 2400);
  servo2.attach(SERVO2_PIN, 500, 2400);

  Serial.println("OLED + 2 Servo integration test start");
}

void loop() {
  Serial.println("Mode: IDLE");
  drawIdleFace();
  bothCenter();
  delay(1000);

  Serial.println("Mode: HAPPY BOTH WAVE");
  drawHappyFace();
  bothWaveHappy();
  delay(1000);

  Serial.println("Mode: ANGRY BOTH MOVE");
  drawAngryFace();
  bothAngryMove();
  delay(1000);
}