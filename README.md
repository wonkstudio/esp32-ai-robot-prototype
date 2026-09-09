# ESP32 AI Robot Prototype

An interactive desktop robot prototype built with an ESP32, an OLED display, and servo motors. The project is being developed incrementally by testing each component first and then combining them into coordinated robot behaviors.

## Demo

![OLED face and servo integration demo](media/oled_servo_integration_demo.gif)

[Watch the original video](media/oled_servo_integration_demo.mp4)

## Current Milestones

- Detected the ESP32 on Windows through `COM6`
- Uploaded Arduino sketches to the ESP32 successfully
- Displayed text on an SSD1306 OLED
- Created `IDLE`, `HAPPY`, and `ANGRY` face animations
- Tested an SG90 servo motor independently
- Completed an OLED and servo integration test

## Hardware

- ESP32 DevKit V1
- 0.96-inch SSD1306 OLED (I2C, 128×64)
- SG90 servo motor
- Breadboard and jumper wires
- HC-SR04 ultrasonic distance sensor (next step)
- INMP441 I2S microphone (planned)
- MAX98357A I2S amplifier and 3 W, 4 Ω speaker (planned)

## Current Wiring

### SSD1306 OLED

| OLED | ESP32 |
|---|---|
| VCC | 3V3 |
| GND | GND |
| SDA | GPIO21 |
| SCL | GPIO22 |

### SG90 Servo

| SG90 wire | ESP32 / power |
|---|---|
| Brown | GND |
| Red | 5V |
| Orange | GPIO13 (GPIO14 for the second servo) |

Use a stable external 5 V supply when operating multiple servos. When using an external supply, connect its GND to the ESP32 GND so that all components share a common ground.

## Running the Integration Test

1. Open `code/oled_2servo_integration_test/oled_2servo_integration_test.ino` in the Arduino IDE.
2. Install these libraries through the Arduino Library Manager:
   - Adafruit GFX Library
   - Adafruit SSD1306
   - ESP32Servo
3. Select the ESP32 board and its serial port. The test system used `COM6`, but the port number may be different on another computer.
4. Upload the sketch.
5. Confirm that the OLED cycles through the three faces while the servo motors move.

## HC-SR04 Safety Note

The HC-SR04 `ECHO` pin outputs a signal of approximately 5 V, while ESP32 GPIO pins use 3.3 V logic. Do not connect `ECHO` directly to an ESP32 GPIO. Use two resistors as a voltage divider first.

Example:

```text
HC-SR04 ECHO ── 1 kΩ ──┬── ESP32 input pin
                        │
                       2 kΩ
                        │
                       GND
```

The HC-SR04 integration is currently on hold until the required resistors are available.

## Repository Structure

```text
esp32-ai-robot-prototype/
├─ README.md
├─ code/
│  └─ oled_2servo_integration_test/
│     └─ oled_2servo_integration_test.ino
└─ media/
   ├─ oled_servo_integration_demo.gif
   └─ oled_servo_integration_demo.mp4
```

## Next Steps

- Run two servo motors reliably from an external 5 V supply
- Connect the HC-SR04 through a safe voltage divider
- Change the OLED expression and servo movement based on distance
- Test audio input with the INMP441 microphone
- Test audio output with the MAX98357A amplifier and speaker
