/*
  DESKRO real-style 2DOF gaze controller reference
  ------------------------------------------------
  ESP32 DevKit V1
  Yaw SG90 signal   -> GPIO 13
  Pitch SG90 signal -> GPIO 14

  Library:
    ESP32Servo

  IMPORTANT:
  - This is a reference controller for later hardware testing.
  - Do not force a servo against a mechanical stop.
  - Real center offsets, signs, and collision limits must be calibrated after
    the pan/tilt bracket and head are physically assembled.

  Serial commands:
    T <yaw_deg> <pitch_deg>
      Example: T 25 8

    C
      Return to center.
*/

#include <Arduino.h>
#include <ESP32Servo.h>
#include <math.h>

static constexpr int YAW_PIN = 13;
static constexpr int PITCH_PIN = 14;

// Relative DESKRO head angles.
// Conservative until the real bracket is measured.
static constexpr float YAW_MIN_DEG = -45.0f;
static constexpr float YAW_MAX_DEG =  45.0f;
static constexpr float PITCH_MIN_DEG = -15.0f;
static constexpr float PITCH_MAX_DEG =  20.0f;

// Mount calibration.
// Change these after physical measurement.
static constexpr float YAW_SERVO_CENTER_DEG = 90.0f;
static constexpr float PITCH_SERVO_CENTER_DEG = 90.0f;
static constexpr float YAW_SIGN = +1.0f;
static constexpr float PITCH_SIGN = +1.0f;

// Friendly motion profile, intentionally below raw servo capability.
static constexpr float YAW_MAX_SPEED_DEG_S = 60.0f;
static constexpr float PITCH_MAX_SPEED_DEG_S = 45.0f;
static constexpr float YAW_MAX_ACCEL_DEG_S2 = 180.0f;
static constexpr float PITCH_MAX_ACCEL_DEG_S2 = 130.0f;

// 50 Hz command update matches normal hobby-servo timing well.
static constexpr uint32_t CONTROL_PERIOD_MS = 20;

// Short attention-switch reaction delay.
static constexpr uint32_t REACTION_DELAY_MS = 200;

Servo yawServo;
Servo pitchServo;

struct AxisState {
  float commandDeg = 0.0f;    // relative robot angle
  float velocityDegS = 0.0f;
  float desiredDeg = 0.0f;
};

AxisState yawAxis;
AxisState pitchAxis;

uint32_t nextControlMs = 0;
uint32_t reactionUntilMs = 0;

static float clampf(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

static void advanceAxis(
  AxisState &axis,
  float maxSpeedDegS,
  float maxAccelDegS2,
  float dt
) {
  const float error = axis.desiredDeg - axis.commandDeg;

  if (fabsf(error) < 0.0001f && fabsf(axis.velocityDegS) < 0.001f) {
    axis.commandDeg = axis.desiredDeg;
    axis.velocityDegS = 0.0f;
    return;
  }

  // Braking-distance profile:
  // v <= sqrt(2*a*distance), so it slows before the target.
  const float brakingSpeed = sqrtf(
    fmaxf(0.0f, 2.0f * maxAccelDegS2 * fabsf(error))
  );

  const float signedTargetSpeed =
    (error >= 0.0f ? 1.0f : -1.0f)
    * fminf(maxSpeedDegS, brakingSpeed);

  const float maxDeltaV = maxAccelDegS2 * dt;
  const float deltaV = clampf(
    signedTargetSpeed - axis.velocityDegS,
    -maxDeltaV,
    +maxDeltaV
  );

  axis.velocityDegS += deltaV;
  const float nextCommand = axis.commandDeg + axis.velocityDegS * dt;

  // Do not numerically cross the desired target.
  if (
    (error > 0.0f && nextCommand >= axis.desiredDeg) ||
    (error < 0.0f && nextCommand <= axis.desiredDeg)
  ) {
    axis.commandDeg = axis.desiredDeg;
    axis.velocityDegS = 0.0f;
  } else {
    axis.commandDeg = nextCommand;
  }
}

static void writeServos() {
  const float yawPhysical =
    YAW_SERVO_CENTER_DEG + YAW_SIGN * yawAxis.commandDeg;
  const float pitchPhysical =
    PITCH_SERVO_CENTER_DEG + PITCH_SIGN * pitchAxis.commandDeg;

  // Relative limits above keep these safely near servo center.
  yawServo.write((int)roundf(yawPhysical));
  pitchServo.write((int)roundf(pitchPhysical));
}

static void setGazeTarget(float yawDeg, float pitchDeg, bool attentionSwitch) {
  yawAxis.desiredDeg = clampf(yawDeg, YAW_MIN_DEG, YAW_MAX_DEG);
  pitchAxis.desiredDeg = clampf(
    pitchDeg,
    PITCH_MIN_DEG,
    PITCH_MAX_DEG
  );

  if (attentionSwitch) {
    reactionUntilMs = millis() + REACTION_DELAY_MS;
  }
}

static void parseSerialCommand(String line) {
  line.trim();
  if (line.length() == 0) return;

  if (line.equalsIgnoreCase("C")) {
    setGazeTarget(0.0f, 0.0f, true);
    Serial.println("OK CENTER");
    return;
  }

  if (line.startsWith("T ") || line.startsWith("t ")) {
    float yaw = 0.0f;
    float pitch = 0.0f;

    const int parsed = sscanf(
      line.c_str(),
      "%*c %f %f",
      &yaw,
      &pitch
    );

    if (parsed == 2) {
      setGazeTarget(yaw, pitch, true);
      Serial.printf(
        "OK TARGET yaw=%.1f pitch=%.1f\n",
        yawAxis.desiredDeg,
        pitchAxis.desiredDeg
      );
    } else {
      Serial.println("ERR use: T <yaw_deg> <pitch_deg>");
    }
    return;
  }

  Serial.println("Commands: T <yaw> <pitch> | C");
}

void setup() {
  Serial.begin(115200);

  // Keep normal hobby-servo 50 Hz behavior.
  yawServo.setPeriodHertz(50);
  pitchServo.setPeriodHertz(50);

  yawServo.attach(YAW_PIN);
  pitchServo.attach(PITCH_PIN);

  yawAxis.commandDeg = 0.0f;
  yawAxis.desiredDeg = 0.0f;
  pitchAxis.commandDeg = 0.0f;
  pitchAxis.desiredDeg = 0.0f;

  writeServos();

  nextControlMs = millis();

  Serial.println();
  Serial.println("DESKRO real gaze controller ready.");
  Serial.println("T <yaw_deg> <pitch_deg>");
  Serial.println("Example: T 25 8");
  Serial.println("C = center");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    parseSerialCommand(line);
  }

  const uint32_t now = millis();
  if ((int32_t)(now - nextControlMs) < 0) {
    return;
  }

  nextControlMs += CONTROL_PERIOD_MS;

  // Hold the current pose during the short reaction delay.
  if ((int32_t)(now - reactionUntilMs) < 0) {
    writeServos();
    return;
  }

  const float dt = CONTROL_PERIOD_MS / 1000.0f;

  advanceAxis(
    yawAxis,
    YAW_MAX_SPEED_DEG_S,
    YAW_MAX_ACCEL_DEG_S2,
    dt
  );
  advanceAxis(
    pitchAxis,
    PITCH_MAX_SPEED_DEG_S,
    PITCH_MAX_ACCEL_DEG_S2,
    dt
  );

  writeServos();
}
