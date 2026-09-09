\# ESP32 AI Robot Prototype



ESP32 기반 인터랙티브 로봇 프로토타입 프로젝트입니다.



\## 현재 성공한 기능



\- ESP32 Windows COM 포트 인식 성공

\- Arduino IDE 업로드 성공

\- Serial Monitor 출력 성공

\- OLED SSD1306 화면 출력 성공

\- OLED 얼굴 애니메이션 성공

\- SG90 서보모터 1개 동작 성공

\- OLED + 서보 통합 테스트 성공



\## 사용 부품



\- ESP32 DevKit V1

\- OLED SSD1306 0.96 inch

\- SG90 Servo Motor x2

\- INMP441 I2S Microphone

\- MAX98357A I2S Amplifier

\- 3W 4Ω Speaker

\- HC-SR04 Ultrasonic Sensor

\- Breadboard

\- Jumper wires



\## 현재 배선



\### OLED SSD1306



| OLED | ESP32 |

|---|---|

| VCC | 3V3 |

| GND | GND |

| SDA | GPIO21 |

| SCL | GPIO22 |



\### SG90 Servo



| Servo | ESP32 |

|---|---|

| Brown | GND |

| Red | VIN / 5V |

| Orange | GPIO13 / GPIO14 |



\## 다음 목표



\- 서보 2개 안정적으로 연결

\- 외부 5V 전원 구성

\- HC-SR04 거리센서 안전 연결

\- 거리값에 따라 OLED 표정과 서보 움직임 변경

\- 마이크 입력 테스트

\- 스피커 출력 테스트

