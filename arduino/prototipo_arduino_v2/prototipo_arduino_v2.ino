/*
 * prototipo_arduino_v2_no_buzzer.ino
 * Sistema de clasificación con DOS servos (sin LCD, sin Buzzer)
 *
 * ─── Hardware ────────────────────────────────────────────────────────
 * Servo 1 (rampa)   : D9   — gira la rampa al destino A/B/C
 * Servo 2 (puerta)  : D10  — abre/cierra la puerta para soltar el objeto
 * LED verde         : D6
 * LED rojo          : D7
 *
 * ─── Librerías necesarias ────────────────────────────────────────────
 * - Servo              (incluida en el IDE de Arduino)
 *
 * ─── Protocolo Serial (9600 baud) ───────────────────────────────────
 * Recibe:
 * 'A' → rampa 120° (destino A / ROJO)
 * 'B' → rampa  90° (destino B / VERDE)  ← posición central
 * 'C' → rampa  60° (destino C / AZUL)
 * 'X' → cancelar (LED rojo + todo vuelve al centro)
 *
 * Envía: "ACK\n" después de procesar cada comando válido.
 */

#include <Servo.h>

// ════════════════════════════════════════════════════════════════════
// CONFIGURACIÓN
// ════════════════════════════════════════════════════════════════════

// ─── Pines ───────────────────────────────────────────────────────────
#define PIN_SERVO_RAMP   9    // Servo 1 — rampa
#define PIN_SERVO_DOOR  10    // Servo 2 — puerta
#define PIN_LED_G        6
#define PIN_LED_R        7

// ─── Ángulos servo 1 (rampa) ─────────────────────────────────────────
#define RAMP_A          120   // +30° físicos  → destino A (ROJO)
#define RAMP_B           90   //   0° físicos  → destino B (VERDE / centro)
#define RAMP_C           60   // -30° físicos  → destino C (AZUL)
#define RAMP_CENTER      90   // Posición de reposo

// ─── Ángulos servo 2 (puerta) ────────────────────────────────────────
#define DOOR_CLOSED       90   // Puerta cerrada
#define DOOR_OPEN        0   // Puerta abierta 90°

// ─── Tiempos (ms) — no bloqueantes ───────────────────────────────────
#define RAMP_SETTLE_MS   600  // Espera tras mover la rampa antes de abrir puerta
#define DOOR_OPEN_MS     1200  // Tiempo que la puerta permanece abierta
#define DOOR_CLOSE_MS    800  // Tiempo para que la puerta cierre del todo
#define RAMP_RETURN_MS   800  // Tiempo para que la rampa vuelva al centro
#define EMERGENCY_HOLD_MS 400 // Tiempo que dura el estado de cancelación activo

// ════════════════════════════════════════════════════════════════════
// OBJETOS
// ════════════════════════════════════════════════════════════════════
Servo servoRamp;
Servo servoDoor;

// ════════════════════════════════════════════════════════════════════
// MÁQUINA DE ESTADOS
// ════════════════════════════════════════════════════════════════════
enum State {
  ST_IDLE,          // Esperando comando
  ST_RAMP_MOVING,   // Rampa girando al destino
  ST_DOOR_OPENING,  // Puerta abriéndose
  ST_DOOR_CLOSING,  // Puerta cerrándose
  ST_RAMP_RETURN,   // Rampa volviendo al centro
  ST_CANCEL         // Cancelación activa
};

State         state        = ST_IDLE;
unsigned long stateStartMs = 0;
char          currentCmd   = 0;

// ════════════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(9600);

  pinMode(PIN_LED_G,  OUTPUT);
  pinMode(PIN_LED_R,  OUTPUT);
  digitalWrite(PIN_LED_G,  LOW);
  digitalWrite(PIN_LED_R,  LOW);

  // Servos en posición de reposo
  servoRamp.attach(PIN_SERVO_RAMP);
  servoDoor.attach(PIN_SERVO_DOOR);
  servoRamp.write(RAMP_CENTER);
  servoDoor.write(DOOR_CLOSED);
}

// ════════════════════════════════════════════════════════════════════
void loop() {
  // ── Leer serial ───────────────────────────────────────────────────
  if (Serial.available() > 0) {
    char cmd = (char)toupper(Serial.read());
    handleCommand(cmd);
  }

  // ── FSM no bloqueante ─────────────────────────────────────────────
  unsigned long now = millis();

  switch (state) {

    // ── Rampa moviéndose al destino ──────────────────────────────────
    case ST_RAMP_MOVING:
      if (now - stateStartMs >= RAMP_SETTLE_MS) {
        // Rampa lista -> abrir compuerta para soltar objeto
        servoDoor.write(DOOR_OPEN);
        state        = ST_DOOR_OPENING;
        stateStartMs = now;
      }
      break;

    // ── Puerta abierta dejando caer el objeto ─────────────────────────
    case ST_DOOR_OPENING:
      if (now - stateStartMs >= DOOR_OPEN_MS) {
        // Objeto liberado -> cerrar compuerta
        servoDoor.write(DOOR_CLOSED);
        state        = ST_DOOR_CLOSING;
        stateStartMs = now;
      }
      break;

    // ── Puerta regresando a su posición cerrada ───────────────────────
    case ST_DOOR_CLOSING:
      if (now - stateStartMs >= DOOR_CLOSE_MS) {
        // Compuerta cerrada -> regresar rampa de dirección al centro
        servoRamp.write(RAMP_CENTER);
        state        = ST_RAMP_RETURN;
        stateStartMs = now;
      }
      break;

    // ── Rampa volviendo al centro ─────────────────────────────────────
    case ST_RAMP_RETURN:
      if (now - stateStartMs >= RAMP_RETURN_MS) {
        // Proceso completo -> apagar indicador y volver a espera
        digitalWrite(PIN_LED_G, LOW);
        state = ST_IDLE;
      }
      break;

    // ── Estado de Cancelación de Emergencia ───────────────────────────
    case ST_CANCEL:
      if (now - stateStartMs >= EMERGENCY_HOLD_MS) {
        digitalWrite(PIN_LED_R, LOW);
        
        // Ambos servos regresan a la posición segura de reposo
        servoDoor.write(DOOR_CLOSED);
        servoRamp.write(RAMP_CENTER);
        state = ST_IDLE;
      }
      break;

    case ST_IDLE:
    default:
      break;
  }
}

// ════════════════════════════════════════════════════════════════════
// Procesa los caracteres recibidos por el puerto serie
// ════════════════════════════════════════════════════════════════════
void handleCommand(char cmd) {
  // Ignorar comandos de clasificación nuevos si ya hay uno en curso
  if (state != ST_IDLE && cmd != 'X') return;

  if (cmd == 'A' || cmd == 'B' || cmd == 'C') {
    currentCmd = cmd;
    
    // Asignar ángulo de destino para la rampa (Servo 1)
    int rampAngle = (cmd == 'A') ? RAMP_A : (cmd == 'B') ? RAMP_B : RAMP_C;
    servoRamp.write(rampAngle);

    // Indicadores visuales
    digitalWrite(PIN_LED_G, HIGH);
    digitalWrite(PIN_LED_R, LOW);

    // Iniciar la secuencia no bloqueante
    state        = ST_RAMP_MOVING;
    stateStartMs = millis();

    Serial.println("ACK");
  }
  else if (cmd == 'X') {
    // Cancelación inmediata desde cualquier estado operativo
    digitalWrite(PIN_LED_R,  HIGH);
    digitalWrite(PIN_LED_G,  LOW);

    state        = ST_CANCEL;
    stateStartMs = millis();

    Serial.println("ACK");
  }
}