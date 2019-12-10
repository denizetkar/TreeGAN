#include <stdio.h>

struct pid {
    float kP;
    float kI;
    float kD;

    float lastError;
    float integral;
};

float pid_step(struct pid * controller, float dt, float error) {
    float p = error * controller -> kP;

    controller -> integral += error * dt * controller -> kI;

    float d = dt == 0 ? 0 : ((error - controller -> lastError) / dt) * controller -> kD;
    controller -> lastError = error;

    return p + controller -> integral + d;
}

int main() {
    printf("PID Controller Example\n");

    struct pid controller = {
        .lastError = 0,
        .integral = 0
    };

    printf("Please enter controller gains in format kP, kI, KD. For example, \"1.2 2.1 3.2\"\n> ");
    scanf("%f %f %f", & controller.kP, & controller.kI, & controller.kD);
    printf("Using kP: %f, kI: %f, kD: %f\n", controller.kP, controller.kI, controller.kD);

    float time_step = 1;

    float error_value;
    while (1) {
        printf("Enter error value\n>");
        scanf("%f", & error_value);

        float output = pid_step( & controller, time_step, error_value);
        printf("Output: %f\n", output);
    }
}