#include <stdio.h>

int power(int x, unsigned int y) {
    if (y == 0)
        return 1;
    if (y % 2 == 0)
        return power(x, y / 2) * power(x, y / 2);
    return x * power(x, y / 2) * power(x, y / 2);
}

int order(int x) {
    int n = 0;
    while (x) {
        n++;
        x = x / 10;
    }
    return n;
}

int isArmstrong(int x) {
    int n = order(x);
    int temp = x, sum = 0;
    while (temp) {
        int r = temp % 10;
        sum += power(r, n);
        temp = temp / 10;
    }

    if (sum == x)
        return 1;
    else
        return 0;
}

int main() {
    int x = 153;
    if (isArmstrong(x) == 1)
        printf("True\n");
    else
        printf("False\n");

    x = 1253;
    if (isArmstrong(x) == 1)
        printf("True\n");
    else
        printf("False\n");

    return 0;
}