#include <stdio.h>

int fib(int number) {
    if (number == 1 || number == 2) return 1;
    else return fib(number - 1) + fib(number - 2);
}

int main() {
    int number;

    printf("Number: ");
    scanf("%d", & number);

    printf("%d \n", fib(number));

    return 0;
}