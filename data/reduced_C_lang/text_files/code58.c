#include<stdio.h>
#include<stdlib.h>

int fib(int n) {
    if (n < 0) {
        printf("\nNo Such term !\n");
        exit(0);
    }
    int f[n + 2];
    int i;

    f[0] = 0;
    f[1] = 1;

    for (i = 2; i <= n; i++) {
        f[i] = f[i - 1] + f[i - 2];
    }

    return f[n];
}

int main() {
    int number;

    printf("Enter the value of n(n starts from 0 ): ");
    scanf("%d", & number);

    printf("The nth term is : %d \n", fib(number));

    return 0;
}