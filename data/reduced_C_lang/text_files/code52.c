#include<stdio.h>

long int factorial(int x) {
    int i;
    long int fac;
    fac = x;
    for (i = 1; i < x; i++) {
        fac = fac * (x - i);
    }
    return fac;
}

int main() {
    long int f1, f2, f3;
    int n;
    float C;
    scanf("%d", & n);
    f1 = factorial(2 * n);
    f2 = factorial(n + 1);
    f3 = factorial(n);
    C = f1 / (f2 * f3);
    printf("%0.2f", C);
    return 0;
}