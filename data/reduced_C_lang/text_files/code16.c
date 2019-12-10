#include <stdio.h>

int main() {

    int remainder, number = 0, decimal_number = 0, temp = 1;
    printf("/n Enter any binary number= ");
    scanf("%d", & number);

    while (number > 0) {

        remainder = number % 10;
        number = number / 10;
        decimal_number += remainder * temp;
        temp = temp * 2;

    }

    printf("%d\n", decimal_number);
}