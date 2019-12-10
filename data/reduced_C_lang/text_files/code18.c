#include <stdio.h>
#include <stdlib.h>
#define MAXBITS 100

int main() {

    int inputNumber;

    int re;

    int bits[MAXBITS];

    int j;
    int i = 0;

    printf("\t\tConverter decimal --> binary\n\n");

    printf("\nenter a positive integer number: ");
    scanf("%d", & inputNumber);

    if (inputNumber < 0) {
        printf("only positive integers >= 0\n");
        return 1;
    }

    while (inputNumber > 0) {

        re = inputNumber % 2;

        inputNumber = inputNumber / 2;

        bits[i] = re;
        i++;

    }

    printf("\n the number in binary is: ");

    for (j = i - 1; j >= 0; j--) {
        printf("%d", bits[j]);
    }

    if (i == 0) {
        printf("0");
    }

    return 0;
}