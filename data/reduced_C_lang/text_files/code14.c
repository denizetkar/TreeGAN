#include<stdio.h>
#include <stdlib.h>
#include<math.h>

int main() {

    int * ARRAY = NULL, ARRAY_LENGTH, i, TEMPORARY_ELEMENT, isSorted = 0;
    float MEAN = 0, VARIANCE = 0, STAND;

    printf("Enter no. for Random Numbers :");
    scanf("%d", & ARRAY_LENGTH);
    ARRAY = (int * ) realloc(ARRAY, ARRAY_LENGTH * (sizeof(int)));
    for (i = 0; i < ARRAY_LENGTH; i++)
        ARRAY[i] = rand() % 100;

    printf("Random Numbers Generated are :\n");
    for (i = 0; i < ARRAY_LENGTH; i++)
        printf("%d ", ARRAY[i]);

    printf("\nSorted Data: ");

    while (!isSorted) {
        isSorted = 1;
        for (i = 0; i < ARRAY_LENGTH - 1; i++) {
            if (ARRAY[i] > ARRAY[i + 1]) {
                isSorted = 0;
                TEMPORARY_ELEMENT = ARRAY[i];
                ARRAY[i] = ARRAY[i + 1];
                ARRAY[i + 1] = TEMPORARY_ELEMENT;
            }
        }
    }
    for (i = 0; i < ARRAY_LENGTH; i++) {
        printf("%d ", ARRAY[i]);
        MEAN = MEAN + ARRAY[i];
    }
    MEAN = MEAN / (float) ARRAY_LENGTH;

    for (i = 0; i < ARRAY_LENGTH; i++)
        VARIANCE = VARIANCE + (pow((ARRAY[i] - MEAN), 2));

    VARIANCE = VARIANCE / (float) ARRAY_LENGTH;
    STAND = sqrt(VARIANCE);

    printf("\n\n- Mean is: %f\n", MEAN);
    printf("- Variance is: %f\n", VARIANCE);
    printf("- Standard Deviation is: %f\n", STAND);

}