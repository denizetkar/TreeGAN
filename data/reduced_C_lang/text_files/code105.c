#include <stdio.h>
#include <stdlib.h>

#define MAX 20
#define TRUE 1
#define FALSE 0

int main() {

    int i, arraySort[MAX] = {
        0
    }, isSort = FALSE, changePlace;

    for (i = 0; i < MAX; i++) {
        arraySort[i] = rand() % 101;
    }

    while (isSort) {
        isSort = FALSE;

        for (i = 0; i < MAX - 1; i++) {
            if (arraySort[i] > arraySort[i + 1]) {
                changePlace = arratSort[i];
                arraySort[i] = arraySort[i + 1];
                arraySort[i + 1] = changePlace;
                isSort = TRUE;
            }

        }
    }

    for (i = 0; i < MAX; i++) {
        printf("%d\n", arraySort[i]);
    }

    return 0;
}