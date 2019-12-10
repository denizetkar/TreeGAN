#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_ERASED -1
#define SUCCESS 0
#define INVALID_POSITION 1
#define POSITION_INIT 2
#define POSITION_NOT_INIT 3
#define POSITION_EMPTY 4
#define ARRAY_FULL 5

struct CArray {
    int * array;
    int size;
};

struct CArray * getCArray(int size);
struct CArray * getCopyCArray(struct CArray * array);

int insertValueCArray(struct CArray * array, int position, int value);
int removeValueCArray(struct CArray * array, int position);
int pushValueCArray(struct CArray * array, int value);
int updateValueCArray(struct CArray * array, int position, int value);

int eraseCArray(struct CArray * array);

int switchValuesCArray(struct CArray * array, int position1, int position2);
int reverseCArray(struct CArray * array);

int bubbleSortCArray(struct CArray * array);
int selectionSortCArray(struct CArray * array);
int insertionSortCArray(struct CArray * array);
int blenderCArray(struct CArray * array);

int valueOcurranceCArray(struct CArray * array, int value);
struct CArray * valuePositionsCArray(struct CArray * array, int value);
int findMaxCArray(struct CArray * array);
int findMinCArray(struct CArray * array);

int displayCArray(struct CArray * array);

void swap(struct CArray * array, int position1, int position2);

struct CArray * getCArray(int size) {
    struct CArray * array = (struct CArray * ) malloc(sizeof(struct CArray));
    array -> array = (int * ) malloc(sizeof(int) * size);
    array -> size = size;
    int i;
    for (i = 0; i < size; i++) {
        array -> array[i] = 0;
    }
    return array;
}

int insertValueCArray(struct CArray * array, int position, int value) {
    if (position >= 0 && position < array -> size) {
        if (array -> array[position] == 0) {
            array -> array[position] = value;
            return SUCCESS;
        } else return POSITION_INIT;
    }
    return INVALID_POSITION;
}

int removeValueCArray(struct CArray * array, int position) {
    if (position >= 0 && position < array -> size) {
        if (array -> array[position] != 0) {
            array -> array[position] = 0;
        } else return POSITION_EMPTY;
    }
    return INVALID_POSITION;
}

int pushValueCArray(struct CArray * array, int value) {
    int i;
    int ok = 0;
    for (i = 0; i < array -> size; i++) {
        if (array -> array[i] == 0) {
            array -> array[i] = value;
            ok = 1;
            break;
        }
    }
    if (ok == 1) return SUCCESS;
    else return ARRAY_FULL;
}

int updateValueCArray(struct CArray * array, int position, int value) {
    if (position >= 0 && position < array -> size) {
        if (array -> array[position] != 0) {

        } else return POSITION_NOT_INIT;
    }
    return INVALID_POSITION;
}

int eraseCArray(struct CArray * array) {
    int i;
    for (i = 0; i < array -> size; i++) {
        array -> array[i] = 0;
    }
    return 0;
}

int switchValuesCArray(struct CArray * array, int position1, int position2) {
    if (position1 >= 0 && position1 < array -> size &&
        position2 >= 0 && position2 < array -> size) {
        int temp = array -> array[position1];
        array -> array[position1] = array -> array[position2];
        array -> array[position2] = temp;
    }
    return INVALID_POSITION;
}

int reverseCArray(struct CArray * array) {
    int i;
    for (i = 0; i < array -> size / 2; i++) {
        swap(array, i, array -> size - i - 1);
    }
    return SUCCESS;
}

int displayCArray(struct CArray * array) {
    int i;
    printf("\nC ARRAY\n");
    for (i = 0; i < array -> size; i++) {
        printf("%d ", array -> array[i]);
    }
    printf("\n");
    return 0;
}

int blenderCArray(struct CArray * array) {
    srand(time(NULL) * array -> size);
    int i;
    int total = array -> size * 100;
    for (i = 0; i < total; i++) {
        swap(array, rand() % array -> size, rand() % array -> size);
    }
    return 0;
}

struct CArray * getCopyCArray(struct CArray * arr) {
    struct CArray * array = (struct CArray * ) malloc(sizeof(struct CArray));
    array -> array = (int * ) malloc(sizeof(int) * arr -> size);
    array -> size = arr -> size;
    int i;
    for (i = 0; i < arr -> size; i++) {
        array -> array[i] = arr -> array[i];
    }
    return array;
}

void swap(struct CArray * array, int position1, int position2) {
    int temp = array -> array[position1];
    array -> array[position1] = array -> array[position2];
    array -> array[position2] = temp;
}

int bubbleSortCArray(struct CArray * array) {
    int i, j;
    for (i = 0; i < array -> size - 1; i++) {
        for (j = 0; j < array -> size - i - 1; j++) {
            if (array -> array[j] > array -> array[j + 1]) {
                swap(array, j, j + 1);
            }
        }
    }
    return 0;
}

int selectionSortCArray(struct CArray * array) {
    int i, j, min;
    for (i = 0; i < array -> size - 1; i++) {
        min = i;
        for (j = i + 1; j < array -> size; j++)
            if (array -> array[j] < array -> array[min]) min = j;
        swap(array, min, i);
    }
    return 0;
}

int insertionSortCArray(struct CArray * array) {
    int i, j, num;
    for (i = 1; i < array -> size; i++) {
        num = array -> array[i];
        j = i - 1;
        while (j >= 0 && array -> array[j] > num) {
            array -> array[j + 1] = array -> array[j];
            j--;
        }
        array -> array[j + 1] = num;
    }
    return 0;
}

int valueOcurranceCArray(struct CArray * array, int value) {
    int i, total = 0;
    for (i = 0; i < array -> size; i++) {
        if (array -> array[i] == value) total++;
    }
    return total;
}

struct CArray * valuePositionsCArray(struct CArray * array, int value) {
    int i, j = 0;
    int total = valueOcurranceCArray(array, value);
    struct CArray * resultArray = getCArray(total);
    for (i = 0; i < array -> size; i++) {
        if (array -> array[i] == value) {
            resultArray -> array[j] = i;
            j++;
        }
    }
    return resultArray;
}

int findMinCArray(struct CArray * array) {
    int i;
    int min = array -> array[0];
    for (i = 1; i < array -> size; i++) {
        if (array -> array[i] < min) {
            min = array -> array[i];
        }
    }
    return min;
}

int findMaxCArray(struct CArray * array) {
    int i;
    int max = array -> array[0];
    for (i = 1; i < array -> size; i++) {
        if (array -> array[i] > max) {
            max = array -> array[i];
        }
    }
    return max;
}

int main() {
    printf("\n");
    printf(" +-------------------------------------+\n");
    printf(" |                                     |\n");
    printf(" |               C Array               |\n");
    printf(" |                                     |\n");
    printf(" +-------------------------------------+\n");
    printf("\n");

    struct CArray * array = getCArray(10);

    int i;
    for (i = 0; i < array -> size; i++) {
        insertValueCArray(array, i, i + 1);
    }
    printf("Entered array is:\n");
    displayCArray(array);
    printf("\nCode: %d\n", pushValueCArray(array, 11));

    for (i = 0; i < array -> size; i++) {
        removeValueCArray(array, i);
    }

    displayCArray(array);

    printf("\nCode: %d", removeValueCArray(array, -1));
    printf("\nCode: %d\n", insertValueCArray(array, -1, 1));

    for (i = 0; i < array -> size; i++) {
        insertValueCArray(array, i, i + 1);
    }
    eraseCArray(array);
    displayCArray(array);

    struct CArray * arr = getCArray(13);
    for (i = 0; i < arr -> size; i++) {
        insertValueCArray(arr, i, i + 1);
    }
    displayCArray(arr);
    for (i = 0; i < arr -> size / 2; i++) {
        switchValuesCArray(arr, i, arr -> size - i - 1);
    }

    displayCArray(arr);

    reverseCArray(arr);

    displayCArray(arr);

    srand(time(NULL));
    struct CArray * barray = getCArray(20);
    for (i = 0; i < barray -> size; i++) {
        insertValueCArray(barray, i, rand());
    }
    struct CArray * carray = getCopyCArray(barray);
    struct CArray * darray = getCopyCArray(barray);
    printf("\nNot sorted Array:");
    displayCArray(barray);

    printf("\nBubble Sort:");
    unsigned long begin1 = clock();
    bubbleSortCArray(barray);
    unsigned long end1 = clock();
    double time_spent1 = (double)(end1 - begin1) / CLOCKS_PER_SEC;
    displayCArray(barray);

    printf("\nSelection Sort:");
    unsigned long begin2 = clock();
    selectionSortCArray(carray);
    unsigned long end2 = clock();
    double time_spent2 = (double)(end2 - begin2) / CLOCKS_PER_SEC;
    displayCArray(carray);

    printf("\nInsertion Sort:");
    unsigned long begin3 = clock();
    insertionSortCArray(darray);
    unsigned long end3 = clock();
    double time_spent3 = (double)(end3 - begin3) / CLOCKS_PER_SEC;
    displayCArray(carray);

    reverseCArray(barray);

    printf("\nTotal time spent for bubble sort: %lf seconds", time_spent1);
    printf("\nTotal time spent for selection sort: %lf seconds", time_spent2);
    printf("\nTotal time spent for insertion sort: %lf seconds", time_spent3);

    struct CArray * aarray = getCArray(1000);
    for (i = 0; i < aarray -> size; i++) {
        insertValueCArray(aarray, i, rand() % 100);
    }

    int j = 24;
    printf("\nOccurrences of the number %d in the array: %d", j,
        valueOcurranceCArray(aarray, j));
    printf("\nAnd its positions:\n");
    struct CArray * positions = valuePositionsCArray(aarray, j);
    displayCArray(positions);
    printf("\nAll %d s", j);
    for (i = 0; i < positions -> size; i++) {
        printf("\nPosition %d has a value of %d",
            positions -> array[i], aarray -> array[positions -> array[i]]);
    }
    printf("\nThe list has a minimum value of %d and a maximum value of %d",
        findMinCArray(aarray), findMaxCArray(aarray));
    insertionSortCArray(aarray);

    free(arr);
    free(array);
    free(aarray);
    free(barray);
    free(carray);
    free(darray);
    printf("\n");
    return 0;
}