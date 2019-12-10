#include <stdio.h>
#include <stdlib.h>

void display(int arr[], int n) {

    int i;
    for (i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }

    printf("\n");

}

void bead_sort(int * a, int len) {
    int i, j, max, sum;
    unsigned char * beads;
#define BEAD(i, j) beads[i * max + j]

    for (i = 1, max = a[0]; i < len; i++)
        if (a[i] > max) max = a[i];

    beads = calloc(1, max * len);

    for (i = 0; i < len; i++)
        for (j = 0; j < a[i]; j++)
            BEAD(i, j) = 1;

    for (j = 0; j < max; j++) {
        for (sum = i = 0; i < len; i++) {
            sum += BEAD(i, j);
            BEAD(i, j) = 0;
        }
        for (i = len - sum; i < len; i++) BEAD(i, j) = 1;
    }

    for (i = 0; i < len; i++) {
        for (j = 0; j < max && BEAD(i, j); j++);
        a[i] = j;
    }
    free(beads);
}

int main(int argc, const char * argv[]) {
    int n;
    printf("Enter size of array:\n");
    scanf("%d", & n);

    printf("Enter the elements of the array\n");
    int i;
    int arr[n];
    for (i = 0; i < n; i++) {
        scanf("%d", & arr[i]);
    }

    printf("Original array: ");
    display(arr, n);

    bead_sort(arr, n);

    printf("Sorted array: ");
    display(arr, n);

    return 0;
}