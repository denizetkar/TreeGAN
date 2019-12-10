#include <stdio.h>
#include <stdlib.h>

void display(int arr[], int n) {

    int i;
    for (i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }

    printf("\n");
}

void swap(int * first, int * second) {

    int temp = * first;
    * first = * second;
    * second = temp;
}

int partition(int arr[], int lower, int upper) {

    int i = (lower - 1);

    int pivot = arr[upper];

    int j;
    for (j = lower; j < upper; j++) {

        if (arr[j] <= pivot) {

            i++;
            swap( & arr[i], & arr[j]);
        }
    }

    swap( & arr[i + 1], & arr[upper]);

    return (i + 1);
}

void quickSort(int arr[], int lower, int upper) {

    if (upper > lower) {

        int partitionIndex = partition(arr, lower, upper);

        quickSort(arr, lower, partitionIndex - 1);
        quickSort(arr, partitionIndex + 1, upper);
    }
}

int main() {

    int n;
    printf("Enter size of array:\n");
    scanf("%d", & n);

    printf("Enter the elements of the array\n");
    int i;
    int * arr = (int * ) malloc(sizeof(int) * n);
    for (i = 0; i < n; i++) {
        scanf("%d", & arr[i]);
    }

    printf("Original array: ");
    display(arr, n);

    quickSort(arr, 0, n - 1);

    printf("Sorted array: ");
    display(arr, n);
    getchar();
    return 0;
}