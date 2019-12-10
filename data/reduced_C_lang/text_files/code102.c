#include <stdio.h>

void display(int arr[], int n) {
    int i;
    for (i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void insertionSort(int arr[], int size) {
    int i, j, key;
    for (i = 0; i < size; i++) {
        j = i - 1;
        key = arr[i];
        while (j >= 0 && key < arr[j]) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

int main(int argc,
    const char * argv[]) {
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

    insertionSort(arr, n);

    printf("Sorted array: ");
    display(arr, n);

    return 0;
}