#include <stdio.h>

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

void bubbleSort(int arr[], int size) {

    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - 1 - i; j++) {
            if (arr[j] > arr[j + 1]) {
                swap( & arr[j], & arr[j + 1]);
            }
        }
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

    bubbleSort(arr, n);

    printf("Sorted array: ");
    display(arr, n);

    return 0;
}