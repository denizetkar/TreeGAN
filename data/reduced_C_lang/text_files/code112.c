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

void selectionSort(int arr[], int size) {

    for (int i = 0; i < size; i++) {
        int min_index = i;
        for (int j = i + 1; j < size; j++) {
            if (arr[min_index] > arr[j]) {
                min_index = j;
            }
        }
        swap( & arr[i], & arr[min_index]);
    }
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

    selectionSort(arr, n);

    printf("Sorted array: ");
    display(arr, n);

    return 0;
}