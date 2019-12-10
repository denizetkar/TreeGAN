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

void cycleSort(int arr[], int n) {
    int writes = 0;

    for (int cycle_start = 0; cycle_start <= n - 2; cycle_start++) {
        int item = arr[cycle_start];

        int pos = cycle_start;
        for (int i = cycle_start + 1; i < n; i++)
            if (arr[i] < item)
                pos++;

        if (pos == cycle_start)
            continue;

        while (item == arr[pos])
            pos += 1;

        if (pos != cycle_start) {
            swap( & item, & arr[pos]);
            writes++;
        }

        while (pos != cycle_start) {
            pos = cycle_start;

            for (int i = cycle_start + 1; i < n; i++)
                if (arr[i] < item)
                    pos += 1;

            while (item == arr[pos])
                pos += 1;

            if (item != arr[pos]) {
                swap( & item, & arr[pos]);
                writes++;
            }
        }
    }

}

int main() {
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

    cycleSort(arr, n);
    printf("Sorted array: ");
    display(arr, n);

    return 0;
}