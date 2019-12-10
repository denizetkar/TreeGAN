#include<stdio.h>

int interpolationSearch(int arr[], int n, int key) {
    int low = 0, high = n - 1;
    while (low <= high && key >= arr[low] && key <= arr[high]) {
        int pos = low + ((key - arr[low]) * (high - low)) / (arr[high] - arr[low]);
        if (key > arr[pos])
            low = pos + 1;
        else if (key < arr[pos])
            high = pos - 1;
        else
            return pos;
    }

    return -1;
}

int main() {
    int x;
    int arr[] = {
        10,
        12,
        13,
        16,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        33,
        35,
        42,
        47
    };
    int n = sizeof(arr) / sizeof(arr[0]);

    printf("Array: ");
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\nEnter the number to be searched: ");
    scanf("%d", & x);

    int index = interpolationSearch(arr, n, x);

    if (index != -1)
        printf("Element found at position: %d\n", index);
    else
        printf("Element not found.\n");
    return 0;
}