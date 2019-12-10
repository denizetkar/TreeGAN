#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int getBig(int a[], int i, int right, int pivot) {
    for (int k = i; k <= right; k++) {
        if (a[k] > pivot)
            return k;
    }
    return right + 1;
}

int getSmall(int a[], int j, int left, int pivot) {
    for (int k = j; k >= left; k--) {
        if (a[k] < pivot)
            return k;
    }
    return -1;
}

void swap(int * a, int * b) {
    int t = * a;
    * a = * b;
    * b = t;
}

void random_quick(int a[], int left, int right) {
    if (left >= right)
        return;
    int index = left + (rand() % (right - left)), i = left, j = right;
    int pivot_index = index;
    int pivot = a[index];
    i = getBig(a, i, right, pivot);
    j = getSmall(a, j, left, pivot);
    while (i <= j) {
        swap( & a[i], & a[j]);
        i = getBig(a, i, right, pivot);
        j = getSmall(a, j, left, pivot);
    }
    if (pivot_index > j && pivot_index > i) {
        swap( & a[i], & a[pivot_index]);
        random_quick(a, left, i - 1);
        random_quick(a, i + 1, right);
    } else if (pivot_index < j && pivot_index < i) {
        swap( & a[j], & a[pivot_index]);
        random_quick(a, left, j - 1);
        random_quick(a, j + 1, right);
    } else {
        random_quick(a, left, pivot_index - 1);
        random_quick(a, pivot_index + 1, right);
    }
}

int main() {
    srand(time(0));
    int num;
    scanf("%d", & num);
    int arr[num];
    for (int i = 0; i < num; i++) {
        scanf("%d", & arr[i]);
    }
    random_quick(arr, 0, num - 1);
    for (int i = 0; i < num; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}