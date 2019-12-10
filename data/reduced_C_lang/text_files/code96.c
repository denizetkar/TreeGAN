#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define NARRAY 8
#define NBUCKET 5
#define INTERVAL 10

struct Node {
    int data;
    struct Node * next;
};

void BucketSort(int arr[]);
struct Node * InsertionSort(struct Node * list);
void print(int arr[]);
void printBuckets(struct Node * list);
int getBucketIndex(int value);

void BucketSort(int arr[]) {
    int i, j;
    struct Node ** buckets;

    buckets = (struct Node ** ) malloc(sizeof(struct Node * ) * NBUCKET);

    for (i = 0; i < NBUCKET; ++i) {
        buckets[i] = NULL;
    }

    for (i = 0; i < NARRAY; ++i) {
        struct Node * current;
        int pos = getBucketIndex(arr[i]);
        current = (struct Node * ) malloc(sizeof(struct Node));
        current -> data = arr[i];
        current -> next = buckets[pos];
        buckets[pos] = current;
    }

    for (i = 0; i < NBUCKET; i++) {
        printf("Bucket[\"%d\"] : ", i);
        printBuckets(buckets[i]);
        printf("\n");
    }

    for (i = 0; i < NBUCKET; ++i) {
        buckets[i] = InsertionSort(buckets[i]);
    }

    printf("--------------\n");
    printf("Buckets after sorted\n");
    for (i = 0; i < NBUCKET; i++) {
        printf("Bucket[\"%d\"] : ", i);
        printBuckets(buckets[i]);
        printf("\n");
    }

    for (j = 0, i = 0; i < NBUCKET; ++i) {
        struct Node * node;
        node = buckets[i];
        while (node) {
            assert(j < NARRAY);
            arr[j++] = node -> data;
            node = node -> next;
        }
    }

    for (i = 0; i < NBUCKET; ++i) {
        struct Node * node;
        node = buckets[i];
        while (node) {
            struct Node * tmp;
            tmp = node;
            node = node -> next;
            free(tmp);
        }
    }
    free(buckets);
    return;
}

struct Node * InsertionSort(struct Node * list) {
    struct Node * k, * nodeList;
    if (list == NULL || list -> next == NULL) {
        return list;
    }

    nodeList = list;
    k = list -> next;
    nodeList -> next = NULL;
    while (k != NULL) {
        struct Node * ptr;
        if (nodeList -> data > k -> data) {
            struct Node * tmp;
            tmp = k;
            k = k -> next;
            tmp -> next = nodeList;
            nodeList = tmp;
            continue;
        }

        for (ptr = nodeList; ptr -> next != NULL; ptr = ptr -> next) {
            if (ptr -> next -> data > k -> data) break;
        }

        if (ptr -> next != NULL) {
            struct Node * tmp;
            tmp = k;
            k = k -> next;
            tmp -> next = ptr -> next;
            ptr -> next = tmp;
            continue;
        } else {
            ptr -> next = k;
            k = k -> next;
            ptr -> next -> next = NULL;
            continue;
        }
    }
    return nodeList;
}

int getBucketIndex(int value) {
    return value / INTERVAL;
}

void print(int ar[]) {
    int i;
    for (i = 0; i < NARRAY; ++i) {
        printf("%d ", ar[i]);
    }
    printf("\n");
}

void printBuckets(struct Node * list) {
    struct Node * cur = list;
    while (cur) {
        printf("%d ", cur -> data);
        cur = cur -> next;
    }
}

int main(void) {
    int array[NARRAY] = {
        29,
        25,
        -1,
        49,
        9,
        37,
        21,
        43
    };

    printf("Initial array\n");
    print(array);
    printf("------------\n");

    BucketSort(array);
    printf("------------\n");
    printf("Sorted array\n");
    print(array);
    return 0;
}