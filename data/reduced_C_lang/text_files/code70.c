#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define LEN 10
#define STEP 5

struct data {
    int * range;
    int length;
};

struct data * int_fact(int);

void print_arr(struct data * );

int * increase(int * , int);

void destroy(struct data * );

int main() {

    int n = 0;

    printf("\t\tPrim factoriziation\n\n");
    printf("positive integer (> 1) ? ");
    scanf("%d", & n);
    struct data * r = int_fact(n);
    printf("\nThe factoriziation are: ");
    print_arr(r);
    destroy(r);
    return 0;
}

struct data * int_fact(int n) {
    assert(n > 1);

    int len = LEN;
    int count = 0;
    int i = 0;
    int * range = (int * ) malloc(sizeof(int) * len);
    assert(range);
    struct data * pstr = (struct data * ) malloc(sizeof(struct data));
    assert(pstr);

    while (n % 2 == 0) {
        n /= 2;
        if (i < len) {
            range[i] = 2;
            i++;
        } else {
            range = increase(range, len);
            len += STEP;
            range[i] = 2;
            i++;
        }
        count++;

    }

    int j = 3;
    while (j * j <= n) {
        while (n % j == 0) {
            n /= j;
            if (i < len) {
                range[i] = j;
                i++;
            } else {
                range = increase(range, len);
                len += STEP;
                range[i] = j;
                i++;
            }
            count++;
        }

        j += 2;
    }

    if (n > 1) {
        if (i < len) {
            range[i] = n;
            i++;
        } else {
            range = increase(range, len);
            len += STEP;
            range[i] = n;
            i++;
        }
        count++;
    }

    pstr -> range = range;
    pstr -> length = count;
    return pstr;

}

void print_arr(struct data * pStr) {
    assert(pStr);
    int i = 0;
    printf("\n");
    for (i; i < pStr -> length; i++) {
        if (i == 0)
            printf("%d", pStr -> range[0]);
        else
            printf("-%d", pStr -> range[i]);
    }
    printf("\n");
}

int * increase(int * arr, int len) {
    assert(arr);
    int * tmp = (int * ) realloc(arr, sizeof(int) * (len + STEP));
    assert(tmp);
    return tmp;
}

void destroy(struct data * r) {
    free(r -> range);
    free(r);
}