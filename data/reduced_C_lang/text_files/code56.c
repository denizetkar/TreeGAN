#include<stdio.h>
#include<math.h>

int main() {
    int i, n, test = 0, count = 0;
    scanf("%d", & n);

    for (i = 1;; i++) {
        test = n / pow(5, i);
        if (test != 0) {
            count = count + test;
        } else
            break;
    }
    printf("%d\n", count);
    return 0;
}