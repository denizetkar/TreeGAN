#include <stdio.h>

int p[1000000];
int find(int x) {
    if (p[x] == x) {
        return x;
    } else {
        p[x] = find(p[x]);
        return p[x];
    }
}
void join(int x, int y) {
    p[find(x)] = find(y);
}

int main() {
    for (int i = 0; i < 10; i++) {
        p[i] = i;
    }
    join(3, 5);
    join(3, 8);
    join(0, 5);
    if (find(0) == find(3)) {
        printf("0 and 3 are groupped together\n");
    }
    printf("The array is now: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", p[i]);
    }
    printf("\n");

    return 0;
}