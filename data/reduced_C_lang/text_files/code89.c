#include<stdio.h>

int n, m;

void binarySearch(int mat[n][m], int i, int j_low, int j_high, int x) {
    while (j_low <= j_high) {
        int j_mid = (j_low + j_high) / 2;

        if (mat[i][j_mid] == x) {
            printf("Found at (%d,%d)\n", i, j_mid);
            return;
        } else if (mat[i][j_mid] > x)
            j_high = j_mid - 1;
        else
            j_low = j_mid + 1;
    }
    printf("element not found\n");
}

void modifiedBinarySearch(int mat[n][m], int n, int m, int x) {
    if (n == 1) {
        binarySearch(mat, 0, 0, m - 1, x);
        return;
    }

    int i_low = 0, i_high = n - 1, j_mid = m / 2;
    while ((i_low + 1) < i_high) {
        int i_mid = (i_low + i_high) / 2;
        if (mat[i_mid][j_mid] == x) {
            printf("Found at (%d,%d)\n", i_mid, j_mid);
            return;
        } else if (mat[i_mid][j_mid] > x)
            i_high = i_mid;
        else
            i_low = i_mid;
    }
    if (mat[i_low][j_mid] == x)
        printf("Found at (%d,%d)\n", i_low, j_mid);
    else if (mat[i_low + 1][j_mid] == x)
        printf("Found at (%d,%d)\n", i_low + 1, j_mid);

    else if (x <= mat[i_low][j_mid - 1])
        binarySearch(mat, i_low, 0, j_mid - 1, x);

    else if (x >= mat[i_low][j_mid + 1] && x <= mat[i_low][m - 1])
        binarySearch(mat, i_low, j_mid + 1, m - 1, x);

    else if (x <= mat[i_low + 1][j_mid - 1])
        binarySearch(mat, i_low + 1, 0, j_mid - 1, x);

    else
        binarySearch(mat, i_low + 1, j_mid + 1, m - 1, x);
}

int main() {
    int x;
    scanf("%d %d %d\n", & n, & m, & x);
    int mat[n][m];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            scanf("%d", & mat[i][j]);
        }
    }
    modifiedBinarySearch(mat, n, m, x);
    return 0;
}