#include<stdio.h>

int main() {
    int n, curr_no;
    scanf("%d", & n);
    curr_no = n;
    while (curr_no != 1) {
        if (curr_no % 2 == 0) {
            curr_no = curr_no / 2;
            printf("%d->", curr_no);
        } else {
            curr_no = (curr_no * 3) + 1;
            printf("%d->", curr_no);
        }
    }
    printf("1");
    return 0;
}