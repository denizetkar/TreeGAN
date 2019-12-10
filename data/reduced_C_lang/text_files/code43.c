#include <string.h>
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

struct List_T {
    void * val;
    struct List_T * next;
};

extern struct List_T * List_init(void);
extern struct List_T * List_push(struct List_T * list, void * val);
extern int List_length(struct List_T * list);
extern void ** List_toArray(struct List_T * list);
extern struct List_T * List_append(struct List_T * list, struct List_T * tail);

extern struct List_T * List_copy(struct List_T * list);
extern int List_pop(struct List_T ** list);

struct List_T * List_init(void) {
    struct List_T * list;
    list = (struct List_T * ) malloc(sizeof(struct List_T));
    list -> next = NULL;
    return list;
}

struct List_T * List_push(struct List_T * list, void * val) {
    struct List_T * new_elem = (struct List_T * ) malloc(sizeof(struct List_T));
    new_elem -> val = val;
    new_elem -> next = list;
    return new_elem;
}

int List_length(struct List_T * list) {
    int n;
    for (n = 0; list; list = list -> next)
        n++;
    return n;
}

void ** List_toArray(struct List_T * list) {
    int i, n = List_length(list);
    void ** array = (void ** ) malloc((n + 1) * sizeof( * array));

    for (i = 0; i < n; i++) {
        array[i] = list -> val;
        list = list -> next;
    }
    array[i] = NULL;
    return array;
}

struct List_T * List_append(struct List_T * list, struct List_T * tail) {
    struct List_T * p = list;
    while (p -> next != NULL) {
        p = p -> next;
    }

    p -> next = tail;
    return list;
}

void print_list(char ** array) {
    int i;
    for (i = 0; array[i]; i++)
        printf("%s", array[i]);
    printf("\n");
}

int main() {
    struct List_T * list1, * list2, * list3;
    char ** str1;

    list1 = List_push(NULL, "Dang ");
    list1 = List_push(list1, "Hoang ");
    list1 = List_push(list1, "Hai ");
    printf("List 1: ");
    str1 = (char ** ) List_toArray(list1);
    print_list(str1);

    list2 = List_push(NULL, "Siemens");
    list2 = List_push(list2, "Graphics ");
    list2 = List_push(list2, "Mentor ");
    printf("List 2: ");
    print_list((char ** ) List_toArray(list2));

    list3 = List_append(list1, list2);
    printf("Test append list2 into list1: ");
    print_list((char ** ) List_toArray(list3));

    return 0;
}