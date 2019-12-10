#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

extern struct Stack_T * Stack_init(void);
extern int Stack_size(struct Stack_T * stack);
extern int Stack_empty(struct Stack_T * stack);
extern void Stack_push(struct Stack_T * stack, void * val);
extern void * Stack_pop(struct Stack_T * stack);
extern void Stack_print(struct Stack_T * stack);

struct elem {
    void * val;
    struct elem * next;
};

struct Stack_T {
    int count;
    struct elem * head;
};

struct Stack_T * Stack_init(void) {
    struct Stack_T * stack;
    stack = (struct Stack_T * ) malloc(sizeof(struct Stack_T));
    stack -> count = 0;
    stack -> head = NULL;
    return stack;
}

int Stack_empty(struct Stack_T * stack) {
    assert(stack);
    return stack -> count == 0;
}

int Stack_size(struct Stack_T * stack) {
    assert(stack);
    return stack -> count;
}

void Stack_push(struct Stack_T * stack, void * val) {
    struct elem * t;

    assert(stack);
    t = (struct elem * ) malloc(sizeof(struct elem));
    t -> val = val;
    t -> next = stack -> head;
    stack -> head = t;
    stack -> count++;
}

void * Stack_pop(struct Stack_T * stack) {
    void * val;
    struct elem * t;

    assert(stack);
    assert(stack -> count > 0);
    t = stack -> head;
    stack -> head = t -> next;
    stack -> count--;
    val = t -> val;
    free(t);
    return val;
}

void Stack_print(struct Stack_T * stack) {
    assert(stack);

    int i, size = Stack_size(stack);
    struct elem * current_elem = stack -> head;
    printf("Stack [Top --- Bottom]: ");
    for (i = 0; i < size; ++i) {
        printf("%p ", (int * ) current_elem -> val);
        current_elem = current_elem -> next;
    }
    printf("\n");
}

int main() {
    struct Stack_T * stk;
    stk = Stack_init();
    Stack_push(stk, (int * ) 1);
    Stack_push(stk, (int * ) 2);
    Stack_push(stk, (int * ) 3);
    Stack_push(stk, (int * ) 4);
    printf("Size: %d\n", Stack_size(stk));
    Stack_print(stk);
    Stack_pop(stk);
    printf("Stack after popping: \n");
    Stack_print(stk);
    Stack_pop(stk);
    printf("Stack after popping: \n");
    Stack_print(stk);
    return 0;
}