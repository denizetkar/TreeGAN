#include <stdio.h>
#include <stdlib.h>
#include <math.h>

struct node {
    int val;
    struct node * par;
    struct node * left;
    struct node * right;
    int color;
};

struct node * newNode(int val, struct node * par) {
    struct node * create = (struct node * )(malloc(sizeof(struct node)));
    create -> val = val;
    create -> par = par;
    create -> left = NULL;
    create -> right = NULL;
    create -> color = 1;
}

int isLeaf(struct node * n) {
    if (n -> left == NULL && n -> right == NULL) {
        return 1;
    }
    return 0;
}

struct node * leftRotate(struct node * node) {
    struct node * parent = node -> par;
    struct node * grandParent = parent -> par;

    parent -> right = node -> left;
    if (node -> left != NULL) {
        node -> left -> par = parent;
    }
    node -> par = grandParent;
    parent -> par = node;
    node -> left = parent;
    if (grandParent != NULL) {
        if (grandParent -> right == parent) {
            grandParent -> right = node;
        } else {
            grandParent -> left = node;
        }
    }
    return node;
}

struct node * rightRotate(struct node * node) {
    struct node * parent = node -> par;
    struct node * grandParent = parent -> par;

    parent -> left = node -> right;
    if (node -> right != NULL) {
        node -> right -> par = parent;
    }
    node -> par = grandParent;
    parent -> par = node;
    node -> right = parent;
    if (grandParent != NULL) {
        if (grandParent -> right == parent) {
            grandParent -> right = node;
        } else {
            grandParent -> left = node;
        }
    }
    return node;
}

void checkNode(struct node * node) {

    if (node == NULL || node -> par == NULL) {
        return;
    }
    struct node * child = node;
    if (node -> color == 0 || (node -> par) -> color == 0) {
        return;
    }

    struct node * parent = node -> par;
    struct node * grandParent = parent -> par;

    if (grandParent == NULL) {
        parent -> color = 0;
        return;
    }

    if (grandParent -> right != NULL && (grandParent -> right) -> color == 1 && grandParent -> left != NULL && (grandParent -> left) -> color == 1) {
        (grandParent -> right) -> color = 0;
        (grandParent -> left) -> color = 0;
        grandParent -> color = 1;
        return;
    } else {
        struct node * greatGrandParent = grandParent -> par;
        if (grandParent -> right == parent) {
            if (parent -> right == node) {
                grandParent -> right = parent -> left;
                if (parent -> left != NULL) {
                    (parent -> left) -> par = grandParent;
                }
                parent -> left = grandParent;
                grandParent -> par = parent;

                parent -> par = greatGrandParent;
                if (greatGrandParent != NULL) {
                    if (greatGrandParent -> left != NULL && greatGrandParent -> left == grandParent) {
                        greatGrandParent -> left = parent;
                    } else {
                        greatGrandParent -> right = parent;
                    }
                }

                parent -> color = 0;
                grandParent -> color = 1;
            } else {
                parent -> left = child -> right;
                if (child -> right != NULL) {
                    (child -> right) -> par = parent;
                }
                child -> right = parent;
                parent -> par = child;

                grandParent -> right = child -> left;
                if (child -> left != NULL) {
                    (child -> left) -> par = grandParent;
                }
                child -> left = grandParent;
                grandParent -> par = child;

                child -> par = greatGrandParent;
                if (greatGrandParent != NULL) {
                    if (greatGrandParent -> left != NULL && greatGrandParent -> left == grandParent) {
                        greatGrandParent -> left = child;
                    } else {
                        greatGrandParent -> right = child;
                    }
                }

                child -> color = 0;
                grandParent -> color = 1;
            }
        } else {
            if (parent -> left == node) {
                grandParent -> left = parent -> right;
                if (parent -> right != NULL) {
                    (parent -> right) -> par = grandParent;
                }
                parent -> right = grandParent;
                grandParent -> par = parent;

                parent -> par = greatGrandParent;
                if (greatGrandParent != NULL) {
                    if (greatGrandParent -> left != NULL && greatGrandParent -> left == grandParent) {
                        greatGrandParent -> left = parent;
                    } else {
                        greatGrandParent -> right = parent;
                    }
                }

                parent -> color = 0;
                grandParent -> color = 1;
            } else {

                parent -> right = child -> left;
                if (child -> left != NULL) {
                    (child -> left) -> par = parent;
                }
                child -> left = parent;
                parent -> par = child;

                grandParent -> left = child -> right;
                if (child -> right != NULL) {
                    (child -> right) -> par = grandParent;
                }
                child -> right = grandParent;
                grandParent -> par = child;

                child -> par = greatGrandParent;
                if (greatGrandParent != NULL) {
                    if (greatGrandParent -> left != NULL && greatGrandParent -> left == grandParent) {
                        greatGrandParent -> left = child;
                    } else {
                        greatGrandParent -> right = child;
                    }
                }

                child -> color = 0;
                grandParent -> color = 1;
            }
        }
    }
}

void insertNode(int val, struct node ** root) {
    struct node * buffRoot = * root;
    while (buffRoot) {
        if (buffRoot -> val > val) {
            if (buffRoot -> left != NULL) {
                buffRoot = buffRoot -> left;
            } else {
                struct node * toInsert = newNode(val, buffRoot);
                buffRoot -> left = toInsert;
                buffRoot = toInsert;

                break;
            }
        } else {

            if (buffRoot -> right != NULL) {
                buffRoot = buffRoot -> right;
            } else {
                struct node * toInsert = newNode(val, buffRoot);
                buffRoot -> right = toInsert;
                buffRoot = toInsert;

                break;
            }
        }
    }

    while (buffRoot != * root) {
        checkNode(buffRoot);
        if (buffRoot -> par == NULL) {
            * root = buffRoot;
            break;
        }
        buffRoot = buffRoot -> par;
        if (buffRoot == * root) {
            buffRoot -> color = 0;
        }
    }
}

void checkForCase2(struct node * toDelete, int delete, int fromDirection, struct node ** root) {

    if (toDelete == ( * root)) {
        ( * root) -> color = 0;
        return;
    }

    if (!delete && toDelete -> color == 1) {
        if (!fromDirection) {
            if (toDelete -> right != NULL) {
                toDelete -> right -> color = 1;
            }
        } else {
            if (toDelete -> left != NULL) {
                toDelete -> left -> color = 1;
            }
        }
        toDelete -> color = 0;
        return;
    }

    struct node * sibling;
    struct node * parent = toDelete -> par;
    int locateChild = 0;
    if (parent -> right == toDelete) {
        sibling = parent -> left;
        locateChild = 1;
    } else {
        sibling = parent -> right;
    }

    if ((sibling -> right != NULL && sibling -> right -> color == 1) || (sibling -> left != NULL && sibling -> left -> color == 1)) {
        if (sibling -> right != NULL && sibling -> right -> color == 1) {

            if (locateChild == 1) {

                int parColor = parent -> color;

                sibling = leftRotate(sibling -> right);

                parent = rightRotate(sibling);

                if (parent -> par == NULL) {
                    * root = parent;
                }

                parent -> color = parColor;
                parent -> left -> color = 0;
                parent -> right -> color = 0;

                if (delete) {
                    if (toDelete -> left != NULL) {
                        toDelete -> left -> par = parent -> right;
                    }
                    parent -> right -> right = toDelete -> left;
                    free(toDelete);
                }

            } else {

                int parColor = parent -> color;

                parent = leftRotate(sibling);

                if (parent -> par == NULL) {
                    * root = parent;
                }

                parent -> color = parColor;
                parent -> left -> color = 0;
                parent -> right -> color = 0;

                if (delete) {
                    if (toDelete -> right != NULL) {
                        toDelete -> right -> par = parent -> left;
                    }
                    parent -> left -> left = toDelete -> left;
                    free(toDelete);
                }

            }
        } else {

            if (locateChild == 0) {

                int parColor = parent -> color;

                sibling = rightRotate(sibling -> left);

                parent = leftRotate(sibling);

                if (parent -> par == NULL) {
                    * root = parent;
                }

                parent -> color = parColor;
                parent -> left -> color = 0;
                parent -> right -> color = 0;

                if (delete) {
                    if (toDelete -> right != NULL) {
                        toDelete -> right -> par = parent -> left;
                    }
                    parent -> left -> left = toDelete -> right;
                    free(toDelete);
                }

            } else {

                int parColor = parent -> color;

                parent = rightRotate(sibling);

                if (parent -> par == NULL) {
                    * root = parent;
                }

                parent -> color = parColor;
                parent -> left -> color = 0;
                parent -> right -> color = 0;

                if (delete) {
                    if (toDelete -> left != NULL) {
                        toDelete -> left -> par = parent -> right;
                    }
                    parent -> right -> right = toDelete -> left;
                    free(toDelete);
                }

            }
        }
    } else if (sibling -> color == 0) {

        sibling -> color = 1;

        if (delete) {
            if (locateChild) {
                toDelete -> par -> right = toDelete -> left;
                if (toDelete -> left != NULL) {
                    toDelete -> left -> par = toDelete -> par;
                }
            } else {
                toDelete -> par -> left = toDelete -> right;
                if (toDelete -> right != NULL) {
                    toDelete -> right -> par = toDelete -> par;
                }
            }
        }

        checkForCase2(parent, 0, locateChild, root);
    } else {
        if (locateChild) {

            toDelete -> par -> right = toDelete -> left;
            if (toDelete -> left != NULL) {
                toDelete -> left -> par = toDelete -> par;
            }

            parent = rightRotate(sibling);

            if (parent -> par == NULL) {
                * root = parent;
            }

            parent -> color = 0;
            parent -> right -> color = 1;
            checkForCase2(parent -> right, 0, 1, root);
        } else {

            toDelete -> par -> left = toDelete -> right;
            if (toDelete -> right != NULL) {
                toDelete -> right -> par = toDelete -> par;
            }
            parent = leftRotate(sibling);

            if (parent -> par == NULL) {
                * root = parent;
            }

            printf("\nroot - %d - %d\n", parent -> val, parent -> left -> val);

            parent -> color = 0;
            parent -> left -> color = 1;
            checkForCase2(parent -> left, 0, 0, root);
        }
    }

}

void deleteNode(int val, struct node ** root) {
    struct node * buffRoot = * root;

    while (1) {

        if (val == buffRoot -> val) {
            break;
        }

        if (val > buffRoot -> val) {
            if (buffRoot -> right != NULL) {
                buffRoot = buffRoot -> right;
            } else {
                printf("struct node Not Found!!!");
                return;
            }
        } else {
            if (buffRoot -> left != NULL) {
                buffRoot = buffRoot -> left;
            } else {
                printf("struct node Not Found!!!");
                return;
            }
        }
    }

    struct node * toDelete = buffRoot;

    if (toDelete -> left != NULL) {
        toDelete = toDelete -> left;
        while (toDelete -> right != NULL) {
            toDelete = toDelete -> right;
        }
    } else if (toDelete -> right != NULL) {
        toDelete = toDelete -> right;
        while (toDelete -> left != NULL) {
            toDelete = toDelete -> left;
        }
    }

    if (toDelete == * root) {
        * root = NULL;
        return;
    }

    buffRoot -> val = toDelete -> val;
    toDelete -> val = val;

    if (toDelete -> color == 1 || (toDelete -> left != NULL && toDelete -> left -> color == 1) || (toDelete -> right != NULL && toDelete -> right -> color == 1)) {

        if (toDelete -> left == NULL && toDelete -> right == NULL) {
            if (toDelete -> par -> left == toDelete) {
                toDelete -> par -> left = NULL;
            } else {
                toDelete -> par -> right = NULL;
            }
        } else {

            if (toDelete -> left != NULL) {
                toDelete -> par -> right = toDelete -> left;
                toDelete -> left -> par = toDelete -> par;
                toDelete -> left -> color = 1;
            } else {
                toDelete -> par -> left = toDelete -> right;
                toDelete -> right -> par = toDelete -> par;
                toDelete -> right -> color = 1;
            }
        }

        free(toDelete);
    } else {
        checkForCase2(toDelete, 1, ((toDelete -> par -> right == toDelete)), root);
    }

}

void printInorder(struct node * root) {
    if (root != NULL) {
        printInorder(root -> left);
        printf("%d c-%d ", root -> val, root -> color);
        printInorder(root -> right);
    }
}

void checkBlack(struct node * temp, int c) {
    if (temp == NULL) {
        printf("%d ", c);
        return;
    }
    if (temp -> color == 0) {
        c++;
    }
    checkBlack(temp -> left, c);
    checkBlack(temp -> right, c);
}

int main() {
    struct node * root = NULL;
    int scanValue, choice = 1;
    printf("1 - Input\n2 - Delete\n3 - Inorder Traversel\n0 - Quit\n\nPlease Enter the Choice - ");
    scanf("%d", & choice);
    while (choice) {
        switch (choice) {
        case 1:
            printf("\n\nPlease Enter A Value to insert - ");
            scanf("%d", & scanValue);
            if (root == NULL) {
                root = newNode(scanValue, NULL);
                root -> color = 0;
            } else {
                insertNode(scanValue, & root);
            }
            printf("\nSuccessfully Inserted %d in the tree\n\n", scanValue);
            break;
        case 2:
            printf("\n\nPlease Enter A Value to Delete - ");
            scanf("%d", & scanValue);
            deleteNode(scanValue, & root);
            printf("\nSuccessfully Inserted %d in the tree\n\n", scanValue);
            break;
        case 3:
            printf("\nInorder Traversel - ");
            printInorder(root);
            printf("\n\n");
            break;
        default:
            if (root != NULL) {
                printf("Root - %d\n", root -> val);
            }
        }
        printf("1 - Input\n2 - Delete\n3 - Inorder Traversel\n0 - Quit\n\nPlease Enter the Choice - ");
        scanf("%d", & choice);
    }
    return 0;
}