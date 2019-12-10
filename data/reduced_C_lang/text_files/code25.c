#include <stdio.h>
#include <stdlib.h>

struct AVLnode {
    int key;
    struct AVLnode * left;
    struct AVLnode * right;
    int height;
};

struct AVLnode * newNode(int key) {
    struct AVLnode * node = (struct AVLnode * ) malloc(sizeof(struct AVLnode));

    if (node == NULL)
        printf("!! Out of Space !!\n");
    else {
        node -> key = key;
        node -> left = NULL;
        node -> right = NULL;
        node -> height = 0;
    }

    return node;
}

int nodeHeight(struct AVLnode * node) {
    if (node == NULL)
        return -1;
    else
        return (node -> height);
}

int heightDiff(struct AVLnode * node) {
    if (node == NULL)
        return 0;
    else
        return (nodeHeight(node -> left) - nodeHeight(node -> right));
}

struct AVLnode * minNode(struct AVLnode * node) {
    struct AVLnode * temp = node;

    while (temp -> left != NULL)
        temp = temp -> left;

    return temp;
}

void printAVL(struct AVLnode * node, int level) {
    int i;
    if (node != NULL) {
        printAVL(node -> right, level + 1);
        printf("\n\n");

        for (i = 0; i < level; i++)
            printf("\t");

        printf("%d", node -> key);

        printAVL(node -> left, level + 1);
    }
}

struct AVLnode * rightRotate(struct AVLnode * z) {
    struct AVLnode * y = z -> left;
    struct AVLnode * T3 = y -> right;

    y -> right = z;
    z -> left = T3;

    z -> height = (max(nodeHeight(z -> left), nodeHeight(z -> right)) + 1);
    y -> height = (max(nodeHeight(y -> left), nodeHeight(y -> right)) + 1);

    return y;
}

struct AVLnode * leftRotate(struct AVLnode * z) {
    struct AVLnode * y = z -> right;
    struct AVLnode * T3 = y -> left;

    y -> left = z;
    z -> right = T3;

    z -> height = (max(nodeHeight(z -> left), nodeHeight(z -> right)) + 1);
    y -> height = (max(nodeHeight(y -> left), nodeHeight(y -> right)) + 1);

    return y;
}

struct AVLnode * LeftRightRotate(struct AVLnode * z) {
    z -> left = leftRotate(z -> left);

    return (rightRotate(z));
}

struct AVLnode * RightLeftRotate(struct AVLnode * z) {
    z -> right = rightRotate(z -> right);

    return (leftRotate(z));
}

struct AVLnode * insert(struct AVLnode * node, int key) {

    if (node == NULL)
        return (newNode(key));

    if (key < node -> key)
        node -> left = insert(node -> left, key);
    else if (key > node -> key)
        node -> right = insert(node -> right, key);

    node -> height = (max(nodeHeight(node -> left), nodeHeight(node -> right)) + 1);

    int balance = heightDiff(node);

    if (balance > 1 && key < (node -> left -> key))
        return rightRotate(node);

    if (balance < -1 && key > (node -> right -> key))
        return leftRotate(node);

    if (balance > 1 && key > (node -> left -> key)) {
        node = LeftRightRotate(node);
    }

    if (balance < -1 && key < (node -> right -> key)) {
        node = RightLeftRotate(node);
    }

    return node;

}

struct AVLnode * delete(struct AVLnode * node, int queryNum) {
    if (node == NULL)
        return node;

    if (queryNum < node -> key)
        node -> left = delete(node -> left, queryNum);
    else if (queryNum > node -> key)
        node -> right = delete(node -> right, queryNum);
    else {
        if ((node -> left == NULL) || (node -> right == NULL)) {
            struct AVLnode * temp = node -> left ?
                node -> left :
                node -> right;

            if (temp == NULL) {
                temp = node;
                node = NULL;
            } else
                *node = * temp;

            free(temp);
        } else {
            struct AVLnode * temp = minNode(node -> right);
            node -> key = temp -> key;
            node -> right = delete(node -> right, temp -> key);
        }
    }

    if (node == NULL)
        return node;

    node -> height = (max(nodeHeight(node -> left), nodeHeight(node -> right)) + 1);

    int balance = heightDiff(node);

    if ((balance > 1) && (heightDiff(node -> left) >= 0))
        return rightRotate(node);

    if ((balance > 1) && (heightDiff(node -> left) < 0)) {
        node = LeftRightRotate(node);
    }

    if ((balance < -1) && (heightDiff(node -> right) >= 0))
        return leftRotate(node);

    if ((balance < -1) && (heightDiff(node -> right) < 0)) {
        node = RightLeftRotate(node);
    }

    return node;

}

struct AVLnode * findNode(struct AVLnode * node, int queryNum) {
    if (node != NULL) {
        if (queryNum < node -> key)
            node = findNode(node -> left, queryNum);
        else if (queryNum > node -> key)
            node = findNode(node -> right, queryNum);
    }

    return node;
}

void printPreOrder(struct AVLnode * node) {
    if (node == NULL)
        return;

    printf("  %d  ", (node -> key));
    printPreOrder(node -> left);
    printPreOrder(node -> right);
}

void printInOrder(struct AVLnode * node) {
    if (node == NULL)
        return;
    printInOrder(node -> left);
    printf("  %d  ", (node -> key));
    printInOrder(node -> right);
}

void printPostOrder(struct AVLnode * node) {
    if (node == NULL)
        return;
    printPostOrder(node -> left);
    printPostOrder(node -> right);
    printf("  %d  ", (node -> key));
}

int main() {
    int choice;
    int flag = 1;
    int insertNum;
    int queryNum;

    struct AVLnode * root = NULL;
    struct AVLnode * tempNode;

    while (flag == 1) {
        printf("\n\nEnter the Step to Run : \n");

        printf("\t1: Insert a node into AVL tree\n");
        printf("\t2: Delete a node in AVL tree\n");
        printf("\t3: Search a node into AVL tree\n");
        printf("\t4: printPreOrder (Ro L R) Tree\n");
        printf("\t5: printInOrder (L Ro R) Tree\n");
        printf("\t6: printPostOrder (L R Ro) Tree\n");
        printf("\t7: printAVL Tree\n");

        printf("\t0: EXIT\n");
        scanf("%d", & choice);

        switch (choice) {
        case 0:
            {
                flag = 0;
                printf("\n\t\tExiting, Thank You !!\n");
                break;
            }

        case 1:
            {

                printf("\n\tEnter the Number to insert: ");
                scanf("%d", & insertNum);

                tempNode = findNode(root, insertNum);

                if (tempNode != NULL)
                    printf("\n\t %d Already exists in the tree\n", insertNum);
                else {
                    printf("\n\tPrinting AVL Tree\n");
                    printAVL(root, 1);
                    printf("\n");

                    root = insert(root, insertNum);
                    printf("\n\tPrinting AVL Tree\n");
                    printAVL(root, 1);
                    printf("\n");
                }

                break;
            }

        case 2:
            {
                printf("\n\tEnter the Number to Delete: ");
                scanf("%d", & queryNum);

                tempNode = findNode(root, queryNum);

                if (tempNode == NULL)
                    printf("\n\t %d Does not exist in the tree\n", queryNum);
                else {
                    printf("\n\tPrinting AVL Tree\n");
                    printAVL(root, 1);
                    printf("\n");
                    root = delete(root, queryNum);

                    printf("\n\tPrinting AVL Tree\n");
                    printAVL(root, 1);
                    printf("\n");
                }

                break;
            }

        case 3:
            {
                printf("\n\tEnter the Number to Search: ");
                scanf("%d", & queryNum);

                tempNode = findNode(root, queryNum);

                if (tempNode == NULL)
                    printf("\n\t %d : Not Found\n", queryNum);
                else {
                    printf("\n\t %d : Found at height %d \n", queryNum, tempNode -> height);

                    printf("\n\tPrinting AVL Tree\n");
                    printAVL(root, 1);
                    printf("\n");
                }

                break;
            }

        case 4:
            {
                printf("\nPrinting Tree preOrder\n");
                printPreOrder(root);

                break;
            }

        case 5:
            {
                printf("\nPrinting Tree inOrder\n");
                printInOrder(root);

                break;
            }

        case 6:
            {
                printf("\nPrinting Tree PostOrder\n");
                printPostOrder(root);

                break;
            }

        case 7:
            {
                printf("\nPrinting AVL Tree\n");
                printAVL(root, 1);

                break;
            }

        default:
            {
                flag = 0;
                printf("\n\t\tExiting, Thank You !!\n");
                break;
            }

        }

    }

    return 0;
}