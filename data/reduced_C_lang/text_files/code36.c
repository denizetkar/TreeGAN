#include <stdio.h>
#include <stdlib.h>

#define MAX_SIZE 40

struct node {
    int vertex;
    struct node * next;
};
struct node * createNode(int v);
struct Graph {
    int numVertices;
    int * visited;
    struct node ** adjLists;
};
struct Stack {
    int arr[MAX_SIZE];
    int top;
};
struct Graph * createGraph(int);
void addEdge(struct Graph * , int, int);
void printGraph(struct Graph * );
void topologicalSortHelper(int, struct Graph * , struct Stack * );
void topologicalSort(struct Graph * );
struct Stack * createStack();
void push(struct Stack * , int);
int pop(struct Stack * );

int main() {
    int vertices, edges, i, src, dst;
    printf("Enter the number of vertices\n");
    scanf("%d", & vertices);
    struct Graph * graph = createGraph(vertices);
    printf("Enter the number of edges\n");
    scanf("%d", & edges);
    for (i = 0; i < edges; i++) {
        printf("Edge %d \nEnter source: ", i + 1);
        scanf("%d", & src);
        printf("Enter destination: ");
        scanf("%d", & dst);
        addEdge(graph, src, dst);
    }
    printf("One topological sort order is:\n");
    topologicalSort(graph);
    printf("\n");

    return 0;
}

void topologicalSortHelper(int vertex, struct Graph * graph, struct Stack * stack) {
    graph -> visited[vertex] = 1;
    struct node * adjList = graph -> adjLists[vertex];
    struct node * temp = adjList;
    while (temp != NULL) {
        int connectedVertex = temp -> vertex;
        if (graph -> visited[connectedVertex] == 0) {
            topologicalSortHelper(connectedVertex, graph, stack);
        }
        temp = temp -> next;
    }
    push(stack, vertex);
}

void topologicalSort(struct Graph * graph) {
    struct Stack * stack = createStack();
    int i = 0;
    for (i = 0; i < graph -> numVertices; i++) {
        if (graph -> visited[i] == 0) {
            topologicalSortHelper(i, graph, stack);
        }
    }
    while (stack -> top != -1)
        printf("%d ", pop(stack));
}
struct node * createNode(int v) {
    struct node * newNode = malloc(sizeof(struct node));
    newNode -> vertex = v;
    newNode -> next = NULL;
    return newNode;
}
struct Graph * createGraph(int vertices) {
    struct Graph * graph = malloc(sizeof(struct Graph));
    graph -> numVertices = vertices;
    graph -> adjLists = malloc(vertices * sizeof(struct node * ));
    graph -> visited = malloc(vertices * sizeof(int));

    int i;
    for (i = 0; i < vertices; i++) {
        graph -> adjLists[i] = NULL;
        graph -> visited[i] = 0;
    }
    return graph;
}
void addEdge(struct Graph * graph, int src, int dest) {
    struct node * newNode = createNode(dest);
    newNode -> next = graph -> adjLists[src];
    graph -> adjLists[src] = newNode;
}
void printGraph(struct Graph * graph) {
    int v;
    for (v = 0; v < graph -> numVertices; v++) {
        struct node * temp = graph -> adjLists[v];
        printf("\n Adjacency list of vertex %d\n ", v);
        while (temp) {
            printf("%d -> ", temp -> vertex);
            temp = temp -> next;
        }
        printf("\n");
    }
}
struct Stack * createStack() {
    struct Stack * stack = malloc(sizeof(struct Stack));
    stack -> top = -1;
}
void push(struct Stack * stack, int element) {
    stack -> arr[++stack -> top] = element;
}
int pop(struct Stack * stack) {
    if (stack -> top == -1)
        return INT_MIN;
    else
        return stack -> arr[stack -> top--];
}