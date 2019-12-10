#include <stdio.h>
#include <stdlib.h>

#define SIZE 40

struct queue {
    int items[SIZE];
    int front;
    int rear;
};

struct queue * createQueue();
void enqueue(struct queue * q, int);
int dequeue(struct queue * q);
void display(struct queue * q);
int isEmpty(struct queue * q);
int pollQueue(struct queue * q);

struct node {
    int vertex;
    struct node * next;
};

struct node * createNode(int);

struct Graph {
    int numVertices;
    struct node ** adjLists;
    int * visited;
};
struct Graph * createGraph(int vertices);
void addEdge(struct Graph * graph, int src, int dest);
void printGraph(struct Graph * graph);
void bfs(struct Graph * graph, int startVertex);

int main() {
    int vertices, edges, source, i, src, dst;
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
    printf("Enter source of bfs\n");
    scanf("%d", & source);
    bfs(graph, source);

    return 0;
}
void bfs(struct Graph * graph, int startVertex) {
    struct queue * q = createQueue();

    graph -> visited[startVertex] = 1;
    enqueue(q, startVertex);
    printf("Breadth first traversal from vertex %d is:\n", startVertex);

    while (!isEmpty(q)) {
        printf("%d ", pollQueue(q));
        int currentVertex = dequeue(q);

        struct node * temp = graph -> adjLists[currentVertex];
        while (temp) {
            int adjVertex = temp -> vertex;
            if (graph -> visited[adjVertex] == 0) {
                graph -> visited[adjVertex] = 1;
                enqueue(q, adjVertex);
            }
            temp = temp -> next;
        }
    }
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

    newNode = createNode(src);
    newNode -> next = graph -> adjLists[dest];
    graph -> adjLists[dest] = newNode;
}
struct queue * createQueue() {
    struct queue * q = malloc(sizeof(struct queue));
    q -> front = -1;
    q -> rear = -1;
    return q;
}
int isEmpty(struct queue * q) {
    if (q -> rear == -1)
        return 1;
    else
        return 0;
}
void enqueue(struct queue * q, int value) {
    if (q -> rear == SIZE - 1)
        printf("\nQueue is Full!!");
    else {
        if (q -> front == -1)
            q -> front = 0;
        q -> rear++;
        q -> items[q -> rear] = value;
    }
}
int dequeue(struct queue * q) {
    int item;
    if (isEmpty(q)) {
        printf("Queue is empty");
        item = -1;
    } else {
        item = q -> items[q -> front];
        q -> front++;
        if (q -> front > q -> rear) {
            q -> front = q -> rear = -1;
        }
    }
    return item;
}

int pollQueue(struct queue * q) {
    return q -> items[q -> front];
}