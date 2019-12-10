#include<stdio.h>
#include<stdlib.h>
#include<limits.h>
#include<string.h>

struct Edge {
    int src, dst, weight;
};

struct Graph {
    int vertexNum;
    int edgeNum;
    struct Edge * edges;
};

void createGraph(struct Graph * G, int V, int E) {
    G -> vertexNum = V;
    G -> edgeNum = E;
    G -> edges = (struct Edge * ) malloc(E * sizeof(struct Edge));
}

void addEdge(struct Graph * G, int src, int dst, int weight) {
    static int ind;
    struct Edge newEdge;
    newEdge.src = src;
    newEdge.dst = dst;
    newEdge.weight = weight;
    G -> edges[ind++] = newEdge;
}

int minDistance(int mdist[], int vset[], int V) {
    int minVal = INT_MAX, minInd;
    for (int i = 0; i < V; i++)
        if (vset[i] == 0 && mdist[i] < minVal) {
            minVal = mdist[i];
            minInd = i;
        }

    return minInd;
}

void print(int dist[], int V) {
    printf("\nVertex  Distance\n");
    for (int i = 0; i < V; i++) {
        if (dist[i] != INT_MAX)
            printf("%d\t%d\n", i, dist[i]);
        else
            printf("%d\tINF", i);
    }
}

void BellmanFord(struct Graph * graph, int src) {
    int V = graph -> vertexNum;
    int E = graph -> edgeNum;
    int dist[V];

    for (int i = 0; i < V; i++)
        dist[i] = INT_MAX;
    dist[src] = 0;

    for (int i = 0; i <= V - 1; i++)
        for (int j = 0; j < E; j++) {
            int u = graph -> edges[j].src;
            int v = graph -> edges[j].dst;
            int w = graph -> edges[j].weight;

            if (dist[u] != INT_MAX && dist[u] + w < dist[v])
                dist[v] = dist[u] + w;
        }

    for (int j = 0; j < E; j++) {
        int u = graph -> edges[j].src;
        int v = graph -> edges[j].dst;
        int w = graph -> edges[j].weight;

        if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
            printf("Graph contains negative weight cycle. Hence, shortest distance not guaranteed.");
            return;
        }
    }

    print(dist, V);

    return;
}

int main() {
    int V, E, gsrc;
    int src, dst, weight;
    struct Graph G;
    printf("Enter number of vertices: ");
    scanf("%d", & V);
    printf("Enter number of edges: ");
    scanf("%d", & E);
    createGraph( & G, V, E);
    for (int i = 0; i < E; i++) {
        printf("\nEdge %d \nEnter source: ", i + 1);
        scanf("%d", & src);
        printf("Enter destination: ");
        scanf("%d", & dst);
        printf("Enter weight: ");
        scanf("%d", & weight);
        addEdge( & G, src, dst, weight);
    }
    printf("\nEnter source:");
    scanf("%d", & gsrc);
    BellmanFord( & G, gsrc);

    return 0;
}