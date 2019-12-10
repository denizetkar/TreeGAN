#include<stdio.h>
#include<stdlib.h>
#include<limits.h>
#include<string.h>

struct Graph {
    int vertexNum;
    int ** edges;
};

void createGraph(struct Graph * G, int V) {
    G -> vertexNum = V;
    G -> edges = (int ** ) malloc(V * sizeof(int * ));
    for (int i = 0; i < V; i++) {
        G -> edges[i] = (int * ) malloc(V * sizeof(int));
        for (int j = 0; j < V; j++)
            G -> edges[i][j] = INT_MAX;
        G -> edges[i][i] = 0;
    }
}

void addEdge(struct Graph * G, int src, int dst, int weight) {
    G -> edges[src][dst] = weight;
}

void print(int dist[], int V) {
    printf("\nThe Distance matrix for Floyd - Warshall\n");
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {

            if (dist[i * V + j] != INT_MAX)
                printf("%d\t", dist[i * V + j]);
            else
                printf("INF\t");
        }
        printf("\n");
    }
}

void FloydWarshall(struct Graph * graph) {
    int V = graph -> vertexNum;
    int dist[V][V];

    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++)
            dist[i][j] = graph -> edges[i][j];

    for (int k = 0; k < V; k++)

        for (int i = 0; i < V; i++)

            for (int j = 0; j < V; j++)

                if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX && dist[i][k] + dist[k][j] < dist[i][j])
                    dist[i][j] = dist[i][k] + dist[k][j];

    int dist1d[V * V];
    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++)
            dist1d[i * V + j] = dist[i][j];

    print(dist1d, V);
}

int main() {
    int V, E;
    int src, dst, weight;
    struct Graph G;
    printf("Enter number of vertices: ");
    scanf("%d", & V);
    printf("Enter number of edges: ");
    scanf("%d", & E);
    createGraph( & G, V);
    for (int i = 0; i < E; i++) {
        printf("\nEdge %d \nEnter source: ", i + 1);
        scanf("%d", & src);
        printf("Enter destination: ");
        scanf("%d", & dst);
        printf("Enter weight: ");
        scanf("%d", & weight);
        addEdge( & G, src, dst, weight);
    }
    FloydWarshall( & G);

    return 0;
}