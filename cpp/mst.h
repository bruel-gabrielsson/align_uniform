//#include <torch/extension.h>
#include <vector>

class Edge
{
  public:
    int node_start;
    int node_end;
    float weight;
    Edge(int node1, int node2, float wt);
};

class Graph{
    private:
      int num_nodes;
      std :: vector<Edge> edgelist; // edgelist will store the edges of minimum spanning tree
      std :: vector<int> parent;
      std :: vector<int> rank;

    public:
      Graph (int num_nodes);

      void AddEdge(Edge e);
      int FindParent(int node);
      //void KruskalMST(std :: vector<Edge>&);
      void KruskalMST (std :: tuple< std :: vector<int>, std :: vector<int> >&);
      void DisplayEdges(std :: vector<Edge>&);
};

// bool CompareCost (const Edge a, const Edge b);
//
// void Graph :: AddEdge (Edge e);
//
// int Graph :: FindParent (int node);
//
// void Graph :: KruskalMST (std :: tuple< std :: vector<int>, std :: vector<int> >& result);

std::tuple< std :: vector<int>, std :: vector<int> > MST(std::vector<std::vector<float>> distances);
