
#include <iostream>
#include <fstream>
#include <queue>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>

#include "../../hnswlib/hnswlib.h"

std::vector<int> LoadAttr(std::string filename, int data_nb, std::vector<std::vector<int>>& attr)
{
    attr.resize(data_nb);
    std::ifstream infile(filename, std::ios::in | std::ios::binary);
    if (!infile.is_open())
    {
        throw Exception("cannot open " + filename);
    }
    for (int i = 0; i < data_nb; i++)
    {
        int len, val;
        infile.read((char *)&len, sizeof(int));
        attr[i].resize(len);
        for (int j = 0; j < len; j++)
        {
            infile.read((char *)&val, sizeof(int));
            attr[i][j] = val;
        }
    }
    infile.close();
}

class Attributes
{
public:
    int max_cate_size; // max cardinality of categorical attributes
    int num_size;  // cardinality of numerical attributes
    int vector_size; // size of vectors

    std::vector<std::vector<int>> cate_attrs; // categorical attributes
    std::vector<std::vector<int>> num_attrs; // numerical attributes

    Attribute() : cate_size(0), num_size(0), vector_size(0) {}
    Attribute(int cate_size, int num_size, int vector_size) : cate_size(cate_size), num_size(num_size), vector_size(vector_size) {}

    void LoadNumAttr(std::string filename)
    {
        LoadAttr(filename, num_size, num_attrs);
    }

    void LoadCateAttr(std::string filename) 
    {
        LoadAttr(filename, cate_size，cate_attrs);
    }
};

class DataLoader
{
public:
    int Dim, query_nb, query_K;
    std::vector<std::vector<float>> query_points;
    int data_nb;
    std::vector<std::vector<float>> data_points;
    Attributes attrs;

    std::unordered_map<int, std::vector<std::pair<int, int>>> query_range;
    std::unordered_map<int, std::vector<std::vector<int>>> groundtruth;

    // Used only when computing groundtruth and constructing index. Do not use this to load data for search process
    void LoadData(std::string filename)
    {
        if (filename == "")
            return;
        std::ifstream infile(filename, std::ios::in | std::ios::binary);
        if (!infile.is_open())
            throw Exception("cannot open " + filename);
        infile.read((char *)&data_nb, sizeof(int));
        infile.read((char *)&Dim, sizeof(int));
        data_points.resize(data_nb);
        for (int i = 0; i < data_nb; i++)
        {
            data_points[i].resize(Dim);
            infile.read((char *)data_points[i].data(), Dim * sizeof(float));
        }
        infile.close();
    }

    void LoadAttr(std::string num_filename, std::string cate_filtname)
    {
        LoadNumAttr(num_filename);
        LoadCateAttr(cate_filtname);
    }

}


int main(int argc, char **argv)
{
    std::unordered_map<std::string, std::string> paths;
    int N, M, vecdim, ef_construction, threads;
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--data_path")
            paths["data_vector"] = argv[i + 1];
        if (arg == "--index_file")
            paths["index_save"] = argv[i + 1];
        if (arg == "--num_attr_file")
            paths["num_attr_file"] = argv[i + 1];
        if (arg == "--cate_attr_file")
            paths["cate_attr_file"] = argv[i + 1];
        if (arg == "--M")
            M = std::stoi(argv[i + 1]);
        if (arg == "--N")
            N = std::stoi(argv[i + 1]);
        if (arg == "--ef_construction")
            ef_construction = std::stoi(argv[i + 1]);
        if (arg == "--threads")
            threads = std::stoi(argv[i + 1]);
        if (arg == "--dim")
            vecdim = std::stoi(argv[i + 1]);
    }

    if (paths["data_vector"] == "")
        throw Exception("data path is empty");
    if (paths["index_save"] == "")
        throw Exception("index path is empty");
    if (M <= 0)
        throw Exception("M should be a positive integer");
    if (ef_construction <= 0)
        throw Exception("ef_construction should be a positive integer");
    if (threads <= 0)
        throw Exception("threads should be a positive integer");

    
    DataLoader storage;
    storage.LoadData(paths["data_vector"]);
    storage.attrs.LoadAttr(paths["num_attr_file"], paths["cate_attr_file"]);

    L2Space l2space(vecdim);

    NSW<float> *nsw_alg;
    nsw_alg = new NSW<float>(&l2space, N, M, ef_construction);


    nsw_alg->constructIndex(storage.data_points, storage.attrs, threads);

    
}