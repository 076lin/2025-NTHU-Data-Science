#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <bitset>
using namespace std;
using Bit3 = bitset<(int)1e3>;
using Bit5 = bitset<(int)1e5>;

struct Itemset{
    Bit3 items;
    Bit5 set;
    Itemset(Bit3 _items, Bit5 _set): 
        items{_items}, set{_set}{}
    Itemset(int i, Bit5 _set): set{_set}{
        items.set(i);
    }
};

ofstream outfile;

vector<Bit5> tidset(1000);

int num_trans;
double min_sup;
int min_cnt;
int num_threads;

int input(string f_in) {
    ifstream in(f_in); 
    string line;
    int t = 0;
    if (!in.is_open()) {
        cerr << "Error: cannot open input file.\n";
        exit(1);
    }
    while(getline(in, line)) {
        stringstream ss(line);
        int num;
        while (ss >> num){
            tidset[num].set(t);
            if (ss.peek() == ',') {
                ss.ignore();
            }
        }
        ++t;
    }
    in.close();
    return t;
}

void write_item(const Bit3& item, const double support, stringstream& ss) {
    bool fst = 1;
    for(int i = 0; i < 1000; ++i) {
        if(item.test(i)){
            if(!fst) ss << ',';
            ss << i;
            fst = 0;
        }
    }
    ss << ':' << fixed << setprecision(4) << round(support * 1e4) / 1e4 << '\n';

}

void ECLAT(const vector<Itemset> &parentSet) {
    int n = parentSet.size();
    vector<vector<Itemset>> newSet(n);
    vector<stringstream> thread_local_output(num_threads);

    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < n; ++i) {
        int tid = omp_get_thread_num();
        for(int j = i + 1; j < n; ++j) {
            auto Set = parentSet[j].set & parentSet[i].set;
            double sup = Set.count();
            
            if(sup < min_cnt) continue;

            auto genItem = parentSet[i].items | parentSet[j].items;

            newSet[i].emplace_back(Itemset(genItem, Set));
            write_item(genItem, sup / num_trans, thread_local_output[tid]);
        }
    }
    for (int t = 0; t < num_threads; ++t)
        outfile << thread_local_output[t].str();
    
    for(auto i : newSet) if(i.size() > 1) {
        ECLAT(i);
    }
}

int main(int argc, char *argv[]) {
    if(argc < 4) {
        cerr<< "error" << endl;
        return 1;
    }
    num_threads = omp_get_max_threads();
    string f_in = argv[2], f_out = argv[3];
    min_sup = atof(argv[1]);

    num_trans = input(f_in);
    outfile.open(f_out);
    
    min_cnt = ceil(min_sup * num_trans);
    vector<stringstream> thread_local_output(num_threads);
    vector<vector<Itemset>> thread_local_initset(num_threads);
    vector<Itemset> initset;
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < 1000; i++) {
        double sup = tidset[i].count();

        if(sup < min_cnt) continue;

        int tid = omp_get_thread_num();
        thread_local_output[tid] << i << ':' << fixed << setprecision(4) << round(1e4 * sup / num_trans) / 1e4 << '\n';
        thread_local_initset[tid].emplace_back(Itemset(i, tidset[i]));
    }
    initset.reserve(1000);
    for (int t = 0; t < num_threads; ++t) {
        outfile << thread_local_output[t].str();
        for(auto item : thread_local_initset[t])
            initset.emplace_back(item);
    }

    ECLAT(initset);
    outfile.close();
    return 0;
}