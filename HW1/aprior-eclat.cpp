#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <thread>
#include <mutex>
#include <omp.h>
#include <memory>
#include <numeric>
#include <unordered_set>
using namespace std;

using Itemset = vector<int>;
//using FreqTable = unordered_map<Itemset, int>;

struct
VectorHash {
    size_t operator()(const vector<int>& v) const {
        size_t seed = v.size();
        for (auto& i : v) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

bool
is_subset(const vector<int>& itemset, const vector<int>& transaction) {
    return includes(transaction.begin(), transaction.end(), itemset.begin(), itemset.end());
}

double
round_4(const double number) {
    return round(number * 1e4) / 1e4;
}

void
print4(ofstream& out, const double number, const Itemset& item) {
    bool first = true;
    for (auto i:item) {
        if (first == false) out<<',';
        out << i;
        first = false;
    }
    out << ':' << fixed << setprecision(4) << round_4(number) << '\n';
}

pair<int, unordered_map<int, unordered_set<int>>>
read_file(const string& input) {
    ifstream file(input);
    int transactions_size = 0;
    unordered_map<int, unordered_set<int>> freq;

    if (!file.is_open()) {
        cerr << "Can't open file: " << input << '\n';
        return {transactions_size, freq};
    }
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        int number;
        while (ss >> number) {
            freq[number].emplace(transactions_size);
            if (ss.peek() == ',') {
                ss.ignore();
            }
        }
        ++transactions_size;
    }

    file.close();
    return {transactions_size, freq};
}

unordered_map<int, unordered_set<int>>
filter_frequencyTable(const unordered_map<int, unordered_set<int>>& full_freq, const int& min_support) {
    unordered_map<int, unordered_set<int>> frequencyTable;

    for (auto& item:full_freq) {
        if (item.second.size() >= min_support){
            frequencyTable[item.first] = item.second;
        }
    }
    return frequencyTable;
}

// 判斷 itemset1 和 itemset2 是否可以合併產生下一層 candidate
bool can_merge(const Itemset& a, const Itemset& b) {
    for (size_t i = 0; i < a.size() - 1; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

Itemset
merge_itemsets(const Itemset& a, const Itemset& b) {
    if (a.back() < b.back()) {
        Itemset merged = a;
        merged.emplace_back(b.back());
        return merged;
    }
    Itemset merged = b;
    merged.emplace_back(a.back());
    return merged;
}

// 用於排序 Itemset，以便 unordered_map 作 key 時正確判斷相同與否
void
sort_and_unique(Itemset& itemset) {
    sort(itemset.begin(), itemset.end());
    itemset.erase(unique(itemset.begin(), itemset.end()), itemset.end());
}

void 
generate_frequent_itemsets(
const unordered_map<int, unordered_set<int>>& single_freq,
int min_support,
int total_transactions,
ofstream& out
) {
    unordered_map<Itemset, unordered_set<int>, VectorHash> prev_frequent;   
    unordered_map<Itemset, int, VectorHash> current_freq_table;

    for (const auto& [item, count] : single_freq) {
        if (count.size() >= min_support) {
            Itemset iset = {item};
            prev_frequent[iset] = count;
            out << item << ':';
            out << fixed << setprecision(4) << round_4(double(count.size()) / total_transactions) << '\n';
        }
    }

    int k = 2;
    while (!prev_frequent.empty()) {
        vector<Itemset> candidates;
        unordered_map<Itemset, unordered_set<int>, VectorHash> next_frequent;

        auto it1 = prev_frequent.begin();
        for (; it1 != prev_frequent.end(); ++it1) {
            auto it2 = it1;
            ++it2;
            for (; it2 != prev_frequent.end(); ++it2) {
                const Itemset& a = it1->first;
                const Itemset& b = it2->first;

                if (!can_merge(a, b)) continue;
                Itemset candidate = merge_itemsets(a, b);

                // 這裡是重點：由兩個 frequent 的 TID sets 做交集
                const auto& smaller = (it1->second.size() < it2->second.size()) ? it1->second : it2->second;
                const auto& larger  = (it1->second.size() < it2->second.size()) ? it2->second : it1->second;

                unordered_set<int> intersect_tids;
                for (int element : smaller) {
                    if (larger.find(element) != larger.end()) {
                        intersect_tids.emplace(element);
                    }
                }

                if (intersect_tids.size() < min_support) continue;

                next_frequent[candidate] = intersect_tids;
                print4(out, double(intersect_tids.size()) / total_transactions, candidate);
                
            }
        }

        prev_frequent = move(next_frequent);
        ++k;
    }

}


int
main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << "[min support] [input file] [output file]\n";
        return 1;
    }
    string input_file = argv[2], output_file = argv[3];
    double double_min_support = atof(argv[1]);
    if (double_min_support > 1 || double_min_support < 0) {
        cerr << "min support value should between 0 to 1.\n";
        return 1;
    }
    auto [transactions_size, full_freq] = read_file(input_file);
    int min_support = double_min_support * transactions_size;

    if (double(min_support)/transactions_size < double_min_support)  ++min_support;
    else if (double(min_support-1)/transactions_size >= double_min_support)  --min_support;
    
    auto freq= filter_frequencyTable(full_freq, min_support);
    if (freq.size() == 0) {
        ofstream out(output_file);
        if (!out.is_open()) {
            cerr << "Can't open file: " << output_file << '\n';
            return 1;
        }
        out.close();
        return 0;
    } else if (freq.size() == 1) {
        ofstream out(output_file);
        if (!out.is_open()) {
            cerr << "Can't open file: " << output_file << '\n';
            return 1;
        }
        out << freq.begin()->first << ':';
        out << fixed << setprecision(4) << round_4(double(freq.begin()->second.size()) / transactions_size) << '\n';
        out.close();
        return 0;
    }


    ofstream out(output_file);
    if (!out.is_open()) {
        cerr << "Can't open file: " << output_file << '\n';
        return 1;
    }

    generate_frequent_itemsets(freq, min_support, transactions_size, out);

    out.close();
    return 0;
}