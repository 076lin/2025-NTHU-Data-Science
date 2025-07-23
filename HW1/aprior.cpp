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
using FreqTable = unordered_map<Itemset, int>;

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
    for (auto i:item){
        if(first == false) out<<',';
        out << i;
        first = false;
    }
    out << ':' << fixed << setprecision(4) << round_4(number) << '\n';
}

pair<vector<vector<int>>, unordered_map<int, int>>
read_file(const string& input) {
    ifstream file(input);
    vector<vector<int>> full_data;
    unordered_map<int, int> frequencyTable;

    if (!file.is_open()) {
        cerr << "Can't open file: " << input << '\n';
        return {full_data, frequencyTable};
    }
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        int number;
        char comma;
        vector<int> transaction;
        while (ss >> number) {
            transaction.emplace_back(number);
            ++frequencyTable[number];
            if (ss.peek() == ',') {
                ss >> comma;
            }
        }
        full_data.emplace_back(transaction);
    }

    file.close();
    return {full_data, frequencyTable};
}

pair<vector<vector<int>>, unordered_map<int, int>>
filter_frequencyTable(unordered_map<int, int>& full_freq, const vector<vector<int>>& full_transactions, const int& min_support) {
    unordered_map<int, int> frequencyTable;

    for (auto& item:full_freq) {
        if (item.second >= min_support){
            frequencyTable[item.first] = item.second;
        }
    }

    vector<vector<int>> filter_transactions;
    for (auto& item:full_transactions) {
        vector<int> filter_transaction;
        for (auto& i:item){
            if (frequencyTable.find(i) != frequencyTable.end()) {
                filter_transaction.emplace_back(i);
            }
        }
        sort(filter_transaction.begin(), filter_transaction.end());
        filter_transactions.emplace_back(filter_transaction);
    }
    return {filter_transactions, frequencyTable};
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
const vector<vector<int>>& transactions,
int min_support,
int total_transactions,
ofstream& out,
const unordered_map<int, int>& single_freq
) {
    vector<Itemset> prev_frequent;
    unordered_map<Itemset, int, VectorHash> current_freq_table;

    for (const auto& [item, count] : single_freq) {
        if (count >= min_support) {
            Itemset iset = {item};
            prev_frequent.push_back(iset);
            out << item << ':';
            out << fixed << setprecision(4) << round_4(double(count) / total_transactions) << '\n';
        }
    }

    int k = 2;
    while (!prev_frequent.empty()) {
        vector<Itemset> candidates;

        // 產生 candidate k-itemsets
        for (size_t i = 0; i < prev_frequent.size(); ++i) {
            for (size_t j = i + 1; j < prev_frequent.size(); ++j) {
                if (can_merge(prev_frequent[i], prev_frequent[j])) {
                    Itemset candidate = merge_itemsets(prev_frequent[i], prev_frequent[j]);
                    candidates.emplace_back(candidate);
                }
            }
        }

        current_freq_table.clear();
        // 計算每個 candidate 的支持度
        for (const auto& transaction : transactions) {
            if (transaction.size() < k) continue;
            for (const auto& candidate : candidates) {
                if (is_subset(candidate, transaction)) {
                    ++current_freq_table[candidate];
                }
            }
        }

        prev_frequent.clear();
        // 篩選出符合支持度門檻的
        for (const auto& [iset, count] : current_freq_table) {
            if (count >= min_support) {
                prev_frequent.emplace_back(iset);
                print4(out, double(count) / total_transactions, iset);
            }
        }
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
    auto [full_transactions, full_freq] = read_file(input_file);
    int transactions_size = full_transactions.size();
    int min_support = double_min_support * transactions_size;

    if (double(min_support)/transactions_size < double_min_support)  ++min_support;
    else if (double(min_support-1)/transactions_size >= double_min_support)  --min_support;
    
    auto [transactions, freq]= filter_frequencyTable(full_freq, full_transactions, min_support);
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
        out << fixed << setprecision(4) << round_4(double(freq.begin()->second) / transactions_size) << '\n';
        out.close();
        return 0;
    }


    ofstream out(output_file);
    if (!out.is_open()) {
        cerr << "Can't open file: " << output_file << '\n';
        return 1;
    }

    generate_frequent_itemsets(transactions, min_support, transactions_size, out, freq);

    out.close();
    return 0;
}