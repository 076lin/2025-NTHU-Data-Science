#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <unordered_set>
#include <bitset>
using namespace std;

using Itemset = vector<int>;
constexpr int MAX_TRANSACTIONS = (int)1e5;
using Bitset = bitset<MAX_TRANSACTIONS>;

double
round_4(const double& number) {
    return round(number * 1e4) / 1e4;
}

void
print4(ostream& out, const double& number, const Itemset& item) {
    bool first = true;
    for (auto& i : item) {
        if (!first) out << ',';
        out << i;
        first = false;
    }
    out << ':' << fixed << setprecision(4) << (round(number * 1e4) / 1e4) << '\n';
}

pair<int, vector<pair<int, Bitset>>>
read_file(const string& input) {
    ifstream file(input);
    int transactions_size = 0;
    vector<pair<int, Bitset>> freq(1000);
    for(int i=0;i<1000;++i) freq[i].first = i;

    if (!file.is_open()) {
        cerr << "Can't open file: " << input << '\n';
        return {transactions_size, freq};
    }
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        int number;
        while (ss >> number) {
            freq[number].second.set(transactions_size);
            if (ss.peek() == ',') {
                ss.ignore();
            }
        }
        ++transactions_size;
    }

    file.close();
    return {transactions_size, freq};
}

vector<pair<int, Bitset>>
filter_frequencyTable(vector<pair<int, Bitset>>& full_freq, const int& min_support) {
    vector<pair<int, Bitset>> frequencyTable;

    for (auto& item : full_freq) {
        if ((int)item.second.count() >= min_support) {
            frequencyTable.emplace_back(item);
        }
    }

    return frequencyTable;
}

bool
can_merge(const Itemset& a, const Itemset& b) {
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

void
generate_frequent_itemsets(
    const vector<pair<int, Bitset>>& single_freq,
    int min_support,
    int total_transactions,
    ofstream& out
) {
    vector<pair<Itemset, Bitset>> prev_frequent;
    
    for (const auto& [item, count] : single_freq) {
        Itemset iset = {item};
        prev_frequent.push_back({iset, count});
        out << item << ':' << fixed << setprecision(4) << round_4(double(count.count()) / total_transactions) << '\n';
    }

    while (!prev_frequent.empty()) {
        vector<pair<Itemset, Bitset>> next_frequent;

        int num_threads = omp_get_max_threads();
        vector<vector<pair<Itemset, Bitset>>> thread_local_freq(num_threads);
        vector<stringstream> thread_local_output(num_threads);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < (int)prev_frequent.size(); ++i) {
            int tid = omp_get_thread_num();
            for (int j = i + 1; j < (int)prev_frequent.size(); ++j) {
                const Itemset& a = prev_frequent[i].first;
                const Itemset& b = prev_frequent[j].first;
                if (!can_merge(a, b)) continue;

                Bitset intersect = (prev_frequent[i].second) & (prev_frequent[j].second);
                int support = intersect.count();

                if (support < min_support) continue;

                Itemset candidate = merge_itemsets(a, b);
                thread_local_freq[tid].push_back({candidate, intersect});
                print4(thread_local_output[tid], double(support) / total_transactions, candidate);
            }
        }
        size_t sum = 0;
        for (int t = 0; t < num_threads; ++t) sum += thread_local_freq[t].size();
        next_frequent.reserve(sum);
        for (int t = 0; t < num_threads; ++t) {
            for (auto& item : thread_local_freq[t]) {
                next_frequent.emplace_back(item);
            }
            out << thread_local_output[t].str();
        }

        prev_frequent = move(next_frequent);
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
    if (double_min_support > 1 || double_min_support <= 0) {
        cerr << "min support must be a positive number within the interval (0, 1]. Got " << double_min_support << '\n';
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
        out << fixed << setprecision(4) << round_4(double(freq.begin()->second.count()) / transactions_size) << '\n';
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