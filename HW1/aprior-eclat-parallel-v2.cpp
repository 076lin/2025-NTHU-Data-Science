#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <unordered_set>
#include <bitset>
using namespace std;

using Itemset = vector<int>;
constexpr int MAX_TRANSACTIONS = (int)1e5;
using Bitset = bitset<MAX_TRANSACTIONS>;

struct
VectorHash {
    size_t operator()(const Itemset& v) const {
        size_t seed = v.size();
        for (auto& i : v) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

double
round_4(const double number) {
    return round(number * 1e4) / 1e4;
}

void
print4(ostream& out, const double number, const Itemset& item) {
    bool first = true;
    for (auto i : item) {
        if (!first) out << ',';
        out << i;
        first = false;
    }
    out << ':' << fixed << setprecision(4) << round_4(number) << '\n';
}

pair<int, unordered_map<int, Bitset>>
read_file(const string& input) {
    ifstream file(input);
    int transactions_size = 0;
    unordered_map<int, Bitset> freq;

    if (!file.is_open()) {
        cerr << "Can't open file: " << input << '\n';
        return {transactions_size, freq};
    }
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        int number;
        while (ss >> number) {
            freq[number].set(transactions_size);
            if (ss.peek() == ',') {
                ss.ignore();
            }
        }
        ++transactions_size;
    }

    file.close();
    return {transactions_size, freq};
}

unordered_map<int, Bitset>
filter_frequencyTable(unordered_map<int, Bitset>& full_freq, const int& min_support) {
    unordered_map<int, Bitset> frequencyTable;

    for (auto& [key, value] : full_freq) {
        if ((int)value.count() >= min_support) {
            frequencyTable[key] = move(value);
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
    const unordered_map<int, Bitset>& single_freq,
    int min_support,
    int total_transactions,
    ofstream& out
) {
    unordered_map<Itemset, Bitset, VectorHash> prev_frequent;
    
    for (const auto& [item, count] : single_freq) {
        if (count.count() >= min_support) {
            Itemset iset = {item};
            prev_frequent[iset] = count;
            out << item << ':' << fixed << setprecision(4) << round_4(double(count.count()) / total_transactions) << '\n';
        }
    }

    int k = 2;
    while (!prev_frequent.empty()) {
        unordered_map<Itemset, Bitset, VectorHash> next_frequent;

        vector<pair<Itemset, Bitset>> local_results;

        int num_threads = omp_get_max_threads();
        vector<unordered_map<Itemset, Bitset, VectorHash>> thread_local_freq(num_threads);
        vector<stringstream> thread_local_output(num_threads);

        vector<Itemset> keys;
        keys.reserve(prev_frequent.size());
        for (auto& p : prev_frequent) keys.emplace_back(p.first);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < (int)keys.size(); ++i) {
            int tid = omp_get_thread_num();
            for (int j = i + 1; j < (int)keys.size(); ++j) {
                const Itemset& a = keys[i];
                const Itemset& b = keys[j];
                if (!can_merge(a, b)) continue;

                Bitset intersect = prev_frequent[a] & prev_frequent[b];
                int support = intersect.count();

                if (support < min_support) continue;

                Itemset candidate = merge_itemsets(a, b);
                thread_local_freq[tid][candidate] = intersect;
                print4(thread_local_output[tid], double(support) / total_transactions, candidate);
            }
        }

        for (int t = 0; t < num_threads; ++t) {
            for (auto& [k, v] : thread_local_freq[t]) {
                next_frequent[k] = move(v);
            }
            out << thread_local_output[t].str();
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