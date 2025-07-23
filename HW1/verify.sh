#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: ./verify.sh [cpp file]"
    echo "For example: ./verify.sh 111020003_hw1.cpp"
    exit 1
fi

CPP_FILE=$1
TEST_COUNT=10
EXE_NAME=${CPP_FILE%.cpp}

echo "compile..."
g++ -std=c++2a -pthread -fopenmp -O2 -o $EXE_NAME $CPP_FILE
if [ $? -ne 0 ]; then
    echo "compile failed !"
    exit 1
fi

declare -a thresholds=(0.1 0.1 0.2 0.25 0.87 0.16 0.001 0.2 0.19 0.69)
all_passed=true

for (( i=1; i<=TEST_COUNT; i++ ))
do
    echo "===================="
    echo "Test #$i"
    echo "Running..."

    threshold=${thresholds[$((i-1))]}
    INPUT="verify_data/test_${i}.txt"
    OUTPUT="ans.txt"
    EXPECTED="verify_data/ans_${i}.txt"

    time ./$EXE_NAME $threshold $INPUT $OUTPUT

    result=$(python3 check.py $OUTPUT $EXPECTED | tail -n 1)
    
    if echo "$result" | grep -q "success"; then
        echo -e "\033[1;32mtest #$i success ouo.\033[0m"
    else
        echo -e "\033[1;31mtest #$i fail QAQ.\033[0m"
        all_passed=false
    fi
done

echo "===================="
if $all_passed; then
    echo -e "\033[1;32m🎉 all test success! Great Job!\033[0m"
else
    echo -e "\033[1;31m⚠️ some test fail QAQ.\033[0m"
fi
