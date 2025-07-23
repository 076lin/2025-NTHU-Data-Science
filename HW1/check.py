#!/usr/bin/env python3
import sys

def parse_data(data: list) -> dict:
    if len(data) == 0:
        return {}
    parsed = {}
    for line in data:
        line = line.strip()
        key, value = line.split(':')
        key_parts = tuple(sorted(map(int, key.split(','))))
        # 為了避免精度丟失，使用字串紀錄
        parsed[key_parts] = str(value)
    return parsed

def verify(output_path, answer_path):
    try:
        with open(output_path, 'r') as output_f, open(answer_path, 'r') as answer_f:
            out = output_f.readlines()
            ans = answer_f.readlines()
        if len(ans) != len(out):
            print('\n\033[1;31;48mfail QAQ.\033[1;37;0m')
            return
        
        output_lines = parse_data(out)
        answer_lines = parse_data(ans)

        if output_lines != answer_lines:
            print('\n\033[1;31;48mfail QAQ.\033[1;37;0m')
        else:
            print('\n\033[1;32;48msuccess ouo.\033[1;37;0m')

    except FileNotFoundError as e:
        print(f"Error: {e}")
if __name__ == "__main__":
    # 檢查是否提供了足夠的參數
    if len(sys.argv) != 3:
        print("Usage: ./check.py [file to be verified] [answer file]")
        sys.exit(1)  # 退出程式，狀態碼 1 代表錯誤
    
    # 讀取命令列參數
    output_path = sys.argv[1]
    answer_path = sys.argv[2]

    # 執行驗證函式
    verify(output_path, answer_path)