import json

num = 100 * 10000
data_path = r"H:\迅雷下载\input\LCCC\LCCD_train.json"
save_path = "../data/lccc_chat.txt"

write_data = []
with open(data_path, "r", encoding="utf8") as fr:
    for line in fr:
        line_di = json.loads(line.strip())
        pair = line_di["conversation"]
        if len(pair) != 2 or "\t" in pair[0] or "\t" in pair[1]:
            continue
        write_data.append("\t".join(pair) + "\n")
        if len(write_data) > num:
            break
with open(save_path, "w", encoding="utf8") as fw:
    fw.writelines(write_data)
