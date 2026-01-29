import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Configuration: only modify these variables when running for different TV series on different machines
TV_name = "The Big Bang Theory"    # "The Big Bang Theory", "I love my family"
XLSX_PATH = f"F:\\data\\TV_series\\tv_data\\{TV_name}\\text_annotated_new.xlsx"
OUT_DIR = "D:\\wangchen\\Research\\tv_series_plus\\3D-Speaker\\docs\\figs\\name_mention"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = f"{TV_name}.png"


# --- name dictionaries ---
name_dic_BB = {'Sheldon': [['谢尔顿·库珀博士'], 
                           ['谢尔顿·库珀', '库珀博士'], 
                           ['库珀', '谢尔顿']], 
               'Leonard': [['莱纳德·霍夫斯塔德博士'], 
                           ['霍夫斯塔德博士', '莱纳德·霍夫斯塔德'], 
                           ['莱纳德', '霍夫斯塔德']], 
               'Penny': [['佩妮']], 
               'Howard': [['霍华德·沃尔维茨', '霍华德·沃罗威茨'], 
                          ['霍华德', '沃尔维茨', '沃罗威茨']], 
               'Raj': [['拉杰·库萨帕里'], 
                       ['拉杰', '库萨帕里']], 
               'Others': [['小拉丽塔', '罗德·泰勒', '盖博豪斯博士'], 
                          ['丹尼斯', '伊米尔·法尔曼法尔米安博士', '克里斯汀', '克里顿', '凯尔', '卢克', '史巴克', '哈里', '哈雷', '埃尔文', '夏纳', '奥本海默', '威廉', '尼奥', '帕特尔', '帕萨迪纳', '弗拉多', '彼得', '德法布', '柯克舰长', '沃森博士', '沃罗威茨', '海伦', '潘查理公主', '爱因斯坦', '牛顿', '玛丽', '种马博士', '米茜', '艾尔顿·约翰', '艾瑞克', '苏菲', '莫扎特', '莫洛克', '莱恩小姐', '莱斯利', '谢利', '贝尔夫人', '贝尔特', '里克', '里奥', '金先生', '阿基米德', '雅克·库斯托', '麦克',
                          '拉丽塔', '泰勒', '盖博豪斯', '罗德']]
                          }

name_dic_IL = {'Fulao': [['老傅同志'], 
                        ['傅明同志', '傅某人', '傅伯伯', '傅局长', '老局长', '老傅', '傅老', '贾敬贤']],
               'Heping': [['和平同志', '和平老大妈', '和平女侠'], 
                        ['和平']],
               'Zhixin': [['志新哥哥', '贾志新同志'], 
                        ['贾志新', '志新同志', '志新哥'], 
                        ['贾总', '贾炕', '志新']],
               'Zhiguo': [['贾志国同志', '贾志国先生'], 
                        ['贾志国', '志国同志'], 
                        ['志国哥哥', '志国', '贾先生', '贾总']],
               'Yuanyuan': [['贾圆圆同学', '贾圆圆同志', '贾圆圆歌友'], 
                        ['贾圆圆'], 
                        ['圆圆', '坚妮']],
               'Xiaofan': [['贾小凡同志', '贾小凡同学'], 
                        ['贾小凡', '小凡姐', '小凡妹妹', '小凡姑娘'], 
                        ['小凡']],
               'Xiaozhang': [['小张阿姨', '凤姑妹子', '凤姑妹妹', '张凤姑', '张姑奶奶'], 
                        ['小张', '凤姑', '小保姆']],
               'Yanhong': [['燕红妹妹', '燕红阿姨', '郑燕红同志'], 
                        ['郑燕红'], 
                        ['燕红']],
               'Others': [['丽达姑娘','宝财哥', '纪春生', '春花姐'], 
                          ['三毛', '丽丽', '余主任', '余大妈', '余小姐', '侯小姐', '刘宇航', '刘建军', '刘文彩', '刘胡兰', '刘颖', '叶之贤', '周厂长', '姓孟的', '孟德', '孟昭晖', '孟老师', '小余', '小刘', '小加', '小史姑娘', '小方', '小晴', '小秘', '小翠', '小许', '尤主编', '张先生', '张国荣', '张欣来', '彭老师', '文怡', '方小姐', '昆儿哥',  '朱总司令', '李秘', '杨晶晶', '毛主席', '王总', '王老师', '童安格', '老二', '老和同志', '老四', '老头儿', '老李', '老玉米', '老赵', '老郑', '胡大个儿', '费丽斯', '邱少云', '郑伯伯', '郑千里', '郑爷爷', '阿东', '阿姨', '阿巩', '阿庆', '阿敏', '阿昆', '阿欢', '阿玉', '阿英', '阿荣', '顾科长', '马小姐', '齐大姐', '齐大姨',
                          '丽达', '宝财', '春生', '春花']]
                          }

if TV_name == "The Big Bang Theory":
    name_dic = name_dic_BB
elif TV_name == "I love my family":
    name_dic = name_dic_IL
else:
    print(f"Unknown TV series name: {TV_name}")
    sys.exit(1)

def flatten_values(d):
    s = set()
    for v in d.values():
        for inner in v:
            for x in inner:
                s.add(str(x))
    return s


def build_alias_to_key(d):
    m = {}
    for key, v in d.items():
        for inner in v:
            for x in inner:
                m[str(x)] = key
    return m


def main():
    xls_df = pd.read_excel(XLSX_PATH)

    # preprocess 'name' column
    ## sort tokens in each cell alphabetically and join back with comma
    xls_df['name'] = xls_df['name'].apply(lambda x: ','.join(sorted(str(x).split(','))))
    ## collect all unique name tokens
    unique_entries = xls_df['name'].dropna().unique()
    unique_entries = [e for e in unique_entries if str(e).lower() != 'nan']
    all_tokens = set()
    for e in unique_entries:
        for t in str(e).split(','):
            tok = t.strip()
            if tok:
                all_tokens.add(tok)
    ## print tokens not covered by predefined name_dic
    flat = flatten_values(name_dic)
    uncovered = sorted([t for t in all_tokens if t not in flat])
    print("Tokens from annotated files not found in chosen name_dic values:")
    for t in uncovered:
        print(t)
    ## build alias->key map
    alias_to_key = build_alias_to_key(name_dic)

    # prepare rows (speakers) and columns
    speakers_raw = xls_df['speaker'].dropna().unique()
    if TV_name == "I love my family":
        # special handling: map Chinese names to canonical keys
        special_map = {
            "傅老": "Fulao",
            "和平": "Heping",
            "志新": "Zhixin",
            "志国": "Zhiguo",
            "圆圆": "Yuanyuan",
            "小凡": "Xiaofan",
            "小张": "Xiaozhang",
            "燕红": "Yanhong",
            "Others": "Others"
        }
        speaker_keys = [special_map.get(str(s), str(s)) for s in speakers_raw]
    else:
        speaker_keys = list(speakers_raw)
    speaker_keys = list(sorted(set(speaker_keys)))
    # initialize count DataFrame
    name_dic_keys = list(name_dic.keys())
    df_counts = pd.DataFrame(0, index=speaker_keys, columns=name_dic_keys, dtype=int)

    # iterate rows and fill counts
    for _, r in xls_df.iterrows():
        sp_raw = r.get('speaker', None)
        if pd.isna(sp_raw) or str(sp_raw).strip() == 'nan':
            continue
        sp_key = special_map[str(sp_raw)] if TV_name == "I love my family" else str(sp_raw)

        names_cell = str(r.get('name', ''))
        cell_tokens = set(t.strip() for t in names_cell.split(',') if t.strip())
        if not cell_tokens or (len(cell_tokens) == 1 and 'nan' in cell_tokens) or names_cell == 'nan':
            continue

        # for each canonical target key, check if any token in cell matches any alias group for that key
        for tok in cell_tokens:
            name_key = alias_to_key.get(tok, "Others")
            df_counts.at[sp_key, name_key] += 1

    # reorder columns according to desired name_notation order if possible
    if TV_name == "I love my family":
        name_notation = {
            '傅老': r"$\varrho_1$",
            '和平': r"$\varrho_2$",
            '志新': r"$\varrho_3$",
            '志国': r"$\varrho_4$",
            '圆圆': r"$\varrho_5$",	
            '小凡': r"$\varrho_6$",
            '小张': r"$\varrho_7$",
            '燕红': r"$\varrho_8$",
            'Others': r"$\varrho_s$"
        }
        name_notation = {special_map[k]: v for k, v in name_notation.items()}
    else:
        name_notation = {
            'Sheldon': r"$\varrho_1$",
            'Leonard': r"$\varrho_2$",
            'Penny': r"$\varrho_3$",
            'Howard': r"$\varrho_4$",
            'Raj': r"$\varrho_5$",	
            'Others': r"$\varrho_s$"
        }
    desired = list(name_notation.keys())

    # reorder rows and columns according to desired order if possible
    new_row_order = [s for s in desired if s in df_counts.index]
    df_counts = df_counts.reindex(new_row_order)
    new_col_order = [n for n in desired if n in df_counts.columns]
    df_counts = df_counts.reindex(columns=new_col_order)

    print("Raw counts:")
    print(df_counts)
    # convert to float and row-normalize
    df_heat = df_counts.astype(float)
    row_sums = df_heat.sum(axis=1)
    # avoid division by zero
    for idx in df_heat.index:
        s = row_sums.loc[idx]
        if s > 0:
            df_heat.loc[idx] = df_heat.loc[idx] / s
    # replace row/column labels with notation
    df_heat.index = [name_notation[s] for s in df_heat.index]
    df_heat.columns = [name_notation[n] for n in df_heat.columns]
    # plot
    plt.style.use('ggplot')
    fig, ax = plt.subplots()

    sns.heatmap(
        df_heat,
        cmap='Blues',
        annot=np.around(df_heat, 2),
        cbar_kws={'ticks': np.arange(0, 1.01, 0.2)},
        vmin=0,
        vmax=1,
        annot_kws={'fontsize': 10},
        fmt='.2f',
        ax=ax
    )

    plt.tick_params(labelsize=12)
    ax.set_ylabel('Speaker of current line', fontsize=12)
    ax.set_xlabel('Names mentioned in current line', fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, OUT_PNG), dpi=300)
    print(f"Saved heatmap to {os.path.join(OUT_DIR, OUT_PNG)}")


if __name__ == '__main__':
    main()
