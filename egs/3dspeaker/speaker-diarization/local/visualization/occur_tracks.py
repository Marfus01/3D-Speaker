import os, re, json, pickle, random
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

name_dic_BB = {'Sheldon': [['谢尔顿·库珀博士'], 
                           ['谢尔顿·库珀', '库珀博士'], 
                           ['库珀', '谢尔顿']], 
               'Leonard': [['莱纳德·霍夫斯塔德博士'], 
                           ['霍夫斯塔德博士', '莱纳德·霍夫斯塔德'], 
                           ['莱纳德', '霍夫斯塔德']], 
               'Penny': [['佩妮']], 
               'Howard': [['霍华德·沃尔维茨', '霍华德·沃罗威茨'], 
                          ['霍华德', '沃尔维茨', '沃罗威茨']], 
               'Rajesh': [['拉杰·库萨帕里'], 
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

def find_name_in_texts(name_dic, txt_folder, save_path):
    """在文本文件中查找人物名字出现的位置。
    Args:
        name_dic: 包含人物别名映射的字典
        txt_folder: 包含文本文件的文件夹路径
        save_path: 结果JSON文件的保存路径
    
    Returns:
        dict: key为人物名称，value为包含该人物别名出现的音频片段ID的列表
    """
    txt_files = [f for f in os.listdir(txt_folder) if f.lower().endswith('.txt')]
    name_dic_all = {k: [name for sublist in name_list for name in sublist] for k, name_list in name_dic.items()}
    result = {k: [] for k in name_dic.keys()}

    for file in sorted(txt_files):
        episode_name = os.path.splitext(file)[0]
        with open(os.path.join(txt_folder, file), 'r', encoding='utf-8') as f:
            for raw in f:
                line = raw.rstrip('\n')
                if not line:
                    continue
                
                parts = line.split('|')
                line_idx = int(parts[1]) - 1
                audio_seg_id = f"{episode_name}-{line_idx}"
                
                text = parts[-1]
                for key, names_all in name_dic_all.items():
                    counts = list(map(lambda name: len(re.findall(re.escape(name), text)), names_all))
                    if sum(counts) > 0:
                        result[key].append(audio_seg_id)

    with open(save_path, 'w', encoding='utf-8') as outf:
        json.dump(result, outf, ensure_ascii=False, indent=2)
    return result

def devide_episode(data, lengths_episode):
	'''
	data(np.array): (line_num, character_num)
	lengths_episode(list): 每集台词数
	'''
	episode_num = len(lengths_episode) # 集数
	data_list = []
	total_num = 0
	for i in range(episode_num):
		num = lengths_episode[i]
		data_list.append(data[total_num : total_num + num])
		
		total_num += num
	return data_list

def plot_one_block(ax, face_episdode, speaker_episode, name_episode, colors_dict, max_num_line, bar_width = 0.8, y_label=None):
    num_line, num_character = face_episdode.shape  # 此集台词数数，角色数
    colors = list(colors_dict.values())
    scale = max_num_line / num_line  # 对每集的归一化系数

    for i in range(num_character):  # 每个人物
        k = num_character - 1 - i  # 第k个角色，对应第i行（反序）
        for j in range(num_line):  # 每句台词
            # face_episdode,speaker_episode, name_episode: (num_line, num_character)
            # name数据
            if name_episode[j, k] == 1:  # 在第0行绘制Name
                ax.broken_barh([(j *scale, 1*scale)], (0, bar_width), facecolors=colors[k])
            # speaker 数据
            if speaker_episode[j, k] == 1: # 在第2行绘制Name
                ax.broken_barh([(j * scale, 1*scale)], (2, bar_width), facecolors=colors[k])
            # face数据
            if face_episdode[j, k] == 1: # 在第(4 + i)行绘制 第k个角色（反序）
                ax.broken_barh([(j * scale, 1 * scale)], (i + 4, bar_width), facecolors=colors[k])

        # face黑色边框
        ax.plot([0, num_line], [i + 4, i + 4], color='black', lw=1)  # top
        ax.plot([0, num_line], [i + 4 + bar_width, i + 4 + bar_width], color='black', lw=1)  # bottom
        ax.plot([0, 0], [i + 4, i + 4 + bar_width], color='black', lw=1)  # left
        ax.plot([num_line, num_line],[i + 4, i + 4 + bar_width], color='black', lw=2)  # right

    # speaker黑色边框
    ax.plot([0, num_line], [2, 2], color='black', lw=1)  # top
    ax.plot([0, num_line], [2 + bar_width, 2 + bar_width], color='black', lw=1)  # bottom
    ax.plot([0, 0], [2, 2 + bar_width], color='black', lw=1)  # left
    ax.plot([num_line, num_line],[2, 2 + bar_width], color='black', lw=2)  # right

    # name黑色边框
    ax.plot([0, num_line], [0, 0], color='black', lw=1)  # top
    ax.plot([0, num_line], [bar_width, bar_width], color='black', lw=1)  # bottom
    ax.plot([0, 0], [0, bar_width], color='black', lw=1)  # left
    ax.plot([num_line, num_line],[0, bar_width], color='black', lw=2)  # right

    # y轴标签
    ax.set_yticks(np.arange(-2, num_character + 4) + 0.4) # 刻度在柱子中间
    ax.set_yticklabels([''] * 2 + ['Name'] + [''] + ['Speaker'] + [''] + [''] * (num_character - 1) + ['Face'], fontsize=20)

    # 轴长
    ax.set_xlim(0, num_line)
    ax.set_ylim(-2, num_character + 4 + 0.1)

    # 隐藏轴线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)  # 四周
    ax.set_xticks([])  # 横坐标刻度
    ax.yaxis.set_ticks_position('none')  # 纵坐标刻度线
        
    if not y_label: ax.set_yticks([]) # 纵坐标刻度
    ax.grid(False)

def plot_all(face_data, speaker_data, name_data, colors_dict, save_path, num_cols = 10, figsize=(30, 12), fontsize=20):
    max_num_line = max(episode.shape[0] for episode in face_data)  # 单集最长台词
    num_episode = len(face_data)  # 总集数
    num_rows = int(num_episode/num_cols) + 1  # 行数

    # fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)
    # 创建图形和子图网格
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(num_rows, num_cols)
    axes = np.zeros((num_rows, num_cols), dtype=object)

    for i in range(num_rows):
        for j in range(num_cols):
            # 使用gs[i, j]来直接创建子图
            axes[i, j] = plt.subplot(gs[i, j])  # 指定每个子图的位置
            episode_idx = i * num_cols + j
            
            if episode_idx >= num_episode: 
                axes[i, j].axis('off')  # 关闭多余子图的显示
                continue
            # else:
            # 	print(episode_idx)
                
            if episode_idx % num_cols == 0:
                y_label = True
            else:
                y_label = None
            
            if j == int(num_cols/2):
                axes[i, j].set_title(f"Episode {i * num_cols + 1} - {i * num_cols + num_cols}", fontsize=fontsize, fontweight='bold')  # 每个子图设置标题
        
            plot_one_block(
                ax = axes[i, j],
                face_episdode = face_data[episode_idx], 
                speaker_episode = speaker_data[episode_idx], 
                name_episode = name_data[episode_idx], 
                colors_dict = colors_dict, 
                max_num_line = max_num_line, 
                bar_width = 0.8, 
                y_label = y_label
            )
            # axes[i, j].set_title(f"Episode {i * num_cols + j}")  # 每个子图设置标题
    
    # 图例
    handles = []
    labels = []
    name_list = list(colors_dict.keys())
    for i in range(len(name_list)):
        handles.append(mpatches.Patch(color=colors_list[i], alpha=0.5))
        labels.append(name_list[i])
    fig.legend(handles, labels, loc='lower right', fontsize=20, title="Characters", title_fontsize=fontsize, bbox_to_anchor=(1.0, 0.0), frameon=True, facecolor='white') # frameon=True to enable the frame
    plt.subplots_adjust(wspace=0)  
    plt.savefig(save_path, bbox_inches='tight')
	

TV_name = "I love my family"  # "the big bang theory" or "I love my family"
exp_name_general = "exp_video"
exp_name_specific = "2. 仅微调dense layer，从hid_feat构建数据集/exp27.7/round9"

# Load useful variables and pseudo labels
pseudo_label_dir = f"/data/home/scv7387/run/tv_series_plus/3D-Speaker/egs/3dspeaker/speaker-diarization/runs/{TV_name}/{exp_name_general}/result/self_supervised/{exp_name_specific}/pseudo_label"

## Load lengths_episode and audio_seg_ids from useful_var_dic.pkl
useful_var_path = os.path.join(pseudo_label_dir, 'useful_var_dic.pkl')
assert os.path.exists(useful_var_path), f"When from_preds is True, useful_var_dic.pkl must exist in {pseudo_label_dir}."
with open(useful_var_path, 'rb') as f:
    useful_var_dic = pickle.load(f)
lengths_episode = useful_var_dic['alengths']
audio_seg_ids = useful_var_dic['audio_seg_ids']

## Load pseudo labels for audio and face modalities
### audio
pseudo_label_files_audio = [f for f in os.listdir(pseudo_label_dir) if f.endswith('.json') and 'pseudo_labels_audio' in f]
assert len(pseudo_label_files_audio) == 1, "There should be exactly one pseudo_labels_audio file."
with open(os.path.join(pseudo_label_dir, pseudo_label_files_audio[0]), 'r') as f:
    pseudo_labels_audio = json.load(f)
### face
pseudo_label_files_face = [f for f in os.listdir(pseudo_label_dir) if f.endswith('.json') and 'pseudo_labels_faces_mid_frame_train' in f]
assert len(pseudo_label_files_face) == 1, "There should be exactly one pseudo_labels_faces_mid_frame_train file."
with open(os.path.join(pseudo_label_dir, pseudo_label_files_face[0]), 'r') as f:
    pseudo_labels_faces = json.load(f)

# Load name entity occurrence results
if TV_name == 'the big bang theory':
    name_dic = name_dic_BB
    txt_folder = f"/data/home/scv7387/run/tv_series_plus/dataset/{TV_name}/speaker_text_cn"
elif TV_name == 'I love my family':
    name_dic = name_dic_IL
    txt_folder = f"/data/home/scv7387/run/tv_series_plus/dataset/{TV_name}/speaker_text"
else:
    raise ValueError("TV_name should be 'the big bang theory' or 'I love my family'.")
name_occur_dict = find_name_in_texts(name_dic, txt_folder, save_path=f"/data/home/scv7387/run/tv_series_plus/dataset/{TV_name}/name2audio_ids_dic.json")


# align audio cluster ids with names
## 暂时使用人工映射，重新跑实验之后要revisit。后续把人物别名和spk id标识的台词本一起发给llm，通过上下文理解进行自动映射
spk2name_map_BB_main = {0: 'Sheldon', 1: 'Leonard', 2: 'Penny', 3: 'Howard', 4: 'Rajesh'}
spk2name_map_IL_main = {0: 'Fulao', 1: 'Zhixin', 2: 'Heping', 3: 'Yuanyuan', 5: 'Xiaozhang', 6: 'Zhiguo', 7: 'Xiaofan', 8: 'Yanhong'}
if TV_name == 'the big bang theory':
    spk2name_map_main = spk2name_map_BB_main
elif TV_name == 'I love my family':
    spk2name_map_main = spk2name_map_IL_main
else:
    raise ValueError("TV_name should be 'the big bang theory' or 'I love my family'.")
spk2name_map = {spk_id: spk2name_map_main.get(spk_id, 'Others') for spk_id in np.unique(list(pseudo_labels_audio.values()))}

# Get face_label, name_flags, speaker_label of shape (n_segments, n_actors)
n_segments = len(audio_seg_ids)
n_actors = len(name_dic)
names2idx_dic = {name: idx for idx, name in enumerate(name_dic.keys())}

name_flags = np.zeros((n_segments, n_actors), dtype=int)
speaker_label = np.zeros((n_segments, n_actors), dtype=int)
face_label = np.zeros((n_segments, n_actors), dtype=int)

for i, audio_seg_id in enumerate(audio_seg_ids):
    # name_flags
    for name, audio_id_list in name_occur_dict.items():
        if audio_seg_id in audio_id_list:
            name_idx = names2idx_dic[name]
            name_flags[i, name_idx] = 1
    
    # speaker_label
    spk_id = pseudo_labels_audio[audio_seg_id]
    spk_name = spk2name_map[spk_id]
    spk_name_idx = names2idx_dic[spk_name]
    speaker_label[i, spk_name_idx] = 1

    # face_label
    face_ids_valid = [k for k in pseudo_labels_faces.keys() if k.startswith(f"{audio_seg_id}_")]
    if len(face_ids_valid) > 0:
        for face_id in face_ids_valid:
            face_name = spk2name_map[pseudo_labels_faces[face_id]]
            face_name_idx = names2idx_dic[face_name]
            face_label[i, face_name_idx] = 1
# when multiple names appear in one line, randomly select one for visualization
one_hot_name_flags = np.copy(name_flags)
for i in range(len(one_hot_name_flags)):
	line = one_hot_name_flags[i]
	one_indices = np.where(line==1)[0]
	if len(one_indices) > 0:
		one_index = random.choice(one_indices)
		new_line = np.zeros_like(line)
		new_line[one_index] = 1
		one_hot_name_flags[i] = new_line

# Devide data into episodes
episode_num = len(lengths_episode)
one_hot_name_flags_list = devide_episode(one_hot_name_flags, lengths_episode)
speaker_label_list = devide_episode(speaker_label, lengths_episode)
face_label_list = devide_episode(face_label, lengths_episode)


# Define color maps
colors_list = [colors.to_hex(cm.coolwarm((n_actors - 1-i) / (n_actors - 1))) for i in range(n_actors)]
# cm.coolwarm 生成从暖色到冷色的渐变
colors_dict = dict(zip(list(name_dic.keys()), colors_list))
colors_dict['Others'] = 'gray'
colors_list = list(colors_dict.values())
# Create legend handles
save_dir = f"/data/home/scv7387/run/tv_series_plus/3D-Speaker/docs/figs/occur_tracks/{TV_name}"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"occur_track.png")
if TV_name == 'the big bang theory':
    plot_all(face_label_list, speaker_label_list, one_hot_name_flags_list, 
        colors_dict, save_path, num_cols = 9, figsize=(27, 14), fontsize=20)
else:
    plot_all(face_label_list, speaker_label_list, one_hot_name_flags_list, 
        colors_dict, save_path, num_cols = 10, figsize=(30, 24), fontsize=25)