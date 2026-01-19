## BB
### face
Original cluster ids and their counts on labeled data: {0: 230, 1: 236, 2: 90, 3: 120, 4: 106, 5: 2, 7: 11, 8: 4, 9: 6, -1: 264}
Renamed cluster ids and their counts on labeled data: {0: 230, 1: 236, 2: 90, 3: 120, 4: 106, 5: 2, 6: 11, 7: 4, 8: 6, 9: 264}
Character name to index mapping: {'Howard': 0, 'Leonard': 1, 'Others': 2, 'Penny': 3, 'Raj': 4, 'Sheldon': 5}
Cluster_id to character_name_id mapping: {3: 0, 1: 1, 6: 2, 2: 3, 4: 4, 0: 5, 5: 2, 7: 2, 8: 2, 9: 2}
Cluster_id to speaker_id(main) mapping: {0: 'Sheldon', 1: 'Leonard', 2: 'Penny', 3: 'Howard', 4: 'Raj'}
### speaker
Original cluster ids and their counts on labeled data: {0: 635, 1: 491, 2: 318, 3: 201, 4: 140, 5: 23, 6: 30, 7: 34, 8: 35, 9: 24, -1: 69}
Renamed cluster ids and their counts on labeled data: {0: 635, 1: 491, 2: 318, 3: 201, 4: 140, 5: 23, 6: 30, 7: 34, 8: 35, 9: 24, 10: 69}
Speaker to index mapping: {'Howard': 0, 'Leonard': 1, 'Others': 2, 'Penny': 3, 'Raj': 4, 'Sheldon': 5}
Cluster_id to speaker_id mapping: {3: 0, 1: 1, 7: 2, 2: 3, 4: 4, 0: 5, 5: 2, 6: 2, 8: 2, 9: 2, 10: 2}
Cluster_id to speaker_id(main) mapping: {0: 'Sheldon', 1: 'Leonard', 2: 'Penny', 3: 'Howard', 4: 'Raj'}

Count matrix of name entity and speaker cluster id co-occurance:
            0  1  2  3  4  5  6  7  8  9  speaker cluster id
'Leonard'[[50  9 52  9  6  2  3  5  3  1]
'Sheldon' [18 70 47 26 13  5  4  4  7  2]
'Penny'   [40 62  6  4  7  5  0  0  0  0]
'Howard'  [12 23 19 13  5  1  1  1  2  0]
'Raj'     [ 7 18 15  7  3  0  0  0  0  1]
'Others'  [61 60 24 28 25  3  2  3  4 11]]
name

## IL
### face
Original cluster ids and their counts on labeled data: {0: 143, 1: 125, 2: 154, 3: 53, 4: 17, 5: 38, 6: 133, 7: 17, 8: 20, 9: 13, -1: 314}
Renamed cluster ids and their counts on labeled data: {0: 143, 1: 125, 2: 154, 3: 53, 4: 17, 5: 38, 6: 133, 7: 17, 8: 20, 9: 13, 10: 314}
Character name to index mapping: {'Others': 0, '傅老': 1, '和平': 2, '圆圆': 3, '小凡': 4, '小张': 5, '志国': 6, '志新': 7, '燕红': 8}
Cluster_id to character_name_id mapping: {4: 0, 0: 1, 2: 2, 3: 3, 7: 4, 5: 5, 6: 6, 1: 7, 8: 8, 9: 0, 10: 0}
Cluster_id to character_name_id(main) mapping: {0: '傅老', 1: '志新', 2: '和平', 3: '圆圆', 5: '小张', 6: '志国', 7: '小凡', 8: '燕红'}
### speaker
Original cluster ids and their counts on labeled data: {0: 385, 1: 340, 2: 327, 3: 116, 4: 56, 5: 64, 6: 246, 7: 49, 8: 65, 9: 58, -1: 294}
Renamed cluster ids and their counts on labeled data: {0: 385, 1: 340, 2: 327, 3: 116, 4: 56, 5: 64, 6: 246, 7: 49, 8: 65, 9: 58, 10: 294}
Speaker to index mapping: {'Others': 0, '傅老': 1, '和平': 2, '圆圆': 3, '小凡': 4, '小张': 5, '志国': 6, '志新': 7, '燕红': 8}
Cluster_id to speaker_id mapping: {9: 0, 0: 1, 2: 2, 3: 3, 7: 4, 5: 5, 6: 6, 1: 7, 8: 8, 4: 0, 10: 0}
Cluster_id to character_name_id(main) mapping: {0: '傅老', 1: '志新', 2: '和平', 3: '圆圆', 5: '小张', 6: '志国', 7: '小凡', 8: '燕红'}

Count matrix of name entity and speaker cluster id co-occurance:
               0   1   2   3   4   5   6   7   8   9  speaker cluster id
'Fulao'    [[ 13   6   5   4  32   0   2   1   2   2]
'Heping'    [ 82  11  13   2   4   2  53   0   0   1]
'Zhixin'    [ 70  16  46   2   4  13  22   4  23   0]
'Zhiguo'    [ 39   9  45   5   6   2  18   2   2   0]
'Yuanyuan'  [ 30  28  48   8   2  10  30  17   4   0]
'Xiaofan'   [ 21  13   8   1   0   5   9   3   1   0]
'Xiaozhang' [ 46  32  31  21   2   6  24   5   1   2]
'Yanhong'   [  1  20  15   4   4   2  20   0   5   0]
'Others'    [ 94  77 100  40  26  29  66  19  13  10]]

经过各种尝试，从共现信息直接对应spk id和人名不靠谱，主要是难以区分次要人物和边缘主要人物，以及主要角色提及自己的次数虽然较少，但也往往不是最少。后续考虑把别名信息、spk id标识的台词本一起发给LLM，通过上下文理解来做对应。