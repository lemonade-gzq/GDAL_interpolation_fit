# 导入扩展库
import re  # 正则表达式库
import collections  # 词频统计库
import numpy as np  # numpy数据处理库
import jieba  # 结巴分词
import wordcloud  # 词云展示库
from PIL import Image  # 图像处理库
import matplotlib.pyplot as plt  # 图像展示库
import pandas as pd

# 读取文件
fn = open('E:\\城市与区域生态\\其他\\2023.10.17卧龙保护区成立60周年\\key word.txt', 'r', encoding='utf-8')
string_data = fn.read()  # 读出整个文件
fn.close()  # 关闭文件

# 文本预处理
pattern = re.compile(u'\t|\n|\.|-|:|;|\)|\(|\?|"')  # 定义正则表达式匹配模式
string_data = re.sub(pattern, '', string_data)  # 将符合模式的字符去除

# 文本分词
seg_list_exact = jieba.cut(string_data, cut_all=False)  # 精确模式分词
object_list = []
remove_words = [u'的', u'，', u'和', u'是', u'随着', u'对于', u'对', u'等', u'能', u'都', u'。', u' ', u'、', u'中', u'在', u'了',
                u'通常', u'如果', u'我们', u'需要', u',']  # 自定义去除词库

for word in seg_list_exact:  # 循环读出每个分词
    if word not in remove_words:  # 如果不在去除词库中
        object_list.append(word)  # 分词追加到列表

# 词频统计
word_counts = collections.Counter(object_list)  # 对分词做词频统计
word_counts_top10 = word_counts.most_common(10)  # 获取前10最高频的词
print(word_counts_top10)  # 输出检查

# 词频展示
mask = np.array(Image.open('E:\\城市与区域生态\\其他\\2023.10.17卧龙保护区成立60周年\\大熊猫透明.png'))
wc = wordcloud.WordCloud(
    font_path='C:/Windows/Fonts/simhei.ttf',  # 设置字体格式
    max_words=200,  # 最多显示词数
    max_font_size=100,  # 字体最大值
    width=800,
    height=600,
    # mask=mask
)

wc.generate_from_frequencies(word_counts)  # 从字典生成词云
image_colors = wordcloud.ImageColorGenerator(mask)  # 从背景图建立颜色方案
plt.imshow(wc)  # 显示词云
plt.axis('off')  # 关闭坐标轴
plt.show()  # 显示图像
# 创建 DataFrame
df = pd.DataFrame(word_counts.items(), columns=['词语', '出现次数'])

# 按照出现次数降序排序
df = df.sort_values(by='出现次数', ascending=False)

# 导出到 Excel
output_file = 'E:\\城市与区域生态\\其他\\2023.10.17卧龙保护区成立60周年\\file.xlsx'
df.to_excel(output_file, index=False)