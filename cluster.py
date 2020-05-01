#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
from matplotlib import pyplot as plt
import jieba
data = pd.read_csv('E:/taidibei/data/fujian2.csv',encoding = 'utf8')
data.head()


# In[32]:


# %load F:/githubclass/HotNewsAnalysis/hot_news_analysis/utils/news_pandas.py

import pandas as pd
import re
import pickle

''''
def save_news(news_df, path):
    """保存新闻"""
    news_df.to_csv(path, index=False, encoding='utf-8')
'''
'''
def replace_line_terminator(x):
    """替换行终止符"""
    try:
        x = re.sub(r'\r\n', '\n', x)
    except TypeError:
        pass
    return x
'''

def load_news(path):
    """加载新闻"""
    news_df = pd.read_csv(path, encoding='utf-8')
    news_df = news_df.applymap(replace_line_terminator)
    return news_df


def save_text(document, path):
    """保存txt文件"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(document)


def load_text(path):
    """加载txt文件"""
    with open(path, 'r', encoding='utf-8') as f:
        document = f.read()
    return document


def save_element(element, path):
    """保存元素"""
    with open(path, 'wb') as f:
        pickle.dump(element, f)


def load_element(path):
    """加载元素"""
    with open(path, 'rb') as f:
        element = pickle.load(f)
    return element


# In[28]:


# %load F:/githubclass/HotNewsAnalysis/hot_news_analysis/utils/ py

import re
import jieba
import jieba.posseg as pseg
import json
from datetime import datetime
from datetime import timedelta

'''
def data_filter(df):
    """数据过滤"""
    # 过滤掉没有内容的新闻
    df = df[df['content'] != ''].copy()
    df = df.dropna(subset=['content']).copy()
    # 去重
    df = df.drop_duplicates(subset=['url'])
    df = df.drop_duplicates(subset=['title'])
    df = df.reset_index(drop=True)
    return df
'''

def get_data(df, last_time, delta):
    """
    获取某段时间的新闻数据
    :param df: 原始数据
    :param last_time: 指定要获取数据的最后时间
    :param delta: 时间间隔
    :return: last_time前timedelta的数据
    """
    last_time = datetime.strptime(last_time, '%Y/%m/%d %H:%M:%S')
    delta = timedelta(delta)
    try:
        df['留言时间'] = df['留言时间'].map(lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M:%S'))
    except TypeError:
        pass
    df = df[df['留言时间'].map(lambda x: (x <= last_time) and (x > last_time - delta))].copy()
    print('df.shape=', df.shape)
    if df.shape[0] == 0:
        print('No Data!')
        return df
    df = df.sort_values(by=['留言时间'], ascending=[0])
    df['留言时间'] = df['留言时间'].map(lambda x: datetime.strftime(x, '%Y/%m/%d %H:%M:%S'))
    df = df.reset_index(drop=True)
    return df

'''
def clean_title_blank(title):
    """清理新闻标题空白"""
    # 清理未知字符
    title = re.sub(r'\?+', ' ', title)
    # 清理空白字符
    title = re.sub(r'\u3000', '', title)
    title = title.strip()
    title = re.sub(r'\s+', ' ', title)
    title = re.sub(r'([|：])+ ', r'\1', title)
    title = re.sub(r' ([|：])+', r'\1', title)
    return title
'''

'''
def clean_content_blank(content):
    """清理新闻内容空白"""
    # 清理未知字符
    content = re.sub(r'\?+', ' ', content)
    # 清理空白字符
    content = re.sub(r'\u3000', '', content)
    content = content.strip()
    content = re.sub(r'[ \t\r\f]+', ' ', content)
    content = re.sub(r'\n ', '\n', content)
    content = re.sub(r' \n', '\n', content)
    content = re.sub(r'\n+', '\n', content)
    return content
'''
'''
def clean_content(content):
    """清理新闻内容"""
    # 清理新闻内容空白
    content = clean_content_blank(content)
    # 英文大写转小写
    content = content.lower()
    # 清理超链接
    content = re.sub(r'https?://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', content)
    # 清理责任编辑等
    content = re.split(r'\n责任编辑', content)[0]
    content = re.split(r'返回搜狐，查看更多', content)[0]
    # 清理原标题
    content = re.sub(r'原标题：.*\n', '', content)
    # 清理来源等和内容无关的文字
    texts = [
        r'新浪财经讯[ ，]*', r'新浪美股讯[ ，]*', r'新浪外汇讯[ ，]*', r'新浪科技讯[ ，]*',
        r'[（\(].{,10}来源[:：].{,30}[）\)]',
        r'(?<=\n).{,2}来源[:：].{,30}\n', r'(?<=\n).{,2}来源[:：].{,30}$',
        r'[（\(].{,20}记者[ :：].{,20}[）\)]',
        r'(?<=\n).{,2}作者：.{,20}\n', r'(?<=\n).{,2}作者：.{,20}$',
        r'(?<=\n).{,2}编辑：.{,20}\n', r'(?<=\n).{,2}编辑：.{,20}$'
    ]
    for text in texts:
        content = re.sub(text, '', content)
    content = re.sub(r'\n+', '\n', content)
    return content
'''

def get_num_en_ch(text):
    """提取数字英文中文"""
    text = re.sub(r'[^0-9A-Za-z\u4E00-\u9FFF]+', ' ', text)
    text = text.strip()
    return text


def pseg_cut(text, userdict_path=None):
    """
    词性标注
    :param text: string，原文本数据
    :param userdict_path: string，用户词词典路径，默认为None
    :return: list， 分词后词性标注的列表
    """
    if userdict_path is not None:
        jieba.load_userdict(userdict_path)
    words = pseg.lcut(text)
    return words


def get_words_by_flags(words, flags=None):
    """
    获取指定词性的词
    :param words: list， 分词后词性标注的列表
    :param flags: list， 词性标注，默认为提取名词和动词
    :return: list， 指定词性的词
    """
    flags = ['n.*', 'v.*'] if flags is None else flags
    words = [w for w, f in words if w != ' ' and re.match('|'.join(['(%s$)' % flag for flag in flags]), f)]
    return words


def userdict_cut(text, userdict_path=None):
    """
    对文本进行jieba分词
    如果使用用户词词典，那么使用用户词词典进行jieba分词
    """
    if userdict_path is not None:
        jieba.load_userdict(userdict_path)
    words = jieba.cut(text)
    return words


def stop_words_cut(words, stop_words_path):
    """停用词处理"""
    with open(stop_words_path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
        stopwords.append(' ')
        words = [word for word in words if word not in stopwords]
    return words


def disambiguation_cut(words, disambiguation_dict_path):
    """消歧词典"""
    with open(disambiguation_dict_path, 'r', encoding='utf-8') as f:
        disambiguation_dict = json.load(f)
        words = [(disambiguation_dict[word]
                  if disambiguation_dict.get(word) else word) for word in words]
    return words


def individual_character_cut(words, individual_character_dict_path):
    """删除无用单字"""
    with open(individual_character_dict_path, 'r', encoding='utf-8') as f:
        individual_character = [line.strip() for line in f.readlines()]
        words = [word for word in words
                 if ((len(word) > 1) or ((len(word) == 1) and (word in individual_character)))]
    return words


def document2txt(raw_document, userdict_path, text_path):
    """文本分词并保存为txt文件"""
    document = clean_content_blank(raw_document)
    document = document.lower()
    document_cut = userdict_cut(document, userdict_path)
    result = ' '.join(document_cut)
    result = re.sub(r' +', ' ', result)
    result = re.sub(r' \n ', '\n', result)
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(result)


# In[29]:


# %load F:/githubclass/HotNewsAnalysis/hot_news_analysis/utils/counter.py

from collections import Counter


def flat(l):
    """平展多维列表"""
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)


def get_word_library(list1):
    """
    获得词库
    :param list1: 一维或多维词列表
    :return: list，所有词去重之后的列表
    """
    list2 = flat(list1)
    list3 = list(set(list2))
    return list3


def get_single_frequency_words(list1):
    """
    获得单频词列表
    :param list1: 一维或多维词列表
    :return: list，所有只出现一次的词组成的列表
    """
    list2 = flat(list1)
    cnt = Counter(list2)
    list3 = [i for i in cnt if cnt[i] == 1]
    return list3


def get_most_common_words(list1, top_n=None, min_frequency=1):
    """
    获取最常见的词组成的列表
    :param list1: 一维或多维词列表
    :param top_n: 指定最常见的前n个词，默认为None
    :param min_frequency: 指定最小频数，默认为1
    :return: list，最常见的前n个词组成的列表
    """
    list2 = flat(list1)
    cnt = Counter(list2)
    list3 = [i[0] for i in cnt.most_common(top_n) if cnt[i[0]] >= min_frequency]
    return list3


def get_num_of_value_no_repeat(list1):
    """
    获取列表中不重复的值的个数
    :param list1: 列表
    :return: int，列表中不重复的值的个数
    """
    num = len(set(list1))
    return num


# In[43]:


# %load F:/githubclass/HotNewsAnalysis/hot_news_analysis/utils/modeling.py

import pandas as pd
import pickle
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from wordcloud import WordCloud
from textrank4zh import TextRank4Sentence
from gensim.models import word2vec


def feature_extraction(series, vectorizer='CountVectorizer', vec_args=None):
    """
    对原文本进行特征提取
    :param series: pd.Series，原文本
    :param vectorizer: string，矢量化器，如'CountVectorizer'或者'TfidfVectorizer'
    :param vec_args: dict，矢量化器参数
    :return: 稀疏矩阵
    """
    vec_args = {'max_df': 1.0, 'min_df': 1} if vec_args is None else vec_args
    vec_args_list = ['%s=%s' % (i[0],
                                "'%s'" % i[1] if isinstance(i[1], str) else i[1]
                                ) for i in vec_args.items()]
    vec_args_str = ','.join(vec_args_list)
    vectorizer1 = eval("%s(%s)" % (vectorizer, vec_args_str))
    matrix = vectorizer1.fit_transform(series)
    return matrix


def get_cluster(matrix, cluster='DBSCAN', cluster_args=None):
    """
    对数据进行聚类，获取训练好的聚类器
    :param matrix: 稀疏矩阵
    :param cluster: string，聚类器
    :param cluster_args: dict，聚类器参数
    :return: 训练好的聚类器
    """
    cluster_args = {'eps': 0.5, 'min_samples': 5, 'metric': 'cosine'} if cluster_args is None else cluster_args
    cluster_args_list = ['%s=%s' % (i[0],
                                    "'%s'" % i[1] if isinstance(i[1], str) else i[1]
                                    ) for i in cluster_args.items()]
    cluster_args_str = ','.join(cluster_args_list)
    cluster1 = eval("%s(%s)" % (cluster, cluster_args_str))
    cluster1 = cluster1.fit(matrix)
    return cluster1


def get_labels(cluster):
    """
    获取聚类标签
    :param cluster: 训练好的聚类器
    :return: list，聚类标签
    """
    labels = cluster.labels_
    return labels


def label2rank(labels_list):
    """
    按标签的数量将标签转换为排行
    :param labels_list: list，聚类标签
    :return: list，聚类排行
    """
    series = pd.Series(labels_list)
    list1 = series[series != -1].tolist()
    n = len(set(list1))
    cnt = Counter(list1)
    key = [cnt.most_common()[i][0] for i in range(n)]
    value = [i for i in range(1, n + 1)]
    my_dict = dict(zip(key, value))
    my_dict[-1] = -1
    rank_list = [my_dict[i] for i in labels_list]
    return rank_list


def get_non_outliers_data(df, label_column='label'):
    """获取属于某个聚类簇的数据"""
    df = df[df[label_column] != -1].copy()
    return df


def get_data_sort_labelnum(df, label_column='label', top=1):
    """
    获取按标签数量排行的第top组数据
    :param df: pd.DataFrame，带有标签列的数据
    :param label_column: string，标签列名
    :param top: int
    :return: pd.DataFrame，前top组的数据
    """
    assert top > 0, 'top不能小于等于0！'
    labels = df[label_column].tolist()
    cnt = Counter(labels)
    label = cnt.most_common()[top - 1][0] if top <= len(set(labels)) else -2
    df = df[df[label_column] == label].copy() if label != -2 else pd.DataFrame(columns=df.columns)
    return df


def list2wordcloud(list1, save_path, font_path):
    """
    将文本做成词云
    :param list1: list，文本列表
    :param save_path: string，词云图片保存的路径
    :param font_path: string，用于制作词云所需的字体路径
    """
    text = ' '.join(list1)
    wc = WordCloud(font_path=font_path, width=800, height=600, margin=2,
                   ranks_only=True, max_words=200, collocations=False).generate(text)
    wc.to_file(save_path)


def get_key_sentences(text, num=1):
    """
    利用textrank算法，获取文本摘要
    :param text: string，原文本
    :param num: int，指定摘要条数
    :return: string，文本摘要
    """
    tr4s = TextRank4Sentence(delimiters='\n')
    tr4s.analyze(text=text, lower=True, source='all_filters')
    abstract = '\n'.join([item.sentence for item in tr4s.get_key_sentences(num=num)])
    return abstract


def feature_reduction(matrix, pca_n_components=50, tsne_n_components=2):
    """降维"""
    data_pca = PCA(n_components=pca_n_components).fit_transform(matrix) if pca_n_components is not None else matrix
    data_pca_tsne = TSNE(n_components=tsne_n_components).fit_transform(
        data_pca) if tsne_n_components is not None else data_pca
    print('data_pca_tsne.shape=', data_pca_tsne.shape)
    return data_pca_tsne


def get_word2vec_model(text_path):
    """训练词向量模型"""
    sentences = word2vec.LineSentence(text_path)
    model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, size=100)
    return model


def get_wordvec(model, word):
    """查询词是否在词库中"""
    try:
        model.wv.get_vector(word)
        return True
    except:
        return False


def get_word_and_wordvec(model, words):
    """获取输入词的词和对应的词向量"""
    word_list = [i for i in words if get_wordvec(model, i)]
    wordvec_list = [model.wv[i].tolist() for i in words if get_wordvec(model, i)]
    return word_list, wordvec_list


def get_top_words(words, label, label_num):
    """获得每个类中的前30个词"""
    df = pd.DataFrame()
    df['word'] = words
    df['label'] = label
    for i in range(label_num):
        df_ = df[df['label'] == i]
        print(df_['word'][:30])


def save_model(model, model_path):
    """保存模型"""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(model_path):
    """加载模型"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


# In[47]:


# %load F:/githubclass/HotNewsAnalysis/hot_news_analysis/hot_news.py

import os
import pandas as pd
from datetime import datetime

import threading

# 获取项目路径
#current_folder_path = os.path.dirname(os.path.realpath(__file__))
# 获取数据存放目录路径
data_path = os.path.join('E:/taidibei/', 'data')
fonts_path = os.path.join(data_path, 'fonts')
images_path = os.path.join(data_path, 'images')
texts_path = os.path.join(data_path, 'texts')
extra_dict_path = os.path.join(data_path, 'extra_dict')
models_path = os.path.join(data_path, 'models')
news_path = os.path.join(data_path, 'news')
temp_news_path = os.path.join(data_path, 'temp_news')
results_path = os.path.join(data_path, 'results')

'''
def my_crawler():
    """爬取新闻数据"""
    # sina_news_df =  get_latest_news('sina', top=1000, show_content=True)
    # sohu_news_df =  get_latest_news('sohu', top=1000, show_content=True)
    # xinhuanet_news_df =  get_latest_news('xinhuanet', top=100, show_content=True)
    #  save_news(sina_news_df, os.path.join(news_path, 'sina_latest_news.csv'))
    #  save_news(sohu_news_df, os.path.join(news_path, 'sohu_latest_news.csv'))
    #  save_news(xinhuanet_news_df, os.path.join(news_path, 'xinhuanet_latest_news.csv'))
    save_file_path = os.path.join(news_path, 'news_df.csv')
    thread_crawler.threaded_crawler(1000, 1000, 10, save_file_path=save_file_path)
'''

def load_data():
    """加载数据"""
    # sina_news_df =  load_news(os.path.join(news_path, 'sample_sina_latest_news.csv'))
    # sohu_news_df =  load_news(os.path.join(news_path, 'sample_sohu_latest_news.csv'))
    # xinhuanet_news_df =  load_news(os.path.join(news_path, 'sample_xinhuanet_latest_news.csv'))
    # sina_news_df =  load_news(os.path.join(news_path, 'sina_latest_news.csv'))
    # sohu_news_df =  load_news(os.path.join(news_path, 'sohu_latest_news.csv'))
    # xinhuanet_news_df =  load_news(os.path.join(news_path, 'xinhuanet_latest_news.csv'))
    # news_df = pd.concat([sina_news_df, sohu_news_df, xinhuanet_news_df], ignore_index=True)
    save_file_path = os.path.join(news_path, 'fujian2.csv')
    news_df = load_news(save_file_path)
    return news_df


def filter_data(news_df):
    """过滤数据"""
    #df = data_filter(news_df)
    df = news_df
    now_time = datetime.strftime(datetime.now(), '%Y/%m/%d %H:%M:%S')
    # now_time = '2020-05-01 23:59'
    df = get_data(df, last_time=now_time, delta=5)
    return df


def title_preprocess(df_title):
    """标题分词处理"""
    df_title['title_'] = df_title['留言主题'].map(lambda x:  clean_title_blank(x))
    df_title['title_'] = df_title['title_'].map(lambda x:  get_num_en_ch(x))
    df_title['title_cut'] = df_title['title_'].map(lambda x:  pseg_cut(
        x, userdict_path=os.path.join(extra_dict_path, 'self_userdict.txt')))
    df_title['title_cut'] = df_title['title_cut'].map(lambda x:  get_words_by_flags(
        x, flags=['n.*', '.*n', 'v.*', 's', 'j', 'l', 'i', 'eng']))
    df_title['title_cut'] = df_title['title_cut'].map(lambda x:  stop_words_cut(
        x, os.path.join(extra_dict_path, 'self_stop_words.txt')))
    df_title['title_cut'] = df_title['title_cut'].map(lambda x:  disambiguation_cut(
        x, os.path.join(extra_dict_path, 'self_disambiguation_dict.json')))
    df_title['title_cut'] = df_title['title_cut'].map(lambda x:  individual_character_cut(
        x, os.path.join(extra_dict_path, 'self_individual_character_dict.txt')))
    df_title['title_'] = df_title['title_cut'].map(lambda x: ' '.join(x))
    return df_title

def save_news(news_df, path):
    """保存新闻"""
    news_df.to_csv(path, index=False, encoding='utf-8')
    

def title_cluster(df, save_df=False):
    """按新闻标题聚类"""
    df_title = df.copy()
    df_title = title_preprocess(df_title)
    word_library_list = get_word_library(df_title['title_cut'])
    single_frequency_words_list = get_single_frequency_words(df_title['title_cut'])
    max_features = len(word_library_list) - len(single_frequency_words_list) // 2
    title_matrix = feature_extraction(df_title['title_'], vectorizer='CountVectorizer',
                                               vec_args={'max_df': 1.0, 'min_df': 1, 'max_features': max_features})
    title_dbscan = get_cluster(title_matrix, cluster='DBSCAN',
                                        cluster_args={'eps': 0.4, 'min_samples': 4, 'metric': 'cosine'})
    title_labels = get_labels(title_dbscan)
    df_title['title_label'] = title_labels
    df_non_outliers = get_non_outliers_data(df_title, label_column='title_label')
    title_label_num = get_num_of_value_no_repeat(df_non_outliers['title_label'].tolist())
    print('按新闻标题聚类，一共有%d个簇(不包括离群点)' % title_label_num)
    title_rank = label2rank(title_labels)
    df_title['title_rank'] = title_rank
    for i in range(1, title_label_num + 1):
        df_ = df_title[df_title['title_rank'] == i]
        title_top_list = get_most_common_words(df_['title_cut'], top_n=10)
        print(title_top_list)
    if save_df:
        df_title.drop(['留言详情', 'title_', 'title_label'], axis=1, inplace=True)
        save_news(df_title, os.path.join(results_path, 'df_title_rank.csv'))
    return df_title


def content_preprocess(df_content):
    """新闻内容分词处理"""
    df_content['content_'] = df_content['留言详情'].map(lambda x:  clean_content(x))
    df_content['content_'] = df_content['content_'].map(lambda x:  get_num_en_ch(x))
    df_content['content_cut'] = df_content['content_'].map(lambda x:  pseg_cut(
        x, userdict_path=os.path.join(extra_dict_path, 'self_userdict.txt')))
    df_content['content_cut'] = df_content['content_cut'].map(lambda x:  get_words_by_flags(
        x, flags=['n.*', '.*n', 'v.*', 's', 'j', 'l', 'i', 'eng']))
    df_content['content_cut'] = df_content['content_cut'].map(lambda x:  stop_words_cut(
        x, os.path.join(extra_dict_path, 'self_stop_words.txt')))
    df_content['content_cut'] = df_content['content_cut'].map(lambda x:  disambiguation_cut(
        x, os.path.join(extra_dict_path, 'self_disambiguation_dict.json')))
    df_content['content_cut'] = df_content['content_cut'].map(lambda x:  individual_character_cut(
        x, os.path.join(extra_dict_path, 'self_individual_character_dict.txt')))
    df_content['content_'] = df_content['content_cut'].map(lambda x: ' '.join(x))
    return df_content


def content_cluster(df, df_save=False):
    """按新闻内容聚类"""
    df_content = df.copy()
    df_content = content_preprocess(df_content)
    word_library_list = get_word_library(df_content['content_cut'])
    single_frequency_words_list = get_single_frequency_words(df_content['content_cut'])
    max_features = len(word_library_list) - len(single_frequency_words_list) // 2
    content_matrix = feature_extraction(df_content['content_'], vectorizer='CountVectorizer',
                                                 vec_args={'max_df': 0.95, 'min_df': 1, 'max_features': max_features})
    content_dbscan = get_cluster(content_matrix, cluster='DBSCAN',
                                          cluster_args={'eps': 0.35, 'min_samples': 4, 'metric': 'cosine'})
    content_labels = get_labels(content_dbscan)
    df_content['content_label'] = content_labels
    df_non_outliers = get_non_outliers_data(df_content, label_column='content_label')
    content_label_num =  get_num_of_value_no_repeat(df_non_outliers['content_label'].tolist())
    print('按新闻内容聚类，一共有%d个簇(不包括离群点)' % content_label_num)
    content_rank =  label2rank(content_labels)
    df_content['content_rank'] = content_rank
    for i in range(1, content_label_num + 1):
        df_ = df_content[df_content['content_rank'] == i]
        content_top_list =  get_most_common_words(df_['content_cut'], top_n=15, min_frequency=1)
        print(content_top_list)
    if df_save:
        df_content.drop(['content_', 'content_label'], axis=1, inplace=True)
        save_news(df_content, os.path.join(results_path, 'df_content_rank.csv'))
    return df_content


def get_wordcloud(df, rank_column, text_list_column):
    """
    按照不同的簇生成每个簇的词云
    :param df: pd.DataFrame，带有排名和分词后的文本列表数据
    :param rank_column: 排名列名
    :param text_list_column: 分词后的文本列表列名
    """
    df_non_outliers =  get_non_outliers_data(df, label_column=rank_column)
    label_num =  get_num_of_value_no_repeat(df_non_outliers[rank_column].tolist())
    wordcloud_folder_path = os.path.join(results_path, rank_column)
    if not os.path.exists(wordcloud_folder_path):
        os.mkdir(wordcloud_folder_path)
    for i in range(1, label_num + 1):
        df_ = df[df[rank_column] == i]
        list_ =  flat(df_[text_list_column].tolist())
        list2wordcloud(list_, save_path=os.path.join(wordcloud_folder_path, '%d.png' % i),
                                font_path=os.path.join(fonts_path, 'simhei.ttf'))


def key_content(df, df_save=False):
    """获取摘要"""

    def f(text):
        text =  clean_content(text)
        text =  get_key_sentences(text, num=1)
        return text

    df['abstract'] = df['留言详情'].map(f)
    if df_save:
        df.drop(['留言详情'], axis=1, inplace=True)
        save_news(df, os.path.join(results_path, 'df_abstract.csv'))
    return df

def load_news(path):
    """加载新闻"""
    news_df = pd.read_csv(path, encoding='utf-8')
    #news_df = news_df.applymap(replace_line_terminator)
    return news_df

def get_key_words():
    df_title =  load_news(os.path.join(results_path, 'df_title_rank.csv'))
    df_content =  load_news(os.path.join(results_path, 'df_content_rank.csv'))
    df_title['title_cut'] = df_title['title_cut'].map(eval)
    df_content['content_cut'] = df_content['content_cut'].map(eval)
    get_wordcloud(df_content, 'content_rank', 'content_cut')
    df_title_content = df_title.copy()
    df_title_content['content_cut'] = df_content['content_cut']
    df_title_content['content_rank'] = df_content['content_rank']
    df_title_content =  get_non_outliers_data(df_title_content, label_column='title_rank')
    title_rank_num =  get_num_of_value_no_repeat((df_title_content['title_rank']))
    for i in range(1, title_rank_num + 1):
        df_i = df_title_content[df_title_content['title_rank'] == i]
        title = '\n'.join(df_i['留言主题'].tolist())
        title =  get_key_sentences(title, num=1)
        print('热点：', title)
        content_rank = [k for k in df_i['content_rank']]
        content_rank = set(content_rank)
        for j in content_rank:
            df_j = df_i[df_i['content_rank'] == j]
            most_commmon_words =  get_most_common_words(df_j['content_cut'], top_n=20, min_frequency=5)
            if len(most_commmon_words) > 0:
                print('相关词汇：', most_commmon_words)


def main():
    # # my_crawler()
    news_df = load_data()
    df = filter_data(news_df)
    # title_cluster(df, True)
    # content_cluster(df, True)
    t1 = threading.Thread(target=title_cluster, args=(df, True))
    t2 = threading.Thread(target=content_cluster, args=(df, True))
    t1.start()
    t2.start()
    threads = [t1, t2]
    for t in threads:
        t.join()
    get_key_words()


if __name__ == '__main__':
    main()


# In[37]:


data_path = os.path.join('E:/taidibei/', 'data')
results_path = os.path.join(data_path, 'results')

print(results_path)


# In[46]:





# In[ ]:




