from gensim.models import Word2Vec
import os
import jieba.posseg as pseg
import random
from jieba import analyse

import jieba


def load_txt_to_word_list(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            word_list = [word for word in content.split(' ') if word]
            return word_list
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在")
    except PermissionError:
        print(f"错误：没有权限打开文件 '{file_path}'")
    except UnicodeDecodeError:
        print(f"错误：文件 '{file_path}' 编码不是utf-8，请检查编码格式（如gbk）")
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")
    return []


def print_random_paragraph(file_path, separator='\n'):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        paragraphs = [p.strip() for p in content.split(separator) if p.strip()]
        if not paragraphs:
            print("文件中未找到有效段落")
            return
        random_para = random.choice(paragraphs)
        print("随机选取的段落：\n")
        print(random_para)
        return random_para

    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在")
    except UnicodeDecodeError:
        print(f"错误：文件编码错误，请尝试修改encoding参数（如gbk）")
    except Exception as e:
        print(f"处理出错：{str(e)}")
    return None


def step1():
    sentences = load_txt_to_word_list(r"data/wiki.zh.txt")
    sentence = random.choice(sentences)

    print("精确模式:", list(jieba.cut(sentence, cut_all=False)))
    print("全模式:", list(jieba.cut(sentence, cut_all=True)))
    print("搜索引擎模式:", list(jieba.cut_for_search(sentence)))
    print("HMM=True:", list(jieba.cut(sentence, HMM=True)))
    print("HMM=False:", list(jieba.cut(sentence, HMM=False)))


def step2():
    sentences = print_random_paragraph(r"data/wiki.zh.txt")
    tfidf_keywords = analyse.extract_tags(sentences, topK=10, withWeight=False)
    print("TF-IDF 关键词:", tfidf_keywords)
    textrank_keywords = analyse.textrank(sentences, topK=10, withWeight=False)
    print("TextRank 关键词:", textrank_keywords)


def step2_v1():
    stopwords = set(
        ['的', '中', '在', '是', '或', '及', '与', '请', '例如', '因为', '许多', '上', '过程', '经验', '目的', '这个'])

    def extract_with_stopwords(text, stopwords):
        words = jieba.lcut(text)
        filtered = [w for w in words if w not in stopwords and len(w) > 1]
        return ' '.join(filtered)

    text = print_random_paragraph(r"data/wiki.zh.txt")
    cleaned_text = extract_with_stopwords(text, stopwords)
    tfidf = analyse.extract_tags(cleaned_text, topK=10, withWeight=False)
    textrank = analyse.textrank(cleaned_text, topK=10, withWeight=False)
    print("TF-IDF（去停用词）:", tfidf)
    print("TextRank（去停用词）:", textrank)


def step2_v2():
    def extract_nouns(text):
        words = pseg.cut(text)
        nouns = [word for word, flag in words if flag.startswith('n') and len(word) > 1]  # n = 名词
        return ' '.join(nouns)

    text = print_random_paragraph(r"data/wiki.zh.txt")
    noun_text = extract_nouns(text)
    tfidf_noun = analyse.extract_tags(noun_text, topK=10)
    textrank_noun = analyse.textrank(noun_text, topK=10)
    print("仅名词 - TF-IDF:", tfidf_noun)
    print("仅名词 - TextRank:", textrank_noun)


def train_and_compare_word2vec(target_words,corpus_path='data/wiki.zh.txt', min_count=3,vector_size=100):
    print("正在读取并分词语料...")
    sentences = []
    if not os.path.exists(corpus_path):
        print(f"错误：未找到语料文件 {corpus_path}，请确认路径正确。")
        return
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                words = list(jieba.cut(line))
                words = [w for w in words if len(w) > 1]
                if len(words) > 3:
                    sentences.append(words)

    print(f"共加载 {len(sentences)} 条句子用于训练。")
    print("\n【训练 CBOW 模型 (sg=0)】...")
    model_cbow = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=5,
        min_count=min_count,
        workers=4,
        sg=0,  # CBOW
        epochs=5
    )
    print("\n【训练 Skip-gram 模型 (sg=1)】...")
    model_sg = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=5,
        min_count=3,
        workers=4,
        sg=1,  # Skip-gram
        epochs=5
    )
    for target_word in target_words:
        print(f"\n=== 目标词: '{target_word}' ===")
        if target_word in model_cbow.wv:
            cbow_sims = model_cbow.wv.most_similar(target_word, topn=5)
            print("\nCBOW (sg=0) 相似词:")
            for word, score in cbow_sims:
                print(f"  {word}: {score:.4f}")
        else:
            print(f"\nCBOW: '{target_word}' 不在词汇表中。")

        if target_word in model_sg.wv:
            sg_sims = model_sg.wv.most_similar(target_word, topn=5)
            print("\nSkip-gram (sg=1) 相似词:")
            for word, score in sg_sims:
                print(f"  {word}: {score:.4f}")
        else:
            print(f"\nSkip-gram: '{target_word}' 不在词汇表中。")


def step3():
    train_and_compare_word2vec(['开源','软件','系统'],min_count=2,vector_size=100)


if __name__ == "__main__":
    step3()
