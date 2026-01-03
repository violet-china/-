import jieba
import jieba.analyse  # 可选：用于关键词提取

# -------------------------- 配置项 --------------------------
# 停用词列表（可根据需求扩展，比如添加行业专属停用词）
STOP_WORDS = {
    '的', '了', '是', '我', '你', '他', '她', '它', '们', '在', '和', '有', '就',
    '不', '也', '都', '而', '及', '与', '之', '于', '着', '过', '呢', '吗', '吧',
    '啊', '哦', '嗯', '这', '那', '此', '彼', '其', '所', '把', '被', '为', '因',
    '以', '对', '对于', '关于', '通过', '随着', '按照', '基于', '个', '只', '条'
}


# -------------------------- 核心分词函数 --------------------------
def jieba_tokenize(text, mode='default', remove_stopwords=True):
    """
    jieba分词核心函数
    :param text: 待分词的文本（长/短均可）
    :param mode: 分词模式
                 - default: 精准模式（默认，适合常规分词）
                 - full: 全模式（列出所有可能的分词结果，会有冗余）
                 - search: 搜索引擎模式（精准模式基础上，对长词再次切分）
    :param remove_stopwords: 是否去除停用词（默认True）
    :return: 分词结果列表
    """
    # 空文本处理
    if not text or text.strip() == '':
        return []

    # 不同模式的分词逻辑
    if mode == 'full':
        words = jieba.lcut(text, cut_all=True)  # 全模式，返回列表
    elif mode == 'search':
        words = jieba.lcut_for_search(text)  # 搜索引擎模式
    else:
        words = jieba.lcut(text)  # 精准模式（默认）

    # 去除停用词和空白字符
    if remove_stopwords:
        words = [word.strip() for word in words if word.strip() and word not in STOP_WORDS]

    return words


# -------------------------- 示例：关键词提取（可选） --------------------------
def extract_keywords(text, top_k=5, with_weight=False):
    """
    基于TF-IDF提取文本关键词（适合长文本）
    :param text: 待提取关键词的文本
    :param top_k: 提取前k个关键词（默认5）
    :param with_weight: 是否返回关键词权重（默认False）
    :return: 关键词列表（或(关键词, 权重)元组列表）
    """
    if not text or text.strip() == '':
        return []

    # 使用jieba的TF-IDF关键词提取
    keywords = jieba.analyse.extract_tags(
        text,
        topK=top_k,
        withWeight=with_weight,
        allowPOS=()  # 可指定词性，比如allowPOS=('n','v')只提取名词、动词
    )
    return keywords


# -------------------------- 测试示例 --------------------------
if __name__ == '__main__':
    # 测试文本（包含短句和长文本）
    short_text = "我喜欢用Python进行自然语言处理"
    long_text = """
    自然语言处理（NLP）是人工智能领域的一个重要分支，它致力于使计算机能够理解、解释和生成人类语言。
    jieba是Python中最常用的中文分词库，支持精准模式、全模式和搜索引擎模式，广泛应用于文本分析、关键词提取等场景。
    无论是短文本（如一句话）还是长文本（如一篇文章），jieba都能高效完成分词任务。
    """

    print("=" * 50, "短文本分词", "=" * 50)
    # 精准模式（默认）
    short_words_default = jieba_tokenize(short_text)
    print("精准模式（去停用词）：", short_words_default)

    # 全模式
    short_words_full = jieba_tokenize(short_text, mode='full')
    print("全模式（去停用词）：", short_words_full)

    # 搜索引擎模式
    short_words_search = jieba_tokenize(short_text, mode='search')
    print("搜索引擎模式（去停用词）：", short_words_search)

    print("\n" + "=" * 50, "长文本分词", "=" * 50)
    long_words = jieba_tokenize(long_text)
    print("长文本分词结果（前20个）：", long_words[:20])

    print("\n" + "=" * 50, "长文本关键词提取", "=" * 50)
    keywords = extract_keywords(long_text, top_k=5, with_weight=True)
    for idx, (word, weight) in enumerate(keywords, 1):
        print(f"第{idx}个关键词：{word}（权重：{weight:.4f}）")