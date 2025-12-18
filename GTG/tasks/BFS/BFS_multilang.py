"""
BFS任务 - 支持多语言版本 (中英文)
这是 BFS.py 的多语言增强版本

Usage:
    from GTG.tasks.BFS.BFS_multilang import generate_a_sample_multilang

    # 生成英文样本 (默认)
    sample = generate_a_sample_multilang(config)

    # 生成中文样本
    sample = generate_a_sample_multilang(config, lang='zh')
"""

from GTG.utils.utils import NID, graph_generation, make_sample, edge_list_str_to_graph
from GTG.utils.language import get_task_templates, graph_to_natural_language_multilang, LANG_EN, LANG_ZH
from GTG.utils.evaluation import NodeListEvaluator

import networkx as nx
import random
import numpy as np
from collections import deque


TASK_NAME = 'BFS'


def question_generation_multilang(config, g, lang=LANG_EN):
    """生成BFS问题 (支持多语言)"""
    templates = get_task_templates(TASK_NAME, lang)

    v = random.randint(0, g.number_of_nodes() - 1)
    ques = {'start': v}

    ques_str = templates['question'].format(NID(ques['start']))

    return ques, ques_str


def answer_and_inference_steps_generation_multilang(config, g, ques, lang=LANG_EN):
    """生成BFS推理步骤和答案 (支持多语言)"""
    templates = get_task_templates(TASK_NAME, lang)

    steps_str = templates['steps_start']

    start = ques['start']
    visited = set()
    queue = deque([start])
    visited.add(start)
    traversal = []

    while len(queue):
        vertex = queue.popleft()
        steps_str += templates['visit_node'].format(NID(vertex))
        traversal.append(vertex)

        unvisited_nei = []
        for neighbor in g.neighbors(vertex):
            if neighbor not in visited:
                queue.append(neighbor)
                unvisited_nei.append(neighbor)
                visited.add(neighbor)

        if len(unvisited_nei) > 0:
            steps_str += templates['unvisited_neighbors'].format(
                NID(vertex), NID(unvisited_nei)
            )
        else:
            steps_str += "\n"

    ans = traversal
    ans_str = NID(traversal)
    steps_str += templates['result']

    if len(ans) < 5:
        reject = True
    else:
        reject = False

    return steps_str, ans, ans_str, reject


def choices_generation(config, g, ques, ans):
    """生成选择题选项 (保持原逻辑)"""
    false_ans = []

    x = np.array(ans[1:])
    np.random.shuffle(x)
    false_ans.append([ans[0]] + list(x))

    cut = len(ans) // 2
    x = np.array(ans[cut:])
    np.random.shuffle(x)
    false_ans.append(ans[:cut] + list(x))

    assert g.number_of_nodes() >= 5
    cut = len(ans) // 3
    false_ans.append(ans[:cut] + ans[-cut:] + ans[cut:-cut])

    _choi_list = [ans] + false_ans
    choi_list = []
    ind = np.arange(4)
    np.random.shuffle(ind)
    for i in ind:
        choi_list.append(NID(_choi_list[i]))
    label_str = str(np.arange(4)[ind == 0][0])
    choi_str = "[" + ", ".join(choi_list) + "]"

    return choi_str, label_str


def make_sample_multilang(task_name, g, ques_str, ans_str, steps_str=None, choi_str=None, label_str=None, lang=LANG_EN):
    """创建样本 (使用多语言图描述)"""
    from GTG.utils.utils import NID, graph_to_edge_list_str, graph_to_adj_str

    sample = {
        'task': task_name,
        'graph': graph_to_edge_list_str(g),
        'graph_adj': graph_to_adj_str(g),
        'graph_nl': graph_to_natural_language_multilang(g, lang),  # 使用多语言版本
        'nodes': NID(np.arange(g.number_of_nodes())),
        'num_nodes': g.number_of_nodes(),
        'num_edges': g.number_of_edges(),
        'directed': g.is_directed(),
        'question': ques_str,
        'answer': ans_str,
        'language': lang,  # 添加语言标识
    }
    if steps_str is not None:
        sample['steps'] = steps_str
    if choi_str is not None:
        sample['choices'] = choi_str,
    if label_str is not None:
        sample['label'] = label_str

    return sample


def generate_a_sample_multilang(config, lang=LANG_EN):
    """
    生成BFS样本 (支持多语言)

    Args:
        config: 配置字典
        lang: 语言代码, 'en' 或 'zh'

    Returns:
        样本字典
    """
    g = graph_generation(config)
    ques, ques_str = question_generation_multilang(config, g, lang)
    steps_str, ans, ans_str, reject = answer_and_inference_steps_generation_multilang(config, g, ques, lang)

    while reject:
        g = graph_generation(config)
        ques, ques_str = question_generation_multilang(config, g, lang)
        steps_str, ans, ans_str, reject = answer_and_inference_steps_generation_multilang(config, g, ques, lang)

    choi_str, label_str = choices_generation(config, g, ques, ans)

    sample = make_sample_multilang(
        TASK_NAME, g, ques_str, ans_str, steps_str, choi_str, label_str, lang
    )
    return sample


# 原始函数保持兼容性
def generate_a_sample(config):
    """生成英文样本 (保持向后兼容)"""
    return generate_a_sample_multilang(config, lang=LANG_EN)


# 测试代码
if __name__ == '__main__':
    # 测试配置
    test_config = {
        'num_nodes_range': '[8, 12]',
    }

    print("=" * 60)
    print("Testing English BFS sample generation:")
    print("=" * 60)
    sample_en = generate_a_sample_multilang(test_config, lang='en')
    print(f"Task: {sample_en['task']}")
    print(f"Language: {sample_en['language']}")
    print(f"Question: {sample_en['question']}")
    print(f"Graph (NL): {sample_en['graph_nl'][:200]}...")
    print(f"Steps: {sample_en['steps'][:300]}...")
    print(f"Answer: {sample_en['answer']}")

    print("\n" + "=" * 60)
    print("Testing Chinese BFS sample generation:")
    print("=" * 60)
    sample_zh = generate_a_sample_multilang(test_config, lang='zh')
    print(f"Task: {sample_zh['task']}")
    print(f"Language: {sample_zh['language']}")
    print(f"Question: {sample_zh['question']}")
    print(f"Graph (NL): {sample_zh['graph_nl'][:200]}...")
    print(f"Steps: {sample_zh['steps'][:300]}...")
    print(f"Answer: {sample_zh['answer']}")
