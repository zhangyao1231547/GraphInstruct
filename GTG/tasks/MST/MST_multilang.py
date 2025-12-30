"""
MST任务 - 支持多语言版本 (中英文)
"""

from GTG.tasks.MST.MST import (
    generate_a_sample,
    question_generation,
    answer_and_inference_steps_generation,
    choices_generation,
)
from GTG.utils.utils import NID, graph_generation_with_edge_weight, graph_with_egde_weight_to_str, graph_with_egde_weight_to_adj_str
from GTG.utils.language import get_task_templates, graph_to_natural_language_multilang, LANG_EN, LANG_ZH
import numpy as np


TASK_NAME = 'MST'


def question_generation_multilang(config, g, lang=LANG_EN):
    """生成MST问题 (支持多语言)"""
    templates = get_task_templates(TASK_NAME, lang)
    ques_str = templates['question']
    return ques_str


def answer_and_inference_steps_generation_multilang(config, g, lang=LANG_EN):
    """生成MST推理步骤和答案 (支持多语言)"""
    import networkx as nx
    templates = get_task_templates(TASK_NAME, lang)

    if not nx.is_connected(g):
        return None, None, None, None, True

    steps_str = templates['steps_start']
    ans = 0
    MST = nx.minimum_spanning_tree(g)
    for edge in MST.edges(data=True):
        ans += edge[2]['weight']

    ans_str1 = str(list(MST.edges(data=True)))
    ans_str2 = str(ans)
    steps_str += templates['result']

    return steps_str, ans, ans_str1, ans_str2, False


def make_sample_multilang(task_name, g, ques_str, ans_str, steps_str=None, choi_str=None, label_str=None, lang=LANG_EN, **kwargs):
    """创建样本 (使用多语言图描述)"""
    from GTG.utils.utils import NID, graph_to_edge_list_str, graph_to_adj_str

    sample = {
        'task': task_name,
        'graph': kwargs.get('g_str', graph_to_edge_list_str(g)),
        'graph_adj': kwargs.get('g_adj_str', graph_to_adj_str(g)),
        'graph_nl': graph_to_natural_language_multilang(g, lang),
        'nodes': NID(np.arange(g.number_of_nodes())),
        'num_nodes': g.number_of_nodes(),
        'num_edges': g.number_of_edges(),
        'directed': g.is_directed(),
        'question': ques_str,
        'answer': ans_str,
        'language': lang,
    }
    if steps_str is not None:
        sample['steps'] = steps_str
    if choi_str is not None:
        sample['choices'] = choi_str
    if label_str is not None:
        sample['label'] = label_str

    return sample


def generate_a_sample_multilang(config, lang=LANG_EN):
    """生成MST样本 (支持多语言)"""
    g = graph_generation_with_edge_weight(config, False)
    ques_str = question_generation_multilang(config, g, lang)
    steps_str, ans, ans_str1, ans_str2, reject = answer_and_inference_steps_generation_multilang(config, g, lang)

    while reject:
        g = graph_generation_with_edge_weight(config, False)
        ques_str = question_generation_multilang(config, g, lang)
        steps_str, ans, ans_str1, ans_str2, reject = answer_and_inference_steps_generation_multilang(config, g, lang)

    choi_str, label_str = choices_generation(config, g, ans)

    sample = make_sample_multilang(
        TASK_NAME, g, ques_str, ans_str2, steps_str, choi_str, label_str, lang,
        g_str=graph_with_egde_weight_to_str(g),
        g_adj_str=graph_with_egde_weight_to_adj_str(g)
    )
    sample['answer_tree'] = ans_str1
    return sample


if __name__ == '__main__':
    test_config = {'num_nodes_range': '[8, 12]'}
    sample = generate_a_sample_multilang(test_config, lang='zh')
    print(f"Task: {sample['task']}, Language: {sample['language']}")
    print(f"Question: {sample['question']}")
