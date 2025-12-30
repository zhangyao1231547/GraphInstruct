"""
clustering_coefficient任务 - 支持多语言版本 (中英文)
"""

from GTG.tasks.clustering_coefficient.clustering_coefficient import (
    generate_a_sample,
    question_generation,
    answer_and_inference_steps_generation,
    choices_generation,
)
from GTG.utils.utils import NID, graph_generation
from GTG.utils.language import get_task_templates, graph_to_natural_language_multilang, LANG_EN, LANG_ZH


TASK_NAME = 'clustering_coefficient'


def question_generation_multilang(config, g, lang=LANG_EN):
    """生成clustering_coefficient问题 (支持多语言)"""
    import random
    templates = get_task_templates(TASK_NAME, lang)

    num_nodes = g.number_of_nodes()
    node = random.randint(0, num_nodes - 1)

    ques = {'u': node}
    ques_str = templates['question'].format(NID(node))

    return ques, ques_str


def answer_and_inference_steps_generation_multilang(config, g, ques, directed, lang=LANG_EN):
    """生成clustering_coefficient推理步骤和答案 (支持多语言)"""
    templates = get_task_templates(TASK_NAME, lang)
    steps_str = templates['steps_start']

    u = ques['u']
    nei = list(g.neighbors(u))
    deg = len(nei)

    if deg <= 1:
        reject = True
        return None, None, None, reject

    edge_list = []
    nu = 0
    if g.is_directed():
        for i in nei:
            for j in nei:
                if g.has_edge(i, j):
                    nu += 1
                    edge_list.append((i, j))
        cc = nu / (deg * (deg - 1))
    else:
        for idx, i in enumerate(nei):
            for j in nei[idx + 1:]:
                if g.has_edge(i, j):
                    nu += 1
                    edge_list.append((i, j))
        cc = 2 * nu / (deg * (deg - 1))

    ans = cc
    ans_str = "{:.4f}".format(ans)
    steps_str += templates['result'].format(NID(ques['u']))

    reject = False
    if ans < 0.00001:
        reject = True

    return steps_str, ans, ans_str, reject


def make_sample_multilang(task_name, g, ques_str, ans_str, steps_str=None, choi_str=None, label_str=None, lang=LANG_EN):
    """创建样本 (使用多语言图描述)"""
    from GTG.utils.utils import NID, graph_to_edge_list_str, graph_to_adj_str
    import numpy as np

    sample = {
        'task': task_name,
        'graph': graph_to_edge_list_str(g),
        'graph_adj': graph_to_adj_str(g),
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
    """生成clustering_coefficient样本 (支持多语言)"""
    g = graph_generation(config)
    ques, ques_str = question_generation_multilang(config, g, lang)
    steps_str, ans, ans_str, reject = answer_and_inference_steps_generation_multilang(config, g, ques, g.is_directed(), lang)

    while reject:
        g = graph_generation(config)
        ques, ques_str = question_generation_multilang(config, g, lang)
        steps_str, ans, ans_str, reject = answer_and_inference_steps_generation_multilang(config, g, ques, g.is_directed(), lang)

    choi_str, label_str = choices_generation(config, g, ques, ans)

    sample = make_sample_multilang(
        TASK_NAME, g, ques_str, ans_str, steps_str, choi_str, label_str, lang
    )
    return sample


if __name__ == '__main__':
    test_config = {'num_nodes_range': '[8, 12]'}
    sample = generate_a_sample_multilang(test_config, lang='zh')
    print(f"Task: {sample['task']}, Language: {sample['language']}")
    print(f"Question: {sample['question']}")
