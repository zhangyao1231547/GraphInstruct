"""
bipartite任务 - 支持多语言版本 (中英文)
"""

from GTG.tasks.bipartite.bipartite import (
    generate_a_sample,
    question_generation,
    answer_and_inference_steps_generation,
)
from GTG.utils.utils import NID, generate_bipartite_graph
from GTG.utils.language import get_task_templates, graph_to_natural_language_multilang, LANG_EN, LANG_ZH


TASK_NAME = 'bipartite'


def question_generation_multilang(config, g, n1, n2, lang=LANG_EN):
    """生成bipartite问题 (支持多语言)"""
    templates = get_task_templates(TASK_NAME, lang)
    ques_str = templates['question']
    return ques_str


def answer_and_inference_steps_generation_multilang(config, g, n1, n2, lang=LANG_EN):
    """生成bipartite推理步骤和答案 (支持多语言)"""
    templates = get_task_templates(TASK_NAME, lang)
    steps_str_en, ans, ans_str = answer_and_inference_steps_generation(config, g, n1, n2)
    steps_str = templates['steps_start'] + templates['result']
    return steps_str, ans, ans_str


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
    """生成bipartite样本 (支持多语言)"""
    g, n1, n2 = generate_bipartite_graph(config)

    ques_str = question_generation_multilang(config, g, n1, n2, lang)
    steps_str, ans, ans_str = answer_and_inference_steps_generation_multilang(config, g, n1, n2, lang)

    sample = make_sample_multilang(
        TASK_NAME, g, ques_str, ans_str, steps_str, None, None, lang
    )
    sample['n1'] = str(n1)
    sample['n2'] = str(n2)
    return sample


if __name__ == '__main__':
    test_config = {'num_nodes_range': '[8, 12]'}
    sample = generate_a_sample_multilang(test_config, lang='zh')
    print(f"Task: {sample['task']}, Language: {sample['language']}")
    print(f"Question: {sample['question']}")
