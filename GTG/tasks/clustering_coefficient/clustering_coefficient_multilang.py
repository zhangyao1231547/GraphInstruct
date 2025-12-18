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
    ques, ques_str = question_generation(config, g)
    # 原始函数需要额外的directed参数
    steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g, ques, g.is_directed())

    while reject:
        g = graph_generation(config)
        ques, ques_str = question_generation(config, g)
        steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g, ques, g.is_directed())

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
