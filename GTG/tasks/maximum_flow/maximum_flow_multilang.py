"""
maximum_flow任务 - 支持多语言版本 (中英文)
"""

from GTG.tasks.maximum_flow.maximum_flow import (
    generate_a_sample,
    question_generation,
    answer_and_inference_steps_generation,
    generate_maximum_flow_graph,
    choices_generation,
)
from GTG.utils.utils import NID, graph_with_egde_weight_to_str, graph_with_egde_weight_to_adj_str, graph_with_edge_weight_to_natural_language
from GTG.utils.language import get_task_templates, graph_to_natural_language_multilang, LANG_EN, LANG_ZH
import numpy as np


TASK_NAME = 'maximum_flow'


def make_sample_multilang(task_name, g, ques_str, ans_str, steps_str=None, choi_str=None, label_str=None, lang=LANG_EN, **kwargs):
    """创建样本 (使用多语言图描述)"""
    from GTG.utils.utils import NID, graph_to_edge_list_str, graph_to_adj_str

    sample = {
        'task': task_name,
        'graph': kwargs.get('g_str', graph_to_edge_list_str(g)),
        'graph_adj': kwargs.get('g_adj_str', graph_to_adj_str(g)),
        'graph_nl': kwargs.get('g_adj_nl', graph_to_natural_language_multilang(g, lang)),
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
    """生成maximum_flow样本 (支持多语言)"""
    g, start_node, end_node = generate_maximum_flow_graph(config)
    # 原始question_generation需要start_node和end_node
    ques_str = question_generation(config, start_node, end_node)
    # 原始answer_and_inference_steps_generation返回3个值
    steps_str, ans, ans_str = answer_and_inference_steps_generation(config, g, start_node, end_node)

    while ans == 0:
        g, start_node, end_node = generate_maximum_flow_graph(config)
        ques_str = question_generation(config, start_node, end_node)
        steps_str, ans, ans_str = answer_and_inference_steps_generation(config, g, start_node, end_node)

    # 原始choices_generation没有g和ques参数
    choi_str, label_str = choices_generation(config, ans)

    sample = make_sample_multilang(
        TASK_NAME, g, ques_str, ans_str, steps_str, choi_str, label_str, lang,
        g_str=graph_with_egde_weight_to_str(g),
        g_adj_str=graph_with_egde_weight_to_adj_str(g),
        g_adj_nl=graph_with_edge_weight_to_natural_language(g)
    )
    return sample


if __name__ == '__main__':
    test_config = {'num_nodes_range': '[8, 12]'}
    sample = generate_a_sample_multilang(test_config, lang='zh')
    print(f"Task: {sample['task']}, Language: {sample['language']}")
    print(f"Question: {sample['question']}")
