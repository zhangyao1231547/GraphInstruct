"""
topological_sort任务 - 支持多语言版本 (中英文)
"""

from GTG.tasks.topological_sort.topological_sort import (
    generate_a_sample,
    question_generation,
    answer_and_inference_steps_generation,
    choices_generation,
)
from GTG.utils.utils import NID, generate_random_directed_acyclic_graph
from GTG.utils.language import get_task_templates, graph_to_natural_language_multilang, LANG_EN, LANG_ZH


TASK_NAME = 'topological_sort'


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


def question_generation_multilang(config, g, lang=LANG_EN):
    """生成topological_sort问题 (支持多语言)"""
    templates = get_task_templates(TASK_NAME, lang)
    ques, ques_str_en = question_generation(config, g)

    if lang == LANG_ZH and 'question' in templates and templates['question']:
        ques_str = templates['question']
    else:
        ques_str = ques_str_en

    return ques, ques_str


def answer_and_inference_steps_generation_multilang(config, g, ques, lang=LANG_EN):
    """生成topological_sort推理步骤和答案 (支持多语言)"""
    from collections import deque
    templates = get_task_templates(TASK_NAME, lang)

    steps_str = templates['steps_start']

    # 计算入度
    in_degree = {node: 0 for node in g}
    for node in g:
        for neighbor in g.neighbors(node):
            in_degree[neighbor] += 1

    # 使用队列执行拓扑排序
    queue = deque([node for node in g if in_degree[node] == 0])
    result = []

    while len(queue):
        steps_str += templates['zero_in_degree'].format(NID(list(queue)))
        node = queue.popleft()
        result.append(node)
        steps_str += templates['visit_node'].format(NID(node))

        for neighbor in g.neighbors(node):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    steps_str += templates['result']

    ans = result
    ans_str = NID(ans)
    reject = False

    return steps_str, ans, ans_str, reject


def generate_a_sample_multilang(config, lang=LANG_EN):
    """生成topological_sort样本 (支持多语言)"""
    g = generate_random_directed_acyclic_graph(config)

    ques, ques_str = question_generation_multilang(config, g, lang)
    steps_str, ans, ans_str, reject = answer_and_inference_steps_generation_multilang(config, g, ques, lang)

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
