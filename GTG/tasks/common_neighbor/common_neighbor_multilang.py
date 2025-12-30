"""
common_neighbor任务 - 支持多语言版本 (中英文)
这是 common_neighbor.py 的多语言增强版本

Usage:
    from GTG.tasks.common_neighbor.common_neighbor_multilang import generate_a_sample_multilang

    # 生成英文样本 (默认)
    sample = generate_a_sample_multilang(config)

    # 生成中文样本
    sample = generate_a_sample_multilang(config, lang='zh')
"""

from GTG.tasks.common_neighbor.common_neighbor import (
    generate_a_sample,
    question_generation,
    answer_and_inference_steps_generation,
    choices_generation,
)
from GTG.utils.utils import NID, graph_generation
from GTG.utils.language import get_task_templates, graph_to_natural_language_multilang, LANG_EN, LANG_ZH


TASK_NAME = 'common_neighbor'


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
    """生成common_neighbor问题 (支持多语言)"""
    import random
    templates = get_task_templates(TASK_NAME, lang)
    num_nodes = g.number_of_nodes()
    u = random.randint(0, num_nodes - 1)
    v = random.randint(0, num_nodes - 1)
    while u == v:
        v = random.randint(0, num_nodes - 1)
    ques = {'u': u, 'v': v}
    ques_str = templates['question'].format(NID(u), NID(v))
    return ques, ques_str


def answer_and_inference_steps_generation_multilang(config, g, ques, lang=LANG_EN):
    """生成common_neighbor推理步骤和答案 (支持多语言)"""
    templates = get_task_templates(TASK_NAME, lang)
    steps_str_en, ans, ans_str, reject = answer_and_inference_steps_generation(config, g, ques)
    steps_str = templates['steps_start'] + templates['result']
    return steps_str, ans, ans_str, reject


def generate_a_sample_multilang(config, lang=LANG_EN):
    """
    生成common_neighbor样本 (支持多语言)

    Args:
        config: 配置字典
        lang: 语言代码, 'en' 或 'zh'

    Returns:
        样本字典
    """
    g = graph_generation(config)

    # 尝试使用多语言版本，如果没有则使用原始版本
    try:
        ques, ques_str = question_generation_multilang(config, g, lang)
        steps_str, ans, ans_str, reject = answer_and_inference_steps_generation_multilang(config, g, ques, lang)
    except:
        # 回退到原始英文版本
        ques, ques_str = question_generation(config, g)
        steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g, ques)

    while reject:
        g = graph_generation(config)
        try:
            ques, ques_str = question_generation_multilang(config, g, lang)
            steps_str, ans, ans_str, reject = answer_and_inference_steps_generation_multilang(config, g, ques, lang)
        except:
            ques, ques_str = question_generation(config, g)
            steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g, ques)

    choi_str, label_str = choices_generation(config, g, ques, ans)

    sample = make_sample_multilang(
        TASK_NAME, g, ques_str, ans_str, steps_str, choi_str, label_str, lang
    )
    return sample


# 保持向后兼容
def generate_a_sample_compat(config):
    """生成英文样本 (保持向后兼容)"""
    return generate_a_sample_multilang(config, lang=LANG_EN)


if __name__ == '__main__':
    # 测试
    test_config = {
        'num_nodes_range': '[8, 12]',
    }

    print("=" * 60)
    print(f"Testing English common_neighbor sample generation:")
    print("=" * 60)
    sample_en = generate_a_sample_multilang(test_config, lang='en')
    print(f"Task: {sample_en['task']}")
    print(f"Language: {sample_en['language']}")
    print(f"Question: {sample_en['question']}")
    print(f"Answer: {sample_en['answer']}")

    print("\n" + "=" * 60)
    print(f"Testing Chinese common_neighbor sample generation:")
    print("=" * 60)
    sample_zh = generate_a_sample_multilang(test_config, lang='zh')
    print(f"Task: {sample_zh['task']}")
    print(f"Language: {sample_zh['language']}")
    print(f"Question: {sample_zh['question']}")
    print(f"Answer: {sample_zh['answer']}")
