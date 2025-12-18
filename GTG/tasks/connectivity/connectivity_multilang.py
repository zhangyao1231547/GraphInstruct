"""
connectivity任务 - 支持多语言版本 (中英文)
这是 connectivity.py 的多语言增强版本

Usage:
    from GTG.tasks.connectivity.connectivity_multilang import generate_a_sample_multilang

    # 生成英文样本 (默认)
    sample = generate_a_sample_multilang(config)

    # 生成中文样本
    sample = generate_a_sample_multilang(config, lang='zh')
"""

from GTG.tasks.connectivity.connectivity import (
    generate_a_sample,
    question_generation,
    answer_and_inference_steps_generation,
    choices_generation,
)
from GTG.utils.utils import NID, graph_generation
from GTG.utils.language import get_task_templates, graph_to_natural_language_multilang, LANG_EN, LANG_ZH


TASK_NAME = 'connectivity'


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
    """生成connectivity问题 (支持多语言)"""
    templates = get_task_templates(TASK_NAME, lang)

    # 使用原始函数获取问题参数
    ques, _ = question_generation(config, g)

    # 根据模板生成多语言问题
    if 'question' in templates and templates['question']:
        try:
            # 尝试使用模板格式化
            if task_name == 'BFS' or task_name == 'DFS':
                ques_str = templates['question'].format(NID(ques.get('start', 0)))
            elif task_name == 'degree' or task_name == 'neighbor' or task_name == 'clustering_coefficient':
                ques_str = templates['question'].format(NID(ques.get('u', 0)))
            elif task_name == 'connectivity' or task_name == 'edge' or task_name == 'common_neighbor' or task_name == 'jaccard':
                ques_str = templates['question'].format(NID(ques.get('u', 0)), NID(ques.get('v', 0)))
            elif task_name == 'shortest_path' or task_name == 'maximum_flow':
                ques_str = templates['question'].format(NID(ques.get('source', 0)), NID(ques.get('target', 0)))
            elif task_name == 'predecessor':
                ques_str = templates['question'].format(NID(ques.get('source', 0)), NID(ques.get('target', 0)))
            elif task_name == 'connected_component':
                ques_str = templates['question'].format(NID(ques.get('u', 0)))
            elif task_name == 'page_rank':
                ques_str = templates['question'].format(ques.get('damping', 0.85), ques.get('iterations', 10))
            else:
                # 对于没有参数的问题
                ques_str = templates['question']
        except:
            # 如果格式化失败，使用原始英文
            _, ques_str = question_generation(config, g)
    else:
        _, ques_str = question_generation(config, g)

    return ques, ques_str


def answer_and_inference_steps_generation_multilang(config, g, ques, lang=LANG_EN):
    """生成connectivity推理步骤和答案 (支持多语言)"""
    templates = get_task_templates(TASK_NAME, lang)

    # 使用原始函数获取答案
    steps_str_en, ans, ans_str, reject = answer_and_inference_steps_generation(config, g, ques)

    # 如果是中文，替换关键词
    if lang == LANG_ZH and 'steps_start' in templates:
        steps_str = templates.get('steps_start', steps_str_en)
        # 可以在这里添加更多的步骤翻译逻辑
        if 'result' in templates:
            steps_str += templates['result']
    else:
        steps_str = steps_str_en

    return steps_str, ans, ans_str, reject


def generate_a_sample_multilang(config, lang=LANG_EN):
    """
    生成connectivity样本 (支持多语言)

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
    print(f"Testing English connectivity sample generation:")
    print("=" * 60)
    sample_en = generate_a_sample_multilang(test_config, lang='en')
    print(f"Task: {sample_en['task']}")
    print(f"Language: {sample_en['language']}")
    print(f"Question: {sample_en['question']}")
    print(f"Answer: {sample_en['answer']}")

    print("\n" + "=" * 60)
    print(f"Testing Chinese connectivity sample generation:")
    print("=" * 60)
    sample_zh = generate_a_sample_multilang(test_config, lang='zh')
    print(f"Task: {sample_zh['task']}")
    print(f"Language: {sample_zh['language']}")
    print(f"Question: {sample_zh['question']}")
    print(f"Answer: {sample_zh['answer']}")
