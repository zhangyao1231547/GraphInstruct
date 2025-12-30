#!/usr/bin/env python3
"""
修复所有multilang任务文件
根据各任务原始实现的函数签名进行修复
"""

import os

TASKS_DIR = '/nvme0/work/workspaces-zy/GraphInstruct/GTG/tasks'

# ============ topological_sort_multilang.py ============
# 原始: answer_and_inference_steps_generation 返回3个值 (steps_str, ans, ans_str)
topological_sort_multilang = '''"""
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
    # 原始函数返回3个值，没有reject
    steps_str, ans, ans_str = answer_and_inference_steps_generation(config, g, ques)
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
'''

# ============ bipartite_multilang.py ============
# 原始: question_generation(config, g, n1, n2) - 需要额外参数
# 原始: answer_and_inference_steps_generation(config, g, n1, n2) - 返回3个值
bipartite_multilang = '''"""
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

    ques_str = question_generation(config, g, n1, n2)
    steps_str, ans, ans_str = answer_and_inference_steps_generation(config, g, n1, n2)

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
'''

# ============ clustering_coefficient_multilang.py ============
# 原始: answer_and_inference_steps_generation(config, g, ques, directed) - 需要额外参数directed
clustering_coefficient_multilang = '''"""
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
'''

# ============ connected_component_multilang.py ============
# 原始: question_generation返回 (ques_str, start_node)
# 原始: answer_and_inference_steps_generation(g, start_node) - 返回3个值，签名不同
connected_component_multilang = '''"""
connected_component任务 - 支持多语言版本 (中英文)
"""

from GTG.tasks.connected_component.connected_component import (
    generate_a_sample,
    question_generation,
    answer_and_inference_steps_generation,
)
from GTG.utils.utils import NID, graph_generation, make_sample
from GTG.utils.language import get_task_templates, graph_to_natural_language_multilang, LANG_EN, LANG_ZH
import numpy as np


TASK_NAME = 'connected_component'


def make_sample_multilang(task_name, g, ques_str, ans_str, steps_str=None, choi_str=None, label_str=None, lang=LANG_EN):
    """创建样本 (使用多语言图描述)"""
    from GTG.utils.utils import NID, graph_to_edge_list_str, graph_to_adj_str

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
    """生成connected_component样本 (支持多语言)"""
    g = graph_generation(config)
    # 原始question_generation返回 (ques_str, start_node)
    ques_str, start_node = question_generation(config, g)
    # 原始answer_and_inference_steps_generation(g, start_node) 返回3个值
    steps_str, ans, ans_str = answer_and_inference_steps_generation(g, start_node)

    sample = make_sample_multilang(
        TASK_NAME, g, ques_str, ans_str, steps_str, None, None, lang
    )
    sample['cc_node_ratio'] = len(ans) / g.number_of_nodes()
    return sample


if __name__ == '__main__':
    test_config = {'num_nodes_range': '[8, 12]'}
    sample = generate_a_sample_multilang(test_config, lang='zh')
    print(f"Task: {sample['task']}, Language: {sample['language']}")
    print(f"Question: {sample['question']}")
'''

# ============ diameter_multilang.py ============
# 原始: question_generation(config, g) 返回单个值 ques_str
# 原始: answer_and_inference_steps_generation(config, g) 返回4个值
# 原始: choices_generation(config, g, ans) - 没有ques参数
diameter_multilang = '''"""
diameter任务 - 支持多语言版本 (中英文)
"""

from GTG.tasks.diameter.diameter import (
    generate_a_sample,
    question_generation,
    answer_and_inference_steps_generation,
    choices_generation,
)
from GTG.utils.utils import NID, graph_generation
from GTG.utils.language import get_task_templates, graph_to_natural_language_multilang, LANG_EN, LANG_ZH
import numpy as np


TASK_NAME = 'diameter'


def make_sample_multilang(task_name, g, ques_str, ans_str, steps_str=None, choi_str=None, label_str=None, lang=LANG_EN):
    """创建样本 (使用多语言图描述)"""
    from GTG.utils.utils import NID, graph_to_edge_list_str, graph_to_adj_str

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
    """生成diameter样本 (支持多语言)"""
    g = graph_generation(config, False)  # directed=False
    # 原始question_generation返回单个值
    ques_str = question_generation(config, g)
    # 原始answer_and_inference_steps_generation返回4个值
    steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g)

    while reject:
        g = graph_generation(config, False)
        ques_str = question_generation(config, g)
        steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g)

    # 原始choices_generation没有ques参数
    choi_str, label_str = choices_generation(config, g, ans)

    sample = make_sample_multilang(
        TASK_NAME, g, ques_str, ans_str, steps_str, choi_str, label_str, lang
    )
    return sample


if __name__ == '__main__':
    test_config = {'num_nodes_range': '[8, 12]'}
    sample = generate_a_sample_multilang(test_config, lang='zh')
    print(f"Task: {sample['task']}, Language: {sample['language']}")
    print(f"Question: {sample['question']}")
'''

# ============ maximum_flow_multilang.py ============
# 原始: 使用自定义的 generate_maximum_flow_graph
# 原始: question_generation(config, start_node, end_node) - 需要额外参数
maximum_flow_multilang = '''"""
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
'''

# ============ MST_multilang.py ============
# 原始: question_generation(config, g) 返回单个值
# 原始: answer_and_inference_steps_generation(config, g) 返回5个值
MST_multilang = '''"""
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
    g = graph_generation_with_edge_weight(config, False)  # directed=False
    # 原始question_generation返回单个值
    ques_str = question_generation(config, g)
    # 原始answer_and_inference_steps_generation返回5个值
    steps_str, ans, ans_str1, ans_str2, reject = answer_and_inference_steps_generation(config, g)

    while reject:
        g = graph_generation_with_edge_weight(config, False)
        ques_str = question_generation(config, g)
        steps_str, ans, ans_str1, ans_str2, reject = answer_and_inference_steps_generation(config, g)

    # 原始choices_generation没有ques参数
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
'''

# ============ shortest_path_multilang.py ============
# 使用带权图 graph_generation_with_edge_weight
shortest_path_multilang = '''"""
shortest_path任务 - 支持多语言版本 (中英文)
"""

from GTG.tasks.shortest_path.shortest_path import (
    generate_a_sample,
    question_generation,
    answer_and_inference_steps_generation,
    choices_generation,
)
from GTG.utils.utils import NID, graph_generation_with_edge_weight, graph_with_egde_weight_to_str, graph_with_egde_weight_to_adj_str, graph_with_edge_weight_to_natural_language
from GTG.utils.language import get_task_templates, graph_to_natural_language_multilang, LANG_EN, LANG_ZH
import numpy as np


TASK_NAME = 'shortest_path'


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
    """生成shortest_path样本 (支持多语言)"""
    # 使用带权图
    g = graph_generation_with_edge_weight(config)
    ques, ques_str = question_generation(config, g)
    steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g, ques)

    while reject:
        g = graph_generation_with_edge_weight(config)
        ques, ques_str = question_generation(config, g)
        steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g, ques)

    choi_str, label_str = choices_generation(config, g, ques, ans)

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
'''

# ============ predecessor_multilang.py ============
# 需要使用有向图 graph_generation(config, directed=True)
predecessor_multilang = '''"""
predecessor任务 - 支持多语言版本 (中英文)
"""

from GTG.tasks.predecessor.predecessor import (
    generate_a_sample,
    question_generation,
    answer_and_inference_steps_generation,
    choices_generation,
)
from GTG.utils.utils import NID, graph_generation
from GTG.utils.language import get_task_templates, graph_to_natural_language_multilang, LANG_EN, LANG_ZH
import numpy as np


TASK_NAME = 'predecessor'


def make_sample_multilang(task_name, g, ques_str, ans_str, steps_str=None, choi_str=None, label_str=None, lang=LANG_EN):
    """创建样本 (使用多语言图描述)"""
    from GTG.utils.utils import NID, graph_to_edge_list_str, graph_to_adj_str

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
    """生成predecessor样本 (支持多语言)"""
    # 必须使用有向图，因为predecessor需要调用g.predecessors()
    g = graph_generation(config, directed=True)
    ques, ques_str = question_generation(config, g)
    steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g, ques)

    while reject:
        g = graph_generation(config, directed=True)
        ques, ques_str = question_generation(config, g)
        steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g, ques)

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
'''


# 写入文件
files_to_create = {
    'topological_sort': topological_sort_multilang,
    'bipartite': bipartite_multilang,
    'clustering_coefficient': clustering_coefficient_multilang,
    'connected_component': connected_component_multilang,
    'diameter': diameter_multilang,
    'maximum_flow': maximum_flow_multilang,
    'MST': MST_multilang,
    'shortest_path': shortest_path_multilang,
    'predecessor': predecessor_multilang,
}


def main():
    print("=" * 60)
    print("Fixing multilang task files")
    print("=" * 60)

    for task_name, content in files_to_create.items():
        filepath = os.path.join(TASKS_DIR, task_name, f'{task_name}_multilang.py')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  [FIXED] {task_name}_multilang.py")

    print("\n" + "=" * 60)
    print(f"Fixed {len(files_to_create)} multilang files")
    print("=" * 60)


if __name__ == '__main__':
    main()
