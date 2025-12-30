#!/usr/bin/env python3
"""
批量修复multilang任务文件的模板问题
"""

import os
import re

# 各任务的参数配置
TASK_CONFIGS = {
    'neighbor': {
        'params': ['u'],
        'question_format': 'templates["question"].format(NID(node))',
        'result_format': 'templates["result"].format(NID(ques["u"]))',
        'ques_dict': '{"u": node}',
    },
    'page_rank': {
        'params': ['damping', 'iterations'],
        'question_format': 'templates["question"].format(0.85, 10)',
        'result_format': 'templates["result"]',
        'ques_dict': '{"damping": 0.85, "iterations": 10}',
    },
    'connectivity': {
        'params': ['u', 'v'],
        'question_format': 'templates["question"].format(NID(u), NID(v))',
        'result_format': 'templates.get("result_yes", templates["yes"]) if ans else templates.get("result_no", templates["no"])',
        'ques_dict': '{"u": u, "v": v}',
        'extra_setup': '''    u = random.randint(0, num_nodes - 1)
    v = random.randint(0, num_nodes - 1)
    while u == v:
        v = random.randint(0, num_nodes - 1)''',
    },
    'cycle': {
        'params': [],
        'question_format': 'templates["question"]',
        'result_format': 'templates["result_yes"] if ans else templates["result_no"]',
        'ques_dict': '{}',
    },
    'edge': {
        'params': ['u', 'v'],
        'question_format': 'templates["question"].format(NID(u), NID(v))',
        'result_format': 'templates["result_yes"].format(NID(ques["u"]), NID(ques["v"])) if ans else templates["result_no"].format(NID(ques["u"]), NID(ques["v"]))',
        'ques_dict': '{"u": u, "v": v}',
        'extra_setup': '''    u = random.randint(0, num_nodes - 1)
    v = random.randint(0, num_nodes - 1)
    while u == v:
        v = random.randint(0, num_nodes - 1)''',
    },
    'common_neighbor': {
        'params': ['u', 'v'],
        'question_format': 'templates["question"].format(NID(u), NID(v))',
        'result_format': 'templates["result"]',
        'ques_dict': '{"u": u, "v": v}',
        'extra_setup': '''    u = random.randint(0, num_nodes - 1)
    v = random.randint(0, num_nodes - 1)
    while u == v:
        v = random.randint(0, num_nodes - 1)''',
    },
    'jaccard': {
        'params': ['u', 'v'],
        'question_format': 'templates["question"].format(NID(u), NID(v))',
        'result_format': 'templates["result"]',
        'ques_dict': '{"u": u, "v": v}',
        'extra_setup': '''    u = random.randint(0, num_nodes - 1)
    v = random.randint(0, num_nodes - 1)
    while u == v:
        v = random.randint(0, num_nodes - 1)''',
    },
}


def generate_question_function(task_name, config):
    """生成question_generation_multilang函数"""
    extra_setup = config.get('extra_setup', '    node = random.randint(0, num_nodes - 1)')
    return f'''def question_generation_multilang(config, g, lang=LANG_EN):
    """生成{task_name}问题 (支持多语言)"""
    import random
    templates = get_task_templates(TASK_NAME, lang)

    num_nodes = g.number_of_nodes()
{extra_setup}
    ques = {config['ques_dict']}
    ques_str = {config['question_format']}

    return ques, ques_str


def answer_and_inference_steps_generation_multilang(config, g, ques, lang=LANG_EN):
    """生成{task_name}推理步骤和答案 (支持多语言)"""
    templates = get_task_templates(TASK_NAME, lang)
    steps_str_en, ans, ans_str, reject = answer_and_inference_steps_generation(config, g, ques)

    steps_str = templates['steps_start']
    steps_str += {config['result_format']}

    return steps_str, ans, ans_str, reject'''


def fix_task_file(task_name, task_dir):
    """修复单个任务文件"""
    if task_name not in TASK_CONFIGS:
        print(f"跳过 {task_name} - 无配置")
        return

    config = TASK_CONFIGS[task_name]
    file_path = os.path.join(task_dir, f'{task_name}_multilang.py')

    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否有需要修复的问题
    if 'task_name ==' not in content:
        print(f"跳过 {task_name} - 已修复或无需修复")
        return

    # 生成新的函数
    new_functions = generate_question_function(task_name, config)

    # 使用正则表达式替换旧的函数
    # 匹配从 def question_generation_multilang 到下一个 def generate_a_sample_multilang
    pattern = r'def question_generation_multilang\(.*?\n\ndef generate_a_sample_multilang'
    replacement = new_functions + '\n\n\ndef generate_a_sample_multilang'

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"已修复: {task_name}")


def main():
    tasks_dir = '/nvme0/work/workspaces-zy/GraphInstruct/GTG/tasks'
    for task_name in TASK_CONFIGS:
        task_dir = os.path.join(tasks_dir, task_name)
        if os.path.isdir(task_dir):
            fix_task_file(task_name, task_dir)


if __name__ == '__main__':
    main()
