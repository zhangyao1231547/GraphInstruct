"""
多语言支持模块 - 用于生成中英文图推理数据

Usage:
    from GTG.utils.language import get_templates, LANG_EN, LANG_ZH

    # 获取英文模板 (默认)
    templates = get_templates(LANG_EN)

    # 获取中文模板
    templates = get_templates(LANG_ZH)
"""

# 语言常量
LANG_EN = 'en'
LANG_ZH = 'zh'

# 英文模板
TEMPLATES_EN = {
    # 通用图描述模板
    'graph': {
        'node_connected_to_single': 'Node {} is connected to node {}.',
        'node_connected_to_multiple': 'Node {} is connected to nodes {}.',
        'node_weight_single': 'Node {} is connected to node {} (weight: {}).',
        'node_weight_multiple': 'Node {} is connected to nodes {}.',
        'weight_format': '{} (weight: {})',
    },

    # BFS 任务模板
    'BFS': {
        'question': 'Start from node {}, output a sequence of traversal in breadth-first search (BFS) order.',
        'steps_start': "Let's run breadth-first search (BFS) step by step.\n",
        'visit_node': 'Visit node {}. ',
        'unvisited_neighbors': 'Unvisited neighbors of node {} are {}.\n',
        'result': 'So the BFS traversal is ',
    },

    # DFS 任务模板
    'DFS': {
        'question': 'Start from node {}, output a sequence of traversal in depth-first search (DFS) order.',
        'steps_start': "Let's run depth-first search (DFS) step by step.\n",
        'visit_node': 'Visit node {}. ',
        'neighbors': 'Neighbors of node {}: {}.\n',
        'backtrack': 'Backtrack.\n',
        'result': 'So the DFS traversal is ',
    },

    # shortest_path 任务模板
    'shortest_path': {
        'question': 'Calculate the distance of the shortest path from node {} to node {}.',
        'steps_start': "Let's solve it step by step. We can use the Dijkstra algorithm.\n",
        'init_distances': 'Initialize distances: {}.\n',
        'visit_node': 'Visit node {} with distance {}.\n',
        'update_distance': 'Update distance of node {} to {}.\n',
        'result': 'The shortest distance from node {} to node {} is ',
    },

    # page_rank 任务模板
    'page_rank': {
        'question': 'Which node has the largest PageRank value? The damping factor is {:.2f}. The number of iterations is {}. The initial PageRank value of each node is 1/N, where N is the number of nodes.',
        'steps_start': "Let's calculate PageRank step by step.\n",
        'iteration': 'Iteration {}: {}.\n',
        'result': 'The node with the largest PageRank is ',
    },

    # topological_sort 任务模板
    'topological_sort': {
        'question': 'Output the topological sorting of this graph. Topological sorting is a linear ordering of vertices such that for every directed edge (u, v), vertex u comes before v in the ordering.',
        'steps_start': "Let's solve it step by step.\n",
        'in_degree': 'In-degree of node {}: {}.\n',
        'add_to_result': 'Add node {} to the result.\n',
        'result': 'The topological sorting is ',
    },

    # cycle 任务模板
    'cycle': {
        'question': 'Is there a cycle in this graph?',
        'steps_start': "Let's check if there is a cycle step by step.\n",
        'yes': 'Yes',
        'no': 'No',
        'found_cycle': 'Found a cycle: {}.\n',
        'no_cycle': 'No cycle found.\n',
        'result_yes': 'Yes, there is a cycle.',
        'result_no': 'No, there is no cycle.',
    },

    # degree 任务模板
    'degree': {
        'question': 'What is the degree of node {}?',
        'steps_start': "Let's count the degree step by step.\n",
        'neighbors': 'Node {} is connected to nodes {}.\n',
        'count': 'The number of neighbors is {}.\n',
        'result': 'The degree of node {} is ',
    },

    # connectivity 任务模板
    'connectivity': {
        'question': 'Is node {} and node {} connected?',
        'steps_start': "Let's check connectivity step by step.\n",
        'yes': 'Yes',
        'no': 'No',
        'result_yes': 'Yes, node {} and node {} are connected.',
        'result_no': 'No, node {} and node {} are not connected.',
    },

    # edge 任务模板
    'edge': {
        'question': 'Is there an edge between node {} and node {}?',
        'steps_start': "Let's check if there is an edge step by step.\n",
        'yes': 'Yes',
        'no': 'No',
        'result_yes': 'Yes, there is an edge between node {} and node {}.',
        'result_no': 'No, there is no edge between node {} and node {}.',
    },

    # neighbor 任务模板
    'neighbor': {
        'question': 'What are the neighbors of node {}?',
        'steps_start': "Let's find the neighbors step by step.\n",
        'result': 'The neighbors of node {} are ',
    },

    # bipartite 任务模板
    'bipartite': {
        'question': 'Is this graph bipartite?',
        'steps_start': "Let's check if the graph is bipartite step by step.\n",
        'yes': 'Yes',
        'no': 'No',
        'result_yes': 'Yes, the graph is bipartite.',
        'result_no': 'No, the graph is not bipartite.',
    },

    # clustering_coefficient 任务模板
    'clustering_coefficient': {
        'question': 'What is the clustering coefficient of node {}?',
        'steps_start': "Let's calculate the clustering coefficient step by step.\n",
        'result': 'The clustering coefficient of node {} is ',
    },

    # common_neighbor 任务模板
    'common_neighbor': {
        'question': 'How many common neighbors do node {} and node {} have?',
        'steps_start': "Let's find the common neighbors step by step.\n",
        'neighbors_of': 'Neighbors of node {}: {}.\n',
        'common': 'Common neighbors: {}.\n',
        'result': 'The number of common neighbors is ',
    },

    # connected_component 任务模板
    'connected_component': {
        'question': 'Which nodes are in the same connected component as node {}?',
        'steps_start': "Let's find the connected component step by step.\n",
        'result': 'The connected component containing node {} is ',
    },

    # diameter 任务模板
    'diameter': {
        'question': 'What is the diameter of this graph?',
        'steps_start': "Let's calculate the diameter step by step.\n",
        'result': 'The diameter of the graph is ',
    },

    # jaccard 任务模板
    'jaccard': {
        'question': 'What is the Jaccard similarity between node {} and node {}?',
        'steps_start': "Let's calculate the Jaccard similarity step by step.\n",
        'neighbors_of': 'Neighbors of node {}: {}.\n',
        'union': 'Union: {}.\n',
        'intersection': 'Intersection: {}.\n',
        'result': 'The Jaccard similarity is ',
    },

    # maximum_flow 任务模板
    'maximum_flow': {
        'question': 'What is the maximum flow from node {} to node {}?',
        'steps_start': "Let's calculate the maximum flow step by step.\n",
        'result': 'The maximum flow is ',
    },

    # MST 任务模板
    'MST': {
        'question': 'What is the total weight of the minimum spanning tree?',
        'steps_start': "Let's find the minimum spanning tree step by step.\n",
        'add_edge': 'Add edge ({}, {}) with weight {}.\n',
        'result': 'The total weight of the MST is ',
    },

    # predecessor 任务模板
    'predecessor': {
        'question': 'What is the predecessor of node {} in the shortest path from node {}?',
        'steps_start': "Let's find the predecessor step by step.\n",
        'result': 'The predecessor of node {} is ',
    },

    # euler_path 任务模板
    'euler_path': {
        'question': 'Find an Euler path in this graph starting from node {}.',
        'steps_start': "Let's find the Euler path step by step.\n",
        'result': 'The Euler path is ',
    },

    # hamiltonian_path 任务模板
    'hamiltonian_path': {
        'question': 'Find a Hamiltonian path in this graph.',
        'steps_start': "Let's find the Hamiltonian path step by step.\n",
        'result': 'The Hamiltonian path is ',
    },
}

# 中文模板
TEMPLATES_ZH = {
    # 通用图描述模板
    'graph': {
        'node_connected_to_single': '节点 {} 连接到节点 {}。',
        'node_connected_to_multiple': '节点 {} 连接到节点 {}。',
        'node_weight_single': '节点 {} 连接到节点 {} (权重: {})。',
        'node_weight_multiple': '节点 {} 连接到节点 {}。',
        'weight_format': '{} (权重: {})',
    },

    # BFS 任务模板
    'BFS': {
        'question': '从节点 {} 开始，输出广度优先搜索 (BFS) 遍历序列。',
        'steps_start': '让我们逐步执行广度优先搜索 (BFS)。\n',
        'visit_node': '访问节点 {}。',
        'unvisited_neighbors': '节点 {} 的未访问邻居是 {}。\n',
        'result': '因此 BFS 遍历序列是 ',
    },

    # DFS 任务模板
    'DFS': {
        'question': '从节点 {} 开始，输出深度优先搜索 (DFS) 遍历序列。',
        'steps_start': '让我们逐步执行深度优先搜索 (DFS)。\n',
        'visit_node': '访问节点 {}。',
        'neighbors': '节点 {} 的邻居: {}。\n',
        'backtrack': '回溯。\n',
        'result': '因此 DFS 遍历序列是 ',
    },

    # shortest_path 任务模板
    'shortest_path': {
        'question': '计算从节点 {} 到节点 {} 的最短路径距离。',
        'steps_start': '让我们逐步解决这个问题。我们可以使用 Dijkstra 算法。\n',
        'init_distances': '初始化距离: {}。\n',
        'visit_node': '访问节点 {}，距离为 {}。\n',
        'update_distance': '更新节点 {} 的距离为 {}。\n',
        'result': '从节点 {} 到节点 {} 的最短距离是 ',
    },

    # page_rank 任务模板
    'page_rank': {
        'question': '哪个节点的 PageRank 值最大？阻尼系数为 {:.2f}，迭代次数为 {}。每个节点的初始 PageRank 值为 1/N，其中 N 是节点数。',
        'steps_start': '让我们逐步计算 PageRank。\n',
        'iteration': '第 {} 次迭代: {}。\n',
        'result': 'PageRank 值最大的节点是 ',
    },

    # topological_sort 任务模板
    'topological_sort': {
        'question': '输出该图的拓扑排序。拓扑排序是顶点的线性排序，使得对于每条有向边 (u, v)，顶点 u 在排序中位于 v 之前。',
        'steps_start': '让我们逐步解决这个问题。\n',
        'in_degree': '节点 {} 的入度: {}。\n',
        'add_to_result': '将节点 {} 添加到结果中。\n',
        'result': '拓扑排序是 ',
    },

    # cycle 任务模板
    'cycle': {
        'question': '该图中是否存在环？',
        'steps_start': '让我们逐步检查是否存在环。\n',
        'yes': '是',
        'no': '否',
        'found_cycle': '发现环: {}。\n',
        'no_cycle': '未发现环。\n',
        'result_yes': '是的，存在环。',
        'result_no': '不，不存在环。',
    },

    # degree 任务模板
    'degree': {
        'question': '节点 {} 的度是多少？',
        'steps_start': '让我们逐步计算度。\n',
        'neighbors': '节点 {} 连接到节点 {}。\n',
        'count': '邻居数量是 {}。\n',
        'result': '节点 {} 的度是 ',
    },

    # connectivity 任务模板
    'connectivity': {
        'question': '节点 {} 和节点 {} 是否连通？',
        'steps_start': '让我们逐步检查连通性。\n',
        'yes': '是',
        'no': '否',
        'result_yes': '是的，节点 {} 和节点 {} 是连通的。',
        'result_no': '不，节点 {} 和节点 {} 不是连通的。',
    },

    # edge 任务模板
    'edge': {
        'question': '节点 {} 和节点 {} 之间是否存在边？',
        'steps_start': '让我们逐步检查是否存在边。\n',
        'yes': '是',
        'no': '否',
        'result_yes': '是的，节点 {} 和节点 {} 之间存在边。',
        'result_no': '不，节点 {} 和节点 {} 之间不存在边。',
    },

    # neighbor 任务模板
    'neighbor': {
        'question': '节点 {} 的邻居是什么？',
        'steps_start': '让我们逐步找出邻居。\n',
        'result': '节点 {} 的邻居是 ',
    },

    # bipartite 任务模板
    'bipartite': {
        'question': '该图是否是二分图？',
        'steps_start': '让我们逐步检查该图是否是二分图。\n',
        'yes': '是',
        'no': '否',
        'result_yes': '是的，该图是二分图。',
        'result_no': '不，该图不是二分图。',
    },

    # clustering_coefficient 任务模板
    'clustering_coefficient': {
        'question': '节点 {} 的聚类系数是多少？',
        'steps_start': '让我们逐步计算聚类系数。\n',
        'result': '节点 {} 的聚类系数是 ',
    },

    # common_neighbor 任务模板
    'common_neighbor': {
        'question': '节点 {} 和节点 {} 有多少个共同邻居？',
        'steps_start': '让我们逐步找出共同邻居。\n',
        'neighbors_of': '节点 {} 的邻居: {}。\n',
        'common': '共同邻居: {}。\n',
        'result': '共同邻居的数量是 ',
    },

    # connected_component 任务模板
    'connected_component': {
        'question': '与节点 {} 在同一连通分量中的节点有哪些？',
        'steps_start': '让我们逐步找出连通分量。\n',
        'result': '包含节点 {} 的连通分量是 ',
    },

    # diameter 任务模板
    'diameter': {
        'question': '该图的直径是多少？',
        'steps_start': '让我们逐步计算直径。\n',
        'result': '该图的直径是 ',
    },

    # jaccard 任务模板
    'jaccard': {
        'question': '节点 {} 和节点 {} 之间的 Jaccard 相似度是多少？',
        'steps_start': '让我们逐步计算 Jaccard 相似度。\n',
        'neighbors_of': '节点 {} 的邻居: {}。\n',
        'union': '并集: {}。\n',
        'intersection': '交集: {}。\n',
        'result': 'Jaccard 相似度是 ',
    },

    # maximum_flow 任务模板
    'maximum_flow': {
        'question': '从节点 {} 到节点 {} 的最大流是多少？',
        'steps_start': '让我们逐步计算最大流。\n',
        'result': '最大流是 ',
    },

    # MST 任务模板
    'MST': {
        'question': '最小生成树的总权重是多少？',
        'steps_start': '让我们逐步找出最小生成树。\n',
        'add_edge': '添加边 ({}, {})，权重为 {}。\n',
        'result': '最小生成树的总权重是 ',
    },

    # predecessor 任务模板
    'predecessor': {
        'question': '在从节点 {} 出发的最短路径中，节点 {} 的前驱是什么？',
        'steps_start': '让我们逐步找出前驱。\n',
        'result': '节点 {} 的前驱是 ',
    },

    # euler_path 任务模板
    'euler_path': {
        'question': '从节点 {} 开始，找出该图的欧拉路径。',
        'steps_start': '让我们逐步找出欧拉路径。\n',
        'result': '欧拉路径是 ',
    },

    # hamiltonian_path 任务模板
    'hamiltonian_path': {
        'question': '找出该图的哈密顿路径。',
        'steps_start': '让我们逐步找出哈密顿路径。\n',
        'result': '哈密顿路径是 ',
    },
}


def get_templates(lang=LANG_EN):
    """
    获取指定语言的模板

    Args:
        lang: 语言代码, 'en' 或 'zh'

    Returns:
        对应语言的模板字典
    """
    if lang == LANG_ZH:
        return TEMPLATES_ZH
    return TEMPLATES_EN


def get_task_templates(task_name, lang=LANG_EN):
    """
    获取指定任务和语言的模板

    Args:
        task_name: 任务名称
        lang: 语言代码

    Returns:
        对应任务的模板字典
    """
    templates = get_templates(lang)
    return templates.get(task_name, {})


def graph_to_natural_language_multilang(g, lang=LANG_EN):
    """
    将图转换为自然语言描述 (支持多语言)

    Args:
        g: networkx 图
        lang: 语言代码

    Returns:
        图的自然语言描述字符串
    """
    from GTG.utils.utils import NID

    templates = get_templates(lang)['graph']
    s = ""

    for u in g.nodes():
        neighbors = list(g.neighbors(u))
        k = len(neighbors)
        if k == 0:
            pass
        elif k == 1:
            s += templates['node_connected_to_single'].format(
                NID(u), NID(neighbors[0])
            ) + '\n'
        else:
            neighbors_str = ', '.join([NID(v) for v in neighbors])
            s += templates['node_connected_to_multiple'].format(
                NID(u), neighbors_str
            ) + '\n'

    return s.strip()


def graph_with_edge_weight_to_natural_language_multilang(g, lang=LANG_EN, weight_name='weight'):
    """
    将带权重的图转换为自然语言描述 (支持多语言)

    Args:
        g: networkx 图
        lang: 语言代码
        weight_name: 权重属性名

    Returns:
        图的自然语言描述字符串
    """
    from GTG.utils.utils import NID, get_neighbor_and_edge_weight

    templates = get_templates(lang)['graph']
    s = ""

    for u in g.nodes():
        neighbors = list(get_neighbor_and_edge_weight(g, u, weight_name))
        k = len(neighbors)
        if k == 0:
            pass
        elif k == 1:
            s += templates['node_weight_single'].format(
                NID(u), NID(neighbors[0][0]), neighbors[0][1]
            ) + '\n'
        else:
            neighbors_str = ', '.join([
                templates['weight_format'].format(NID(v[0]), v[1])
                for v in neighbors
            ])
            s += templates['node_weight_multiple'].format(
                NID(u), neighbors_str
            ) + '\n'

    return s.strip()
