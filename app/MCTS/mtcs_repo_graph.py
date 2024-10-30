from __future__ import division

from copy import deepcopy
from MCTS.mtcs import mcts, treeNode
import pickle
import time
from MCTS.SWEGPT.run_gpt_and_bm25 import get_gpt4_and_bm25_results, get_bm25_score, get_bm25_retriever, get_reward_value, get_summary_results
from MCTS.SWEGPT.graph_meta_info import get_graph_info_filter, save_graph, load_graph
"""
class Node:
    def __init__(self, obj_name: str, node_type: str, path: str,
                 code_start_line: int = -1, code_end_line: int = -1, code_content: str = ""):
        self.obj_name = obj_name
        self.node_type = node_type
        self.path = path
        self.code_start_line = code_start_line
        self.code_end_line = code_end_line
        self.code_content = code_content
        self.reference_who = []
        self.who_reference_me = []
        self.child = []
"""

class GlobalInfo():
    issue = ""
    repo_path = ""
    # bm25分类器
    bm25_retriever = None
    # embedding分类器
    gpt_value_cache = dict()
    save_path = ""

    @classmethod
    def init(cls, issue, repo_path, save_path):
        cls.issue = issue
        cls.repo_path = repo_path
        cls.bm25_retriever = get_bm25_retriever(repo_path=repo_path)
        cls.save_path = save_path

    @classmethod
    def load_cache_value(cls, file_path):
        try:
            with open(file_path, 'rb') as file:
                cls.gpt_value_cache = pickle.load(file)
        except (FileNotFoundError, EOFError, pickle.PickleError):
            print("Failed to load cached value in {}".format(file_path))
            cls.gpt_value_cache = {}

    @classmethod
    def save_cache_value(cls, file_path):
        try:
            with open(file_path, 'wb') as file:
                pickle.dump(cls.gpt_value_cache, file)
        except pickle.PickleError as e:
            print("Failed to save cache:", e)

    @classmethod
    def set_issue(cls, issue):
        cls.issue = issue

    @classmethod
    def get_issue(cls):
        return cls.issue

    @classmethod
    def get_repo_path(cls):
        return cls.repo_path

    @classmethod
    def set_gpt_value_cache(cls, key, value):
        cls.gpt_value_cache[key] = value
        # todo 现在先保存多次，以后一个repo只保留一次
        cls.save_cache_value(cls.save_path)
        # 可以合并同一个instance_id的dict。

    @classmethod
    def get_gpt_value_cache(cls, key):
        if key not in cls.gpt_value_cache.keys():
            cls.set_gpt_value_cache(key, -1)
        return cls.gpt_value_cache[key]

class GraphSearchState():
    def __init__(self, root_node):
        self.node = root_node
        self.currentPlayer = 1
        # 这里缓存节点的GPT判定结果，如果有，则直接返回。
        # 可以加速，但是gpt输出的不稳定性，输出多次可能会有助于结果。

    def get_node(self):
        return self.node

    def getCurrentPlayer(self):
        return self.currentPlayer

    def getPossibleActions(self):
        # 过滤bm25分数太低的节点，然后按照bm25排序
        scoredActions = []
        for child in self.node.child:
            bm25_score = get_bm25_score(GlobalInfo.issue, child.code_content, GlobalInfo.bm25_retriever)
            if bm25_score >= 0:  # 只考虑分数高于20的节点
                scoredActions.append((bm25_score, Action(node=child)))

        # 按照bm25分数从高到低排序
        scoredActions.sort(reverse=True, key=lambda x: x[0])

        # 从元组中提取出Action对象
        possibleActions = [action for _, action in scoredActions]
        return possibleActions

    def takeAction(self, action):
        newState = deepcopy(self)
        newState.node = action.node
        newState.currentPlayer = self.currentPlayer
        return newState

    def isTerminal(self):
        if not self.node.child:
            return True
        return False

    def getReward(self):
        # todo: STEP(1) 使用gpt4将self.node信息（位置信息，代码信息）及引用信息，issue输入给模型进行打分。
        issue = GlobalInfo.get_issue()
        method_start, method_end = self.node.code_start_line, self.node.code_end_line
        # {method_type} method {method_name} in {rel_file_path} file
        method_name = self.node.obj_name
        global_key = f'_{method_name}_{str(method_start)}_{str(method_end)}'
        if self.node.node_type == "class_function":
            method_name += " in " + self.node.parent.obj_name + " class"
        global_value = GlobalInfo.get_gpt_value_cache(key=global_key)
        if global_value != -1:
            # print(f'use cache value: {global_key}:{global_value}')
            return global_value, {"input_tokens": 0, "output_tokens": 0}
        score, input_tokens, output_tokens = get_reward_value(issue, self.node.code_content, self.node.node_type, method_name, self.node.path)
        GlobalInfo.set_gpt_value_cache(key=global_key, value=score)
        # print(f'### {method_name} ### rsp: {score}')
        
        # 新增记录，返回每一次的开销；
        mcts_cost_record = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        
        return score, mcts_cost_record


class Action():
    def __init__(self, node):
        self.node = node

    def __str__(self):
        return f"{self.node.obj_name}: {self.node.node_type}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.node == other.node

    def __hash__(self):
        return hash((self.node.obj_name, self.node.node_type, self.node.path))

def rolloutPolicy(state):
    # todo done: STEP(1) 根据bm25分数或者embedding相似性分数选择action,每次选择最高的分数
    while not state.isTerminal():
        actions = state.getPossibleActions()
        if len(actions) == 0:
            return 0, None
        max_score, index = -1, 0
        for idx, action in enumerate(actions):
            # STEP(1.1) 获取issue和state对应code的bm25分数
            code_content = action.node.code_content
            issue = GlobalInfo.get_issue()
            score = get_bm25_score(issue, code_content, GlobalInfo.bm25_retriever)
            # print(f"*** {action.node.obj_name} bm25 score is {score}")
            # todo 这里也可以把socre存下来，因为是固定值，可以用于后续select那里。
            if score > max_score:
                index = idx
                max_score = score
        # print(f'select action {actions[index].node.obj_name}, score {max_score}')
        state = state.takeAction(actions[index])
    # print(f'gpt for state {state.node.obj_name}, type: {state.node.node_type}')
    return state.getReward(), state.node

def insert_to_treenode(leaf_graph_node, treenode_root):
    path = []

    # 不包括根，从文件开始
    while leaf_graph_node.parent is not None:
        path.append(leaf_graph_node)
        leaf_graph_node = leaf_graph_node.parent

    # 反转路径，逆向递归构建treeNode
    path.reverse()

    current_node = treenode_root
    for node2 in path:
        action = Action(node=node2)
        if action not in current_node.children:
            actions = current_node.state.getPossibleActions()
            select_action = current_node.state.takeAction(action)
            newNode = treeNode(select_action, current_node, hash(action))
            current_node.children[action] = newNode
            if len(actions) <= len(current_node.children):
                current_node.isFullyExpanded = True
            # print(f"Local expand new node created: {str(newNode)}")
            current_node = newNode
        else:
            current_node = current_node.children[action]

    return current_node


def collectBestPaths_GPTValue(node, topN=10):
    paths_heap = []
    larger_paths_heap = []
    code_map = {}
    # STEP (1): 不在MTCS的树上搜索，而是在整个Graph上搜索叶子节点，因为MTCS的树可能不全。
    # 遍历node，取所有的gpt分数。
    for files_instance in node.child:
        current_path = f"<{files_instance.node_type}>{files_instance.obj_name}.py</{files_instance.node_type}>"
        for class_fun_instance in files_instance.child:
            sub_current_path = current_path + f"<{class_fun_instance.node_type}>{class_fun_instance.obj_name}</{class_fun_instance.node_type}>"
            if not class_fun_instance.child:
                method_start, method_end = class_fun_instance.code_start_line, class_fun_instance.code_end_line
                source_code = class_fun_instance.code_content
                # {method_type} method {method_name} in {rel_file_path} file
                method_name = class_fun_instance.obj_name
                global_key = f'_{method_name}_{str(method_start)}_{str(method_end)}'
                value = GlobalInfo.get_gpt_value_cache(global_key)
                if value >= 6:
                    paths_heap.append({sub_current_path: value})
                if value >= 4:
                    code_map[sub_current_path] = source_code
                    larger_paths_heap.append({sub_current_path: value})
            else:
                for fun_instance in class_fun_instance.child:
                    sub_sub_current_path = sub_current_path + f"<{fun_instance.node_type}>{fun_instance.obj_name}</{fun_instance.node_type}>"
                    if not fun_instance.child:
                        method_start, method_end = fun_instance.code_start_line, fun_instance.code_end_line
                        source_code = fun_instance.code_content
                        # {method_type} method {method_name} in {rel_file_path} file
                        method_name = fun_instance.obj_name
                        global_key = f'_{method_name}_{str(method_start)}_{str(method_end)}'
                        value = GlobalInfo.get_gpt_value_cache(global_key)
                        if value >= 6:
                            paths_heap.append({sub_sub_current_path: value})
                        if value >= 4:
                            code_map[sub_sub_current_path] = source_code
                            larger_paths_heap.append({sub_sub_current_path: value})
    if len(paths_heap) > 0:
        sorted_list_of_dicts = sorted(paths_heap, key=lambda x: list(x.values())[0], reverse=True)
    else:
        sorted_list_of_dicts = sorted(larger_paths_heap, key=lambda x: list(x.values())[0], reverse=True)
    return sorted_list_of_dicts[:topN], code_map


def mtcs_repo_graph(issue: str, repo_path: str, graph_save_dir, graph_load_dir=None, instance_id=""):
    """
    repo_path: 绝对路径
    """
    # STEP(1) repo issue和code构建bm25索引和gpt4获取文件候选项目录
    # STEP(1.1) 利用bm25和gpt4过滤文件
    filter_file_paths = get_gpt4_and_bm25_results(query=issue, repo_path=repo_path, use_gpt=True)
    print('### filter_file_paths ###', filter_file_paths, len(filter_file_paths))

    # STEP(2) 构建repo graph，利用(1)中的目录
    print('building graph')
    if graph_load_dir:
        print('loading graph')
        root_node = load_graph(graph_load_dir)
    else:
        root_node = get_graph_info_filter(repo_path, filter_file_paths)
        save_graph(root_node, graph_save_dir)
    print('build graph done')

    # STEP(3) MTCS
    initialState = GraphSearchState(root_node=root_node)
    searcher = mcts(iterationLimit=20, rolloutPolicy=rolloutPolicy, topNPaths=10)   # rolloutPolicy=randomPolicy
    # searcher = mcts(iterationLimit=50, rolloutPolicy=rolloutPolicy, topNPaths=10)   # rolloutPolicy=randomPolicy
    action = searcher.search(initialState=initialState, needDetails=True,
                             collectBestPaths=collectBestPaths_GPTValue, insertNode=insert_to_treenode,
                             issue_content=issue, instance_id=instance_id)
    # print(action['paths'])
    tool_output = action['paths']
    summary = action['summary']
    mcts_cost = action['mcts_cost']
    if summary != '':
        summary_result = get_summary_results(summary)
        tool_output += f'Analyze results: {summary_result}\n'
    return tool_output, mcts_cost


def merge_dictionaries(file_path1, file_path2, save_path):
    with open(file_path1, 'rb') as file:
        data1 = pickle.load(file)
    with open(file_path2, 'rb') as file:
        data2 = pickle.load(file)
    merged_dict = data1.copy()
    # 如果键重复，dict2 中的键值将覆盖 dict1 中的键值
    merged_dict.update(data2)
    with open(save_path, 'wb') as file:
        pickle.dump(merged_dict, file)
    print('merge_dictionaries finished')


def get_orcal_data(patch, repo_path):
    import re
    from SWEGPT.search_utils import get_all_functions_in_file, get_all_classes_in_file
    def get_file_from_patch(path: str):
        matched_paths = re.findall(r"diff --git a(.*?) b", path)
        return matched_paths

    def findall(query, text):
        ret = []
        for m in re.finditer(query, text):
            ret.append(m.start())
        return ret
    # 分割patch
    def split_patch(patch):
        diffs = findall("diff --git", patch)
        # print(diffs)
        split_patch = []
        for idx, split_idx in enumerate(diffs):
            if len(diffs) > 1 and idx + 1 == len(diffs):
                break
            split_temp = patch[diffs[idx]:diffs[idx + 1]] if len(diffs) > 1 else patch
            # print(get_file_from_patch(split_temp))
            file_path = get_file_from_patch(split_temp)
            # print(file_path[0])
            if not file_path:
                continue
            file_path = file_path[0]
            # 匹配行号
            change_positions = re.findall(r"@@ (.*?) @@", split_temp)
            positions_save = []
            for positions in change_positions:
                temp_position = positions.split(" ")[0]
                # print(temp_position)
                # print(positions)
                if len(temp_position.split(",")) == 1:  # 有时候会有个-1, ""的情况
                    final_position = [temp_position.split(",")[0][1:], 100]
                else:
                    final_position = [temp_position.split(",")[0][1:], temp_position.split(",")[1]]
                positions_save.append(final_position)
            split_patch.append([file_path, positions_save])
            # print(positions_save)
            # print(split_patch)
        return split_patch

    # 获取oracle数据
    # 获取文件路径
    patchs = split_patch(patch)
    # return patchs
    function_names = []
    # 获取文件内容
    # ['/src/sqlfluff/core/linter/common.py', [['67', '21']]]
    for patch_list in patchs:
        file_path = os.path.join(repo_path, patch_list[0][1:])
        if os.path.exists(file_path):
            # print(f'文件:{file_path}')
            # 获得file里的所有函数、类
            functions = get_all_functions_in_file(file_path)
            classes = get_all_classes_in_file(file_path)
        else:
            raise Exception(f"File {file_path} does not exist")
        for pos_list in patch_list[1]:
            flag = False
            start_line = int(pos_list[0])
            for function in functions:
                if start_line >= int(function[1]) and start_line <= int(function[2]):
                    function_names.append(file_path.split("/")[-1]+"/method:"+function[0])
                    flag = True
                    # print(f'找到函数:{function[0]}')
                    break
            # function里没有，去类里看看？
            if not flag:
                # print(f'没有找到函数，去类里找')
                for class_instance in classes:
                    # print(f'start_line:{start_line}, {int(class_instance[1])}, {int(class_instance[1])}')
                    if start_line >= int(class_instance[1]) and start_line <= int(class_instance[2]):
                        function_names.append(file_path.split("/")[-1]+"/class_name:"+class_instance[0])
                        flag = True
                        # print(f'找到类:{class_instance[0]}')
                        break
            if not flag:
                function_names.append(str(start_line))
    # 获取文件行数
    return set(function_names)
    # 获取文件路径


def run_mcts(issue, repo_path, global_save_path, instance_id, global_load_path=None, graph_load_path=None):
    current_timestamp = int(time.time())
    file_name = f"{global_save_path}/cache_{instance_id}_{current_timestamp}.pickle"
    graph_save_dir = f"{global_save_path}/graph_{instance_id}_{current_timestamp}.pkl"
    GlobalInfo.init(issue=issue, repo_path=repo_path, save_path=file_name)
    if global_load_path:
        GlobalInfo.load_cache_value(file_path=global_load_path)
    # 开始搜索
    result, mcts_cost = mtcs_repo_graph(issue=issue, repo_path=repo_path,
                             graph_save_dir=graph_save_dir, graph_load_dir=graph_load_path, instance_id=instance_id)
    return result, mcts_cost


if __name__=="__main__":
    print("__todo__")
    import json
    import os
    json_path = '../dev/swe-bench-dev-search.json'

    merge_file_path = 'MCTS/data/cache_sqlfluff__sqlfluff-4764_merge.pickle'

    with open(json_path, 'r') as f:
        datas = json.load(f)

    filter_instances = ["sqlfluff__sqlfluff-4764", "pvlib__pvlib-python-1213", "pvlib__pvlib-python-1093"]
    for data in datas:
        instance_id = data['instance_id']
        print('instance_id', instance_id)
        query = data['requirement']
        sub_repo_path = data['repo_path']   # "dev/sqlfluff__sqlfluff/2.0/codebase/sqlfluff__sqlfluff__2.0/"
        orcal_files = data['orcal_files']
        patch = data['patch']

        new_repo_path = sub_repo_path

        current_timestamp = int(time.time())
        file_name = f"MCTS/data/cache_{instance_id}_{current_timestamp}.pickle"
        GlobalInfo.init(issue=query, repo_path=new_repo_path, save_path=file_name)
        GlobalInfo.load_cache_value(file_path=merge_file_path)
        # 开始搜索
        print('issue:', query)
        mtcs_repo_graph(issue=query, repo_path=new_repo_path)

        print('### orcal_files ###', orcal_files)
        patchs = get_orcal_data(patch, sub_repo_path)
        print('patchs', patchs)

        print('### MTCS ending ###')

        break