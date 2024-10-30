from __future__ import division

import json
import os.path
import time
import math
import random
import heapq
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import zip_longest

from MCTS.SWEGPT.run_gpt_and_bm25 import get_summary_results
# from MCTS.mtcs_repo_graph import Action

def randomPolicy(state):
    while not state.isTerminal():
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()



class treeNode():
    def __init__(self, state, parent, id):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.gpt_score = 0
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}
        self.id = id

    def __str__(self):
        s=[]
        s.append("stateName: %s"%(str(self.state.node)))
        s.append("totalReward: %s"%(self.totalReward))
        s.append("numVisits: %d"%(self.numVisits))
        if self.numVisits!=0:
            s.append("avgReward: %d"%(self.totalReward/self.numVisits))
        s.append("isTerminal: %s"%(self.isTerminal))
        # s.append("possibleActions: %s"%(self.children.keys()))
        return "%s: {%s}"%(self.__class__.__name__, ', '.join(s))


global_repo_prompt_src = """
An code repository awareness tool has been deployed to identify the suspicious related code to be fixed. 
You can get some functions name and locations from tool output that may be related to the issue. 

Note:
1. You can choose to use the results from this tool, if you think they are useful.
2. If no result is relevant, the above tool output results are not considered.
3. The functions obtained by the repository awareness tool may still be insufficient, and you need to continue to use your search tool to collect relevant information according to the issue.

"""

global_repo_prompt = """
An code repository awareness tool has been deployed to identify the related code about the issue. 
You can get some functions name and locations from tool output that may be related to the issue. 

Notice:
1. You can choose to use the results from this tool, if you think they are useful.
2. If no result is relevant, the above tool output results are not considered.
3. The functions obtained by the repository awareness tool may still be insufficient. It only provides information that may be related to the issue. You must continue to use your search tool to collect as much relevant information as possible based on the issue to locate the root cause to ensure that you can solve it.
4. If the issue has detailed reproduction steps, error information, log information, code snippets, test cases, impact scope and other information, it can help you quickly locate and solve the bug. Please gather as much information as you may need.

"""

# If the context collected is not sufficient to solve the issue, please indicate what you want to collect in the next step.
global_summary_prompt = """You are a senior software developer maintaining a large project.
You are working on an issue submitted to your project.
The issue contains a description marked between <issue> and </issue>.
We've collected some code snippets from the code repository that may be relevant.
To help diagnose and fix issues in software repositories, let's systematically analyze the collected context step by step. 
<issue>
{issue_content}
</issue>
<collected content>
{collected_content}
</collected content>
Analyze results:
"""

def interleave_lists(list1, list2):
    new_list = []
    for item1, item2 in zip_longest(list1, list2, fillvalue=None):
        if item1 is not None and item1 not in new_list:
            new_list.append(item1)
        if item2 is not None and item2 not in new_list:
            new_list.append(item2)
    return new_list


class mcts():
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2),
                 rolloutPolicy=randomPolicy, topNPaths=3):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy
        self.topNPaths = topNPaths

    def search(self, initialState, needDetails=True, collectBestPaths=None, insertNode=None, issue_content="", instance_id=""):
        self.root = treeNode(initialState, None, 'root')
        mcts_cost = {"input_tokens": 0, "output_tokens": 0}
        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            timeLimit = time.time() + 60
            i = 0
            while time.time() < timeLimit or i < self.searchLimit:
                # 每五轮输出一次结果
                if i!=0 and i%5==0:
                    best_paths, _ = collectBestPaths(initialState.get_node(), topN=self.topNPaths)
                    print(f'{i/5}th: {len(best_paths)}\ncollecting best paths {best_paths}')
                    # mid
                    if len(best_paths) >= 6 or i>=50:
                        break
                # 并行执行每一轮
                input_tokens, output_tokens = self.executeRound_Parallel_all(insertNode=insertNode)
                mcts_cost['input_tokens'] += input_tokens
                mcts_cost['output_tokens'] += output_tokens
                i += 1

        """
        bestChild = self.getBestChild(self.root, 0)
        action = (action for action, node in self.root.children.items() if node is bestChild).__next__()
        if needDetails:
            return {"action": action, "expectedReward": bestChild.totalReward / bestChild.numVisits}
        else:
            return action
        """

        best_paths, code_map = collectBestPaths(initialState.get_node(), topN=self.topNPaths)
        print('code_map_keys', code_map.keys())

        sbfl_path = f'SWESynInfer/experiment/history_sbfl/{instance_id}.json'
        if os.path.exists(sbfl_path):
            with open(sbfl_path, 'r') as f:
                sbfl_data = json.load(f)
            suspicious_methods = sbfl_data['suspicious_methods']
        else:
            suspicious_methods = []

        if len(best_paths) > 0 or len(suspicious_methods) > 0:
            tool_output = global_repo_prompt
            related_code = ""
            tool_output += f"Related methods:\n"
            keys = []

            mcts_keys = []
            mcts_codes = {}
            for path_key in best_paths:
                function_key = list(path_key.keys())[0]
                code_mcts = code_map[function_key]
                mcts_keys.append(function_key)
                mcts_codes[function_key] = code_mcts

            sbfl_keys = []
            sbfl_codes = {}
            for suspicious_method in suspicious_methods:
                sbfl_keys.append(suspicious_method[0])
                sbfl_codes[suspicious_method[0]] = suspicious_method[1]

            print('mcts', mcts_keys)
            print('sbfl', sbfl_keys)
            new_keys = interleave_lists(mcts_keys, sbfl_keys)
            print('new_keys', new_keys)
            new_codes = []
            for new_key in new_keys:
                tool_output += f"{new_key}\n"
                if new_key in mcts_codes.keys():
                    new_code = mcts_codes[new_key]
                else:
                    new_code = sbfl_codes[new_key]
                related_code += f"{new_key}\n<code>\n{new_code}\n</code>\n"

            summary_output = global_summary_prompt.format(issue_content=issue_content, collected_content=related_code)
        else:
            tool_output = ""
            summary_output = ""
        print('collecting best paths lens:', len(best_paths))

        if needDetails:
            return {"paths": tool_output, 'summary': summary_output, 'mcts_cost': mcts_cost}
        else:
            return {"paths": tool_output, 'summary': summary_output, 'mcts_cost': mcts_cost}

    def executeRound(self):
        """
            execute a selection-expansion-simulation-backpropagation round
        """
        node = self.selectNode(self.root)
        reward, graph_node = self.rollout(node.state)
        self.backpropogate(node, reward)

    def executeRound_GraphParallel(self, insertNode):
        """
            execute a selection-expansion-simulation-backpropagation round
        """
        node = self.selectNode(self.root)
        reward, graph_node = self.rollout(node.state)
        if reward > 6:
            local_nodes = self.local_expand(graph_node, self.root, insertNode)
            # 并行local_nodes
            with ThreadPoolExecutor(max_workers=len(local_nodes)) as executor:
                future_to_node = {executor.submit(self.rollout, node.state): node for node in local_nodes}
                for future in future_to_node:
                    reward, graph_node = future.result()
                    self.backpropogate(future_to_node[future], reward)
        self.backpropogate(node, reward)

    def executeRoundParallel(self):
        """
            parallel execute a selection-expansion-simulation-backpropagation round
        """
        nodes = self.selectNodes(self.root, num_nodes=5)  # 选择多个节点(<=10)
        with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
            future_to_node = {executor.submit(self.rollout, node.state): node for node in nodes}
            for future in future_to_node:
                reward, graph_node = future.result()
                self.backpropogate(future_to_node[future], reward)


    def executeRound_Parallel_all(self, insertNode):
        nodes = self.selectNodes(self.root, num_nodes=10)  # 选择多个节点(<=10)
        good_nodes = []
        good_nodes_iter2 = []
        max_workers = min(len(nodes), 10)
        input_tokens = 0
        output_tokens = 0
        if max_workers > 0:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_node = {executor.submit(self.rollout, node.state): node for node in nodes}
                for future in as_completed(future_to_node):
                    try:
                        
                        rewards, graph_node = future.result()
                        reward, cost_record = rewards[0], rewards[1]
                        input_tokens += cost_record['input_tokens']
                        output_tokens += cost_record['output_tokens']
                        if reward >= 8:  # before 6
                            good_nodes.append(graph_node)
                        self.backpropogate(future_to_node[future], reward)
                    except Exception as e:
                        print(f'任务执行错误：{e}')
                        time.sleep(1)

        return input_tokens, output_tokens
    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                # print(f"Node fully expanded. Moving to best child.")
                node = self.getBestChild(node, self.explorationConstant)
            else:
                # print(f"Expanding node: {str(node)}")
                return self.expand(node)
        return node

    def selectNodes(self, node, num_nodes=10):
        selectedNodes = []

        while len(selectedNodes) < num_nodes and not node.isTerminal:
            if node.isFullyExpanded:
                # 是否可以选择多个节点，最优和次优
                if len(selectedNodes) > 0:
                    break
                # print(f"Node fully expanded. Moving to best child.")
                node = self.getBestChild(node, self.explorationConstant)
            else:
                # print(f"Expanding node: {str(node)}")
                new_node = self.expand(node)
                if new_node:
                    selectedNodes.append(new_node)

        # 最后输出
        if len(selectedNodes) == 0:
            return [node]
        else:
            return selectedNodes

    def selectNodes2(self, node, num_nodes=10):
        selectedNodes = []
        while len(selectedNodes) < num_nodes and not node.isTerminal:
            if node.isFullyExpanded:
                if len(selectedNodes) > 0:
                    break
                nodes = self.getBestChilds(node, self.explorationConstant)
                for temp_node in nodes:
                    selectedNodes.extend(self.selectNodes(temp_node, 5))
                selectedNodes = selectedNodes[:num_nodes]
                return selectedNodes
            else:
                # print(f"Expanding node: {str(node)}")
                new_node = self.expand(node)
                selectedNodes.append(new_node)
        return selectedNodes


    def expand(self, node):
        actions = node.state.getPossibleActions()
        # print(f"Expanding actions: {str(actions)}")
        for action in actions:
            if action not in node.children:
                select_action = node.state.takeAction(action)
                newNode = treeNode(select_action, node, hash(action))
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                # print(f"New node created: {str(newNode)}")
                return newNode

        node.isFullyExpanded = True
        return None

    def local_expand(self, graph_node, tree_root, insertNode=None):
        # STEP(1) 只选择 class_function 和 top-level function.
        reference_who = [node for node in graph_node.reference_who if node.node_type not in ['class', 'global_var']]
        who_reference_me = [node for node in graph_node.who_reference_me if node.node_type not in ['class', 'global_var']]
        all_reference = reference_who + who_reference_me
        # STEP(2) 把这些节点添加到树
        local_nodes = []
        for node in all_reference:
            current_node = insertNode(node, tree_root)
            local_nodes.append(current_node)
        return local_nodes


    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            # print(f"Backpropagating: Node {str(node)}, Reward {reward}")
            node = node.parent

    def getBestChild(self, node, explorationValue):
        # todo: https://github.com/yzhq97/AlphaGomokuZero/blob/master/training/src/model/mcts_alphaZero.py
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            try:
                nodeValue = node.state.getCurrentPlayer() * (0.1) * child.totalReward / child.numVisits + explorationValue * math.sqrt(
                    2 * math.log(node.numVisits) / child.numVisits)
            except:
                nodeValue = 0.0
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)


    def getBestChilds(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        secondNode = None
        for child in node.children.values():
            nodeValue = node.state.getCurrentPlayer() * (0.1) * child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                secondNode = bestNodes[0] if len(bestNodes) > 0 else None
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        if len(bestNodes) > 1:
            selectedNodes = random.sample(bestNodes, 2)
        else:
            if secondNode:
                selectedNodes = [bestNodes[0], secondNode]
            else:
                selectedNodes = [bestNodes[0]]

        return selectedNodes

    def collectBestPaths(self, node, current_path, paths_heap, current_score=0, topN=1):
        current_score += node.totalReward / node.numVisits  # 更新当前路径的累积得分
        if node.isTerminal:
            # 使用当前路径的累积得分作为优先队列的键值
            heapq.heappush(paths_heap, (-current_score, node.id, current_path))
            if len(paths_heap) > topN:
                heapq.heappop(paths_heap)
        else:
            all_child_scores = []
            for action, child in node.children.items():
                # 直接使用平均回报进行评估，是否可以只使用totalReward呢？
                score = child.totalReward / child.numVisits
                all_child_scores.append((score, action, child))
            all_child_scores.sort(key=lambda x: x[0], reverse=True)  # 根据分数降序排列

            # 递归地收集最高得分的路径
            for _, action, child in all_child_scores[:topN]:  # 只考虑前 topN 个子节点
                self.collectBestPaths(child, current_path + [(action, child)], paths_heap, current_score, topN)
