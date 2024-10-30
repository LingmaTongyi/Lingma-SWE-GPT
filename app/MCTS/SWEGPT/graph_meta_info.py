import os
import pickle

import jedi

from MCTS.SWEGPT.search_utils import get_all_py_files, get_all_classes_in_file, get_class_signature, get_all_funcs_in_class_in_file
from MCTS.SWEGPT.search_utils import get_func_snippet_in_class, get_top_level_functions, get_code_snippets, get_global_variables_corrected
from MCTS.SWEGPT.search_utils import get_class_content
from enum import Enum
from tqdm import tqdm

class NodeType(Enum):
    _repo = "repo"
    _file = "file"
    _class = "class"
    _class_function = "class_function"
    _function = "top-level function"  # 文件内的常规function
    _global_var = "global_var"

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
        self.parent = None

    def __repr__(self):
        return f"{self.obj_name}: {self.node_type}"

    # 这里的引用是 类，函数，和全局变量
    def add_reference_me(self, child_node):
        # 谁引用了我
        self.who_reference_me.append(child_node)

    def add_reference_who(self, parent_node):
        # 我引用了谁
        self.reference_who.append(parent_node)

    def add_child(self, structure_child_node):
        self.child.append(structure_child_node)

    def set_parent(self, parent_node):
        self.parent = parent_node

    def print_child_info(self, level=0):
        if level > 3:  # 只打印到第三层
            return
        indent = "    " * level  # 缩进显示层级
        print(f"{indent}{self.obj_name}: {self.node_type}")
        if self.reference_who:
            print(f"{indent}- reference_who:{self.reference_who}")
        if self.who_reference_me:
            print(f"{indent}- who_reference_me:{self.who_reference_me}")
        for child in self.child:
            child.print_child_info(level + 1)

    def find_node_by_name(self, name):
        node_list = []
        # todo
        for child in self.child:
            if child.obj_name == name:
                node_list.append(child)
            else:
                node_list.extend(child.find_node_by_name(name))
        return node_list

    def find_node_by_name_and_file(self, name, file_path):
        node_list = []
        # todo
        for child in self.child:
            if file_path in child.path and child.obj_name == name:
                node_list.append(child)
            else:
                node_list.extend(child.find_node_by_name_and_file(name, file_path))
        return node_list

    def find_all_node_by_file(self, file_path):
        node_list = []
        for child in self.child:
            if file_path == child.path and child.node_type != NodeType._file.value:
                node_list.append(child)

            node_list.extend(child.find_all_node_by_file(file_path))
        return node_list

def find_all_referencer(
    variable_name, file_path, line_number, column_number, repo_path, in_file_only=False
):
    project = jedi.Project(path=repo_path)
    script = jedi.Script(path=file_path, project=project)
    try:
        if in_file_only:
            references = script.get_references(
                line=line_number, column=column_number, scope="file"
            )
        else:
            references = script.get_references(line=line_number, column=column_number)
        # 过滤出变量名为 variable_name 的引用，并返回它们的位置
        variable_references = [ref for ref in references if ref.name == variable_name]
        # print('2.3. variable_references', len(variable_references))
        return [
            (os.path.relpath(ref.module_path, repo_path), ref.line, ref.column)
            for ref in variable_references
            if not (ref.line == line_number and ref.column == column_number)
        ]
    except Exception as e:
        # 打印错误信息和相关参数
        print(f"Error occurred: {e}")
        return []

def get_graph_info(repo_path: str) -> Node:
    dg = Node(obj_name="repo_root", node_type=NodeType._repo.value, path=repo_path)

    all_py_files = get_all_py_files(repo_path)
    # print(all_py_files)
    for file_path in tqdm(all_py_files, total=len(all_py_files), desc='第一阶段'):
        with open(file_path, 'r') as f:
            code_content = f.read()
        rel_path = os.path.relpath(file_path, repo_path)
        file_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(file_name)[0]

        file_node = Node(obj_name=file_name_without_ext, node_type=NodeType._file.value, path=rel_path,
                         code_start_line=-1, code_end_line=-1, code_content="file_content")

        # STEP(1) 访问文件内的所有类
        all_class = get_all_classes_in_file(file_path)
        if all_class:
            for class_name, class_start_line, class_end_line in all_class:
                class_content = get_class_signature(file_path, class_name)
                class_node = Node(obj_name=class_name, node_type=NodeType._class.value, path=rel_path,
                                  code_start_line=class_start_line, code_end_line=class_end_line, code_content=class_content)
                class_node.set_parent(file_node)
                file_node.add_child(class_node)
                # STEP(1.5) 访问类内的所有函数
                all_funcs_in_class = get_all_funcs_in_class_in_file(file_path, class_name)
                if all_funcs_in_class:
                    for func_name, func_start_line, func_end_line in all_funcs_in_class:
                        func_content = get_func_snippet_in_class(file_path, class_name, func_name)
                        func_node = Node(obj_name=func_name, node_type=NodeType._class_function.value, path=rel_path,
                                         code_start_line=func_start_line, code_end_line=func_end_line, code_content=func_content)
                        func_node.set_parent(class_node)
                        class_node.add_child(func_node)

        # STEP(2) 访问文件内的所有函数
        all_funcs = get_top_level_functions(file_path)
        if all_funcs:
            for func_name, func_start_line, func_end_line in all_funcs:
                func_content = get_code_snippets(file_path, func_start_line, func_end_line)
                func_node = Node(obj_name=func_name, node_type=NodeType._function.value, path=rel_path,
                                 code_start_line=func_start_line, code_end_line=func_end_line, code_content=func_content)
                func_node.set_parent(file_node)
                file_node.add_child(func_node)

        # STEP(3) 访问文件内的所有全局变量
        all_vars = get_global_variables_corrected(file_path)
        if all_vars:
            for var_name, var_start_line, var_end_line in all_vars:
                var_content = get_code_snippets(file_path, var_start_line, var_end_line)
                var_node = Node(obj_name=var_name, node_type=NodeType._global_var.value, path=rel_path,
                                code_start_line=var_start_line, code_end_line=var_end_line, code_content=var_content)
                var_node.set_parent(file_node)
                file_node.add_child(var_node)

                # print(var_content)
        # 放入file_node
        file_node.set_parent(dg)
        dg.add_child(file_node)

    for file_path in tqdm(all_py_files, total=len(all_py_files), desc="第二阶段"):
        # STEP(4) 构建图中的引用关系。
        print('1. file_path', file_path)
        print('2. repo_path', repo_path)

        rel_path = os.path.relpath(file_path, repo_path)
        project = jedi.Project(repo_path, load_unsafe_extensions=False)
        script = jedi.Script(path=file_path, project=project)

        try:
            all_names = script.get_names(all_scopes=True)
        except Exception as e:
            print(f'[Error: get_names with all_scopes=True], {e}')
            try:
                all_names = script.get_names(all_scopes=False)
            except Exception as e:
                print(f'[Error: get_names with all_scopes=False], {e}')
                continue

        # STEP(4.1) 访问文件内的函数、类、全局变量等内容，其他不访问
        all_node = dg.find_all_node_by_file(rel_path)
        for name in all_names:
            # print(f"name:{name.name}, module_path:{os.path.relpath(name.module_path, repo_path)}, type:{name.type}, "
            #       f"module_name:{name.module_name}, line:{name.line}, col:{name.column}")

            # temp_node该文件内节点，主节点
            for temp_node in all_node:
                if temp_node.obj_name == name.name and temp_node.code_start_line == name.line:
                    # 返回name.name在哪儿引用了
                    all_reference = find_all_referencer(name.name, file_path, name.line, name.column, repo_path)
                    for reference in all_reference:
                        all_ref_file_node = dg.find_all_node_by_file(reference[0])
                        for file_node in all_ref_file_node:
                            # 找到引用的object，然后更新两个值，file_node引用的节点
                            if file_node.code_start_line <= reference[1] and file_node.code_end_line >= reference[1]:
                                # 如果被引用的节点是引用节点的子节点，则不添加，暂时添加
                                if file_node in temp_node.child:
                                    continue
                                # 如果已经加入了一次，就不加入了
                                if file_node not in temp_node.who_reference_me:
                                    temp_node.add_reference_me(file_node)
                                if temp_node not in file_node.reference_who:
                                    file_node.add_reference_who(temp_node)
    # 打印整个图
    dg.print_child_info()
    return dg

def get_graph_info_filter(repo_path: str, filter_path_list) -> Node:
    # 该函数使用gpt和bm25过滤后的文件分数。
    print('进入get_graph_info_filter函数')
    dg = Node(obj_name="repo_root", node_type=NodeType._repo.value, path=repo_path)
    all_py_files = filter_path_list
    # print(all_py_files)
    for file_path in tqdm(all_py_files, total=len(all_py_files), desc='第一阶段'):
        with open(file_path, 'r') as f:
            code_content = f.read()
        rel_path = os.path.relpath(file_path, repo_path)
        file_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(file_name)[0]

        file_node = Node(obj_name=file_name_without_ext, node_type=NodeType._file.value, path=rel_path,
                         code_start_line=-1, code_end_line=-1, code_content=code_content)

        # STEP(1) 访问文件内的所有类
        all_class = get_all_classes_in_file(file_path)
        if all_class:
            for class_name, class_start_line, class_end_line in all_class:
                # class_content = get_class_signature(file_path, class_name)
                class_content = get_class_content(file_path, class_start_line, class_end_line)
                class_node = Node(obj_name=class_name, node_type=NodeType._class.value, path=rel_path,
                                  code_start_line=class_start_line, code_end_line=class_end_line, code_content=class_content)
                class_node.set_parent(file_node)
                file_node.add_child(class_node)
                # STEP(1.5) 访问类内的所有函数
                all_funcs_in_class = get_all_funcs_in_class_in_file(file_path, class_name)
                if all_funcs_in_class:
                    for func_name, func_start_line, func_end_line in all_funcs_in_class:
                        func_content = get_func_snippet_in_class(file_path, class_name, func_name)
                        func_node = Node(obj_name=func_name, node_type=NodeType._class_function.value, path=rel_path,
                                         code_start_line=func_start_line, code_end_line=func_end_line, code_content=func_content)
                        func_node.set_parent(class_node)
                        class_node.add_child(func_node)
        # STEP(2) 访问文件内的所有函数
        all_funcs = get_top_level_functions(file_path)
        if all_funcs:
            for func_name, func_start_line, func_end_line in all_funcs:
                func_content = get_code_snippets(file_path, func_start_line, func_end_line)
                func_node = Node(obj_name=func_name, node_type=NodeType._function.value, path=rel_path,
                                 code_start_line=func_start_line, code_end_line=func_end_line, code_content=func_content)
                func_node.set_parent(file_node)
                file_node.add_child(func_node)
        # STEP(3) 访问文件内的所有全局变量
        all_vars = get_global_variables_corrected(file_path)
        if all_vars:
            for var_name, var_start_line, var_end_line in all_vars:
                var_content = get_code_snippets(file_path, var_start_line, var_end_line)
                var_node = Node(obj_name=var_name, node_type=NodeType._global_var.value, path=rel_path,
                                code_start_line=var_start_line, code_end_line=var_end_line, code_content=var_content)
                var_node.set_parent(file_node)
                file_node.add_child(var_node)
                # print(var_content)
        # 放入file_node
        file_node.set_parent(dg)
        dg.add_child(file_node)

    for file_path in tqdm(all_py_files, total=len(all_py_files), desc='第二阶段'):
        # STEP(4) 构建图中的引用关系。

        rel_path = os.path.relpath(file_path, repo_path)
        project = jedi.Project(repo_path, load_unsafe_extensions=False)
        script = jedi.Script(path=file_path, project=project)
        all_names = script.get_names(all_scopes=True)
        # STEP(4.1) 访问文件内的函数、类、全局变量等内容，其他不访问
        all_node = dg.find_all_node_by_file(rel_path)
        for name in all_names:
            try:
                if name.type == 'statement' or name.type == 'param':
                    continue
            except Exception as e:
                print(f'[Error]: {e}')
                continue
            # temp_node该文件内节点，主节点
            for temp_node in all_node:
                if temp_node.obj_name == name.name and temp_node.code_start_line == name.line:
                    # 返回name.name在哪儿引用了
                    all_reference = find_all_referencer(name.name, file_path, name.line, name.column, repo_path)
                    for reference in all_reference:
                        all_ref_file_node = dg.find_all_node_by_file(reference[0])
                        for file_node in all_ref_file_node:
                            # 找到引用的object，然后更新两个值，file_node引用的节点
                            if file_node.code_start_line <= reference[1] and file_node.code_end_line >= reference[1]:
                                # 如果被引用的节点是引用节点的子节点，则不添加，暂时添加
                                if file_node in temp_node.child:
                                    continue
                                # 如果已经加入了一次，就不加入了
                                if file_node not in temp_node.who_reference_me:
                                    temp_node.add_reference_me(file_node)
                                if temp_node not in file_node.reference_who:
                                    file_node.add_reference_who(temp_node)
    # 打印整个图
    dg.print_child_info()
    return dg


def save_graph(graph, file_path):
    """
    保存图到文件。

    :param graph: 要保存的图对象。
    :param file_path: 保存图的文件路径。
    """
    with open(file_path, 'wb') as file:
        pickle.dump(graph, file)

def load_graph(file_path):
    """
    从文件加载图。

    :param file_path: 图文件的路径。
    :return: 加载的图对象。
    """
    with open(file_path, 'rb') as file:
        graph = pickle.load(file)
    return graph


if __name__ == '__main__':
    repo_path = './test_repo/a_write_run/'
    get_graph_info(repo_path)
