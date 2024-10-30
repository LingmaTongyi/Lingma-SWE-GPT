import ast
import glob
import re
from dataclasses import dataclass
from os.path import join as pjoin
from typing import List, Optional, Tuple

from app import utils as apputils

def convert_print_statements(content: str) -> str:
    """
    Convert Python 2 print statements to Python 3 print function calls,
    handling single-line and multi-line print statements with multiple commas.
    """
    # 处理print后接控制语句的情况
    pattern = r'(print\s+.*?,\s*\n\s*(?:if|for|while|with|try:|print|import|else:)\s+.*)'


    # match = re.search(pattern, content)
    matches = re.findall(pattern, content)
    if matches:
        for matched_string in matches:
            update_matched_string = matched_string
            # 将匹配到的代码替换为 print("temp code")，保留缩进
            string_lines = matched_string.split('\n')
            for string_line in string_lines:
                if string_line.strip().startswith('print'):
                    new_string_line = string_line.replace(',', '')
                    update_matched_string = update_matched_string.replace(string_line, new_string_line)
            content = content.replace(matched_string, update_matched_string)


    # Regular expression to match Python 2 print statements not using parentheses
    print_regex = re.compile(r'^\s*print\s+(?!\().*')
    lines = content.split('\n')
    converted_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        match = print_regex.match(line)
        if match:
            indent = line[:line.index('print')]
            print_content = line[line.index('print') + 5:].strip()
            print_content = print_content.replace(">>", "")
            # Collect all lines for a multi-line print statement
            original_lines = [line]
            is_multiline = False
            while (print_content.endswith(',') or print_content.endswith('\\') or
                   print_content.endswith('%') or print_content.endswith('(') or
                   ("(" in print_content and ")" not in print_content)
            ):
                is_multiline = True
                i += 1
                next_line = lines[i]
                original_lines.append(next_line)
                print_content = print_content.rstrip('\\') + '\n' + next_line.strip()
            if is_multiline:
                # Handle inline comments
                comment = ''
                if '#' in print_content:
                    print_content, comment = print_content.split('#', 1)
                    comment = '#' + comment.strip()
                # Split by commas and strip each part
                print_args = [arg.strip() for arg in print_content.split(',')]
                # Join the arguments with commas and wrap in parentheses
                print_content = f"({', '.join(print_args)})"
                # Add print statement with the appropriate indent and comment
                converted_line = f"{indent}print{print_content} {comment}".rstrip()
                converted_line_parts = converted_line.split('\n')
                # Ensure the number of lines remains the same
                for j, original_line in enumerate(original_lines):
                    if j < len(converted_line_parts):
                        converted_lines.append(converted_line_parts[j])
                    else:
                        converted_lines.append('')
            else:
                # Replace single-line print statement with placeholder
                converted_line = f'{indent}print("temp code snippet")'
                converted_lines.append(converted_line)
        else:
            converted_lines.append(line)
        i += 1

    return '\n'.join(converted_lines)



def convert_exec_statements(content: str) -> str:
    """
    Convert Python 2 exec statements to a placeholder in Python 3.
    Specifically, it changes `exec code in env` to `exec(stream, g)`.
    """
    # Regular expression to match Python 2 exec statements
    exec_regex = re.compile(r'^\s*exec\s+(.+?)\s*(#.*)?$')
    lines = content.split('\n')
    converted_lines = []
    for line in lines:
        match = exec_regex.match(line)
        if match:
            indent = line[:line.index('exec')]
            comment = match.group(2) or ''
            converted_line = f"{indent}exec(stream, g) {comment}".rstrip()
            converted_lines.append(converted_line)
        else:
            converted_lines.append(line)
    return '\n'.join(converted_lines)

def convert_async_keyword(content: str) -> str:
    """
    Convert Python 2 async keyword arguments to Python 3 compatible form.
    Specifically, it changes `async=True` to `async_=True`.
    """
    async_regex = re.compile(r'(\basync)')
    lines = content.split('\n')
    converted_lines = []
    for line in lines:
        converted_line = async_regex.sub(r'async_', line)
        converted_lines.append(converted_line)
    return '\n'.join(converted_lines)


def convert_except_statements(content: str) -> str:
    """
    Convert Python 2 except statements to Python 3 syntax.
    """
    lines = content.split('\n')
    converted_lines = []
    for line in lines:
        # 匹配 'except ExceptionType, variable:' 或 'except (ExceptionType1, ExceptionType2), variable:' 模式
        match = re.match(r'^(\s*)except\s+(\(?.+?\)?)?\s*,\s*(\w+):', line)
        if match:
            indentation, exception_types, variable = match.groups()
            if exception_types:
                # 移除可能存在的括号
                exception_types = exception_types.strip('()')
                # 检查是否有多个异常类型
                if ',' in exception_types:
                    exception_types = f"({exception_types})"
                converted_line = f"{indentation}except {exception_types} as {variable}:"
            else:
                converted_line = f"{indentation}except Exception as {variable}:"
            converted_lines.append(converted_line)
        else:
            converted_lines.append(line)
    return '\n'.join(converted_lines)


def convert_raise_statements(content: str) -> str:
    """
    Convert Python 2 raise statements to Python 3 syntax.
    Handle both single line and multi-line raise statements.
    """
    # Regular expression to match Python 2 raise statements
    exec_regex = re.compile(r'^\s*raise\s+([^\s,]+)\s*,\s*(.*)')

    lines = content.split('\n')
    converted_lines = []
    buffer = []
    in_multiline_raise = False
    current_indent = ""

    for line in lines:
        if in_multiline_raise:
            buffer.append(line)
            # Check if the current line ends with a backslash indicating continuation
            if not line.strip().endswith('\\') and not line.strip().endswith('('):
                # End of the multi-line raise statement
                in_multiline_raise = False
                converted_multiline_raise = convert_multiline_raise(buffer, current_indent)
                converted_lines.extend(converted_multiline_raise)
                buffer = []
            continue

        match = exec_regex.match(line)
        if match:
            # If the line is already in Python 3 syntax, skip it
            if ("(" in line and ")" not in line) and "raise" in line:
                converted_lines.append(line)
                continue

            # 有可能在注释里
            if "raise" in line and line.count("\"\"\"") >= 1:
                converted_lines.append(line)
                continue

            # Check if the line ends with a backslash indicating continuation
            if line.strip().endswith('\\') or line.strip().endswith(','):
                buffer.append(line)
                in_multiline_raise = True
                current_indent = line[:line.index('raise')]
                continue

            indent = line[:line.index('raise')]
            comment = match.group(2) or ''
            converted_line = f"{indent}raise(stream, g)".rstrip()
            converted_lines.append(converted_line)
        else:
            converted_lines.append(line)

    # Handle any remaining buffered lines (in case of file ending with a multi-line raise)
    if buffer:
        converted_multiline_raise = convert_multiline_raise(buffer, current_indent)
        converted_lines.extend(converted_multiline_raise)

    return '\n'.join(converted_lines)

def convert_multiline_raise(lines: list, indent: str) -> list:
    """
    Convert a multi-line raise statement from Python 2 to Python 3 syntax.
    """
    # Join the lines to form the full raise statement
    full_statement = '\n'.join(lines)
    # Regular expression to match the Python 2 raise statement
    raise_regex = re.compile(r'^\s*raise\s+([^,]+)\s*,\s*(.*)', re.DOTALL)
    match = raise_regex.match(full_statement)
    if match:
        exception, value = match.groups()
        converted_statement = f"{indent}raise {exception}({value})"
        # Split the converted statement back into lines to match the original number of lines
        converted_lines = converted_statement.split('\n')
        # Ensure the number of lines matches the original
        while len(converted_lines) < len(lines):
            converted_lines.append(indent)
        return converted_lines
    return lines

def convert_xrange_to_range(content: str) -> str:
    """
    Convert Python 2 xrange() to Python 3 range().
    """
    return content.replace('xrange(', 'range(')

def convert_long_to_int(content: str) -> str:
    """
    Convert Python 2 long type to Python 3 int.
    """
    return content.replace('long(', 'int(').replace(' long ', ' int ')



def convert_long_integers(content: str) -> str:
    """
    Convert Python 2 long integers to Python 3 integers.
    Specifically, it changes `0L` to `0`, `0xFFL` to `0xFF`, etc.
    """
    # \b(\d+)L\b handles decimal long integers
    # \b(0[xX][0-9a-fA-F]+)L\b handles hexadecimal long integers
    long_int_regex = re.compile(r'\b(\d+L|0[xX][0-9a-fA-F]+L)\b')

    def replace_long(match):
        return match.group(0)[:-1]  # Remove the last character 'L'

    converted_content = long_int_regex.sub(replace_long, content)
    return converted_content


def convert_unicode_and_str(content: str) -> str:
    # Convert unicode() to str(), remove u'' literals
    content = content.replace('unicode(', 'str(')
    content = re.sub(r"u'([^']*)'", r"'\1'", content)
    content = re.sub(r'u"([^"]*)"', r'"\1"', content)
    content = re.sub(r'ur"([^"]*)"', r'r"\1"', content)
    content = re.sub(r"ur'([^']*)'", r"r'\1'", content)
    content = content.replace('from __future__ import unicode_literals\n', '')
    return content



def convert_dict_methods(content: str) -> str:
    """
    Convert Python 2 dict methods to Python 3 equivalents.
    """
    content = content.replace('.iteritems()', '.items()')
    content = content.replace('.iterkeys()', '.keys()')
    content = content.replace('.itervalues()', '.values()')
    return content


def convert_octal_literals(content: str) -> str:
    """
    Convert Python 2 octal literals to Python 3 octal literals.
    Specifically, it changes `027` to `0o27` and invalid octal literals like `099` to `99`.
    """

    def replace_octal(match):
        number = match.group(0)
        try:
            # Try to interpret as octal
            int(number, 8)
            return '0o' + number[1:]
        except ValueError:
            # If it fails, it's not a valid octal, just remove the leading zero
            return number.lstrip('0')

    octal_regex = re.compile(r'\b0[0-7]+\b|\b0[89]+\b')
    converted_content = octal_regex.sub(replace_octal, content)
    return converted_content


def convert_invalid_literals(content: str) -> str:
    """
    Convert invalid literals in Python 2 code to valid Python 3 literals.
    This includes converting invalid octal literals like `0.0o01` to `0.001`.
    """

    def replace_invalid_literal(match):
        number = match.group(0)
        if '0o' in number:
            # This is an invalid octal literal, convert to float
            return number.replace('0o', '')
        return number

    invalid_literal_regex = re.compile(r'0\.\d+o\d+')
    converted_content = invalid_literal_regex.sub(replace_invalid_literal, content)
    return converted_content



def convert_py2_to_py3(content: str) -> str:
    """
    Convert Python 2 code to Python 3.
    """
    content = convert_print_statements(content)
    content = convert_except_statements(content)
    content = convert_xrange_to_range(content)
    content = convert_unicode_and_str(content)
    content = convert_dict_methods(content)
    content = convert_long_to_int(content)
    content = convert_exec_statements(content)
    content = convert_async_keyword(content)
    content = convert_long_integers(content)
    content = convert_octal_literals(content)
    content = convert_invalid_literals(content)
    content = convert_raise_statements(content)
    return content



@dataclass
class SearchResult:
    """Dataclass to hold search results."""

    file_path: str  # this is absolute path
    class_name: Optional[str]
    func_name: Optional[str]
    code: str

    def to_tagged_upto_file(self, project_root: str):
        """Convert the search result to a tagged string, upto file path."""
        rel_path = apputils.to_relative_path(self.file_path, project_root)
        file_part = f"<file>{rel_path}</file>"
        return file_part

    def to_tagged_upto_class(self, project_root: str):
        """Convert the search result to a tagged string, upto class."""
        prefix = self.to_tagged_upto_file(project_root)
        class_part = (
            f" <class>{self.class_name}</class>" if self.class_name is not None else ""
        )
        return f"{prefix}{class_part}"

    def to_tagged_upto_func(self, project_root: str):
        """Convert the search result to a tagged string, upto function."""
        prefix = self.to_tagged_upto_class(project_root)
        func_part = (
            f" <func>{self.func_name}</func>" if self.func_name is not None else ""
        )
        return f"{prefix}{func_part}"

    def to_tagged_str(self, project_root: str):
        """Convert the search result to a tagged string."""
        prefix = self.to_tagged_upto_func(project_root)
        code_part = f" <code>{self.code}</code>"
        return f"{prefix}{code_part}"

    @staticmethod
    def collapse_to_file_level(lst, project_root: str) -> str:
        """Collapse search results to file level."""
        res = dict()  # file -> count
        for r in lst:
            if r.file_path not in res:
                res[r.file_path] = 1
            else:
                res[r.file_path] += 1
        res_str = ""
        for file_path, count in res.items():
            rel_path = apputils.to_relative_path(file_path, project_root)
            file_part = f"<file>{rel_path}</file>"
            res_str += f"{file_part} ({count} matches)\n"
        return res_str

    @staticmethod
    def collapse_to_method_level(lst, project_root: str) -> str:
        """Collapse search results to method level."""
        res = dict()  # file -> dict(method -> count)
        for r in lst:
            if r.file_path not in res:
                res[r.file_path] = dict()
            func_str = r.func_name if r.func_name is not None else "Not in a function"
            if func_str not in res[r.file_path]:
                res[r.file_path][func_str] = 1
            else:
                res[r.file_path][func_str] += 1
        res_str = ""
        for file_path, funcs in res.items():
            rel_path = apputils.to_relative_path(file_path, project_root)
            file_part = f"<file>{rel_path}</file>"
            for func, count in funcs.items():
                if func == "Not in a function":
                    func_part = func
                else:
                    func_part = f" <func>{func}</func>"
                res_str += f"{file_part}{func_part} ({count} matches)\n"
        return res_str


def get_all_py_files(dir_path: str) -> List[str]:
    """Get all .py files recursively from a directory.

    Skips files that are obviously not from the source code, such third-party library code.

    Args:
        dir_path (str): Path to the directory.
    Returns:
        List[str]: List of .py file paths. These paths are ABSOLUTE path!
    """

    py_files = glob.glob(pjoin(dir_path, "**/*.py"), recursive=True)
    res = []
    for file in py_files:
        rel_path = file[len(dir_path) + 1 :]
        if rel_path.startswith("build"):
            continue
        if rel_path.startswith("doc"):
            # discovered this issue in 'pytest-dev__pytest'
            continue
        if rel_path.startswith("requests/packages"):
            # to walkaround issue in 'psf__requests'
            continue
        if (
            rel_path.startswith("tests/regrtest_data")
            or rel_path.startswith("tests/input")
            or rel_path.startswith("tests/functional")
        ):
            # to walkaround issue in 'pylint-dev__pylint'
            continue
        if rel_path.startswith("tests/roots") or rel_path.startswith(
            "sphinx/templates/latex"
        ):
            # to walkaround issue in 'sphinx-doc__sphinx'
            continue
        if rel_path.startswith("tests/test_runner_apps/tagged/") or rel_path.startswith(
            "django/conf/app_template/"
        ):
            # to walkaround issue in 'django__django'
            continue
        if "pytest" not in file:
            # 如果不是pytest类库，去除所有单测文件；
            if rel_path.startswith("test"):
                continue

        res.append(file)
    return res


def get_all_classes_in_file(file_full_path: str) -> List[Tuple[str, int, int]]:
    """Get all classes defined in one .py file.

    Args:
        file_path (str): Path to the .py file.
    Returns:
        List of classes in this file.
    """

    with open(file_full_path, "r", encoding="utf-8-sig") as f:
        file_content = f.read()

    classes = []
    # print(file_path)
    try: 
        tree = ast.parse(file_content)
    except:
        print('error_file_path:', file_full_path)
        new_file_content = convert_py2_to_py3(file_content)
        tree = ast.parse(new_file_content)
        print('now right_file_path:', file_full_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            start_lineno = node.lineno
            end_lineno = node.end_lineno
            # line numbers are 1-based
            classes.append((class_name, start_lineno, end_lineno))
    return classes



def get_top_level_functions_signatures(file_full_path: str):
    with open(file_full_path, "r", encoding="utf-8-sig") as f:
        file_content = f.read()

    with open(file_full_path, 'r', encoding="utf-8-sig") as f1:
        file_lines = f1.readlines()

    functions = []
    function_signatures = []
    
    try: 
        tree = ast.parse(file_content)
    except:
        # print('error_file_path:', file_full_path)
        new_file_content = convert_py2_to_py3(file_content)
        tree = ast.parse(new_file_content)
        # print('now right_file_path:', file_full_path)
        
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            start_lineno = node.lineno
            end_lineno = node.end_lineno

            # Adjust start line number to include all decorators
            if node.decorator_list:
                decorator_lines = [d.lineno for d in node.decorator_list]
                start_lineno = min(decorator_lines)

            # line numbers are 1-based
            # functions.append((function_name, start_lineno, end_lineno))
            functions.append(get_code_snippets(file_full_path, start_lineno, end_lineno))

            sig_lines = extract_func_sig_from_ast(node)
            func_signature = ""
            for line in sig_lines:
                func_signature += file_lines[line-1]
                # func_signature += "\n"
            function_signatures.append(func_signature)

    return functions, function_signatures


def get_top_level_functions_src(file_full_path: str) -> List[Tuple[str, int, int]]:
    """Get top-level functions defined in one .py file.

    This excludes functions defined in any classes.

    Args:
        file_path (str): Path to the .py file.
    Returns:
        List of top-level functions in this file.
    """
    with open(file_full_path, "r", encoding="utf-8-sig") as f:
        file_content = f.read()

    functions = []
    tree = ast.parse(file_content)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            start_lineno = node.lineno
            end_lineno = node.end_lineno
            # line numbers are 1-based
            functions.append((function_name, start_lineno, end_lineno))
    return functions


def get_top_level_functions(file_full_path: str) -> List[Tuple[str, int, int]]:
    """Get top-level functions defined in one .py file, including line numbers for decorators.
    This excludes functions defined in any classes.
    Args:
        file_path (str): Path to the .py file.
    Returns:
        List of tuples with (function_name, start_lineno, end_lineno).
    """
    with open(file_full_path, "r", encoding="utf-8-sig") as f:
        file_content = f.read()

    functions = []
    try: 
        tree = ast.parse(file_content)
    except:
        new_file_content = convert_py2_to_py3(file_content)
        tree = ast.parse(new_file_content)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            start_lineno = node.lineno
            end_lineno = node.end_lineno

            # Adjust start line number to include all decorators
            if node.decorator_list:
                decorator_lines = [d.lineno for d in node.decorator_list]
                start_lineno = min(decorator_lines)

            # line numbers are 1-based
            functions.append((function_name, start_lineno, end_lineno))

    return functions


# mainly used for building index
def get_all_funcs_in_class_in_file(
    file_full_path: str, class_name: str
) -> List[Tuple[str, int, int]]:
    """
    For a class in a file, get all functions defined in the class.
    Assumption:
        - the given function exists, and is defined in the given file.
    Returns:
        - List of tuples, each tuple is (function_name, start_lineno, end_lineno).
    """
    with open(file_full_path, "r", encoding="utf-8-sig") as f:
        file_content = f.read()

    functions = []
    try: 
        tree = ast.parse(file_content)
    except:
        new_file_content = convert_py2_to_py3(file_content)
        tree = ast.parse(new_file_content)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for n in ast.walk(node):
                if isinstance(n, ast.FunctionDef):
                    function_name = n.name
                    start_lineno = n.lineno
                    end_lineno = n.end_lineno
                    functions.append((function_name, start_lineno, end_lineno))

    return functions


def get_func_snippet_in_class(
    file_full_path: str, class_name: str, func_name: str, include_lineno=False
) -> Optional[str]:
    """Get actual function source code in class.

    All source code of the function is returned.
    Assumption: the class and function exist.
    """
    with open(file_full_path, "r", encoding="utf-8-sig") as f:
        file_content = f.read()

    # tree = ast.parse(file_content)
    try: 
        tree = ast.parse(file_content)
    except:
        new_file_content = convert_py2_to_py3(file_content)
        tree = ast.parse(new_file_content)
        
        
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for n in ast.walk(node):
                if isinstance(n, ast.FunctionDef) and n.name == func_name:
                    start_lineno = n.lineno
                    end_lineno = n.end_lineno
                    assert end_lineno is not None, "end_lineno is None"
                    if include_lineno:
                        return get_code_snippets_with_lineno(
                            file_full_path, start_lineno, end_lineno
                        )
                    else:
                        return get_code_snippets(
                            file_full_path, start_lineno, end_lineno
                        )
    # In this file, cannot find either the class, or a function within the class
    return None


def get_code_region_containing_code(
    file_full_path: str, code_str: str
) -> List[Tuple[int, str]]:
    """In a file, get the region of code that contains a specific string.

    Args:
        - file_full_path: Path to the file. (absolute path)
        - code_str: The string that the function should contain.
    Returns:
        - A list of tuple, each of them is a pair of (line_no, code_snippet).
        line_no is the starting line of the matched code; code snippet is the
        source code of the searched region.
    """
    with open(file_full_path, "r", encoding="utf-8-sig") as f:
        file_content = f.read()

    context_size = 3
    # since the code_str may contain multiple lines, let's not split the source file.

    # we want a few lines before and after the matched string. Since the matched string
    # can also contain new lines, this is a bit trickier.
    pattern = re.compile(re.escape(code_str))
    # each occurrence is a tuple of (line_no, code_snippet) (1-based line number)
    occurrences: List[Tuple[int, str]] = []
    for match in pattern.finditer(file_content):
        matched_start_pos = match.start()
        # first, find the line number of the matched start position (1-based)
        matched_line_no = file_content.count("\n", 0, matched_start_pos) + 1
        # next, get a few surrounding lines as context
        search_start = match.start() - 1
        search_end = match.end() + 1
        # from the matched position, go left to find 5 new lines.
        for _ in range(context_size):
            # find the \n to the left
            left_newline = file_content.rfind("\n", 0, search_start)
            if left_newline == -1:
                # no more new line to the left
                search_start = 0
                break
            else:
                search_start = left_newline
        # go right to fine 5 new lines
        for _ in range(context_size):
            right_newline = file_content.find("\n", search_end + 1)
            if right_newline == -1:
                # no more new line to the right
                search_end = len(file_content)
                break
            else:
                search_end = right_newline

        start = max(0, search_start)
        end = min(len(file_content), search_end)
        context = file_content[start:end]
        occurrences.append((matched_line_no, context))

    return occurrences


def get_func_snippet_with_code_in_file(file_full_path: str, code_str: str) -> List[str]:
    """In a file, get the function code, for which the function contains a specific string.

    Args:
        file_full_path (str): Path to the file. (absolute path)
        code_str (str): The string that the function should contain.

    Returns:
        A list of code snippets, each of them is the source code of the searched function.
    """
    with open(file_full_path, "r", encoding="utf-8-sig") as f:
        file_content = f.read()

    # tree = ast.parse(file_content)
    try: 
        tree = ast.parse(file_content)
    except:
        new_file_content = convert_py2_to_py3(file_content)
        tree = ast.parse(new_file_content)
        
    all_snippets = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        func_start_lineno = node.lineno
        func_end_lineno = node.end_lineno
        assert func_end_lineno is not None
        func_code = get_code_snippets(
            file_full_path, func_start_lineno, func_end_lineno
        )
        # This func code is a raw concatenation of source lines which contains new lines and tabs.
        # For the purpose of searching, we remove all spaces and new lines in the code and the
        # search string, to avoid non-match due to difference in formatting.
        stripped_func = " ".join(func_code.split())
        stripped_code_str = " ".join(code_str.split())
        if stripped_code_str in stripped_func:
            all_snippets.append(func_code)

    return all_snippets


def get_code_snippets_with_lineno(file_full_path: str, start: int, end: int) -> str:
    """Get the code snippet in the range in the file.

    The code snippet should come with line number at the beginning for each line.

    TODO: When there are too many lines, return only parts of the output.
          For class, this should only involve the signatures.
          For functions, maybe do slicing with dependency analysis?

    Args:
        file_path (str): Path to the file.
        start (int): Start line number. (1-based)
        end (int): End line number. (1-based)
    """
    with open(file_full_path, "r") as f:
        file_content = f.readlines()

    snippet = ""
    for i in range(start - 1, end):
        snippet += f"{i+1} {file_content[i]}"
    return snippet


def get_code_snippets(file_full_path: str, start: int, end: int) -> str:
    """Get the code snippet in the range in the file, without line numbers.

    Args:
        file_path (str): Full path to the file.
        start (int): Start line number. (1-based)
        end (int): End line number. (1-based)
    """
    with open(file_full_path, "r") as f:
        file_content = f.readlines()
    snippet = ""
    for i in range(start - 1, end):
        snippet += file_content[i]
    return snippet


def extract_func_sig_from_ast(func_ast: ast.FunctionDef) -> List[int]:
    """Extract the function signature from the AST node.

    Includes the decorators, method name, and parameters.

    Args:
        func_ast (ast.FunctionDef): AST of the function.

    Returns:
        The source line numbers that contains the function signature.
    """
    func_start_line = func_ast.lineno
    if func_ast.decorator_list:
        # has decorators
        decorator_start_lines = [d.lineno for d in func_ast.decorator_list]
        decorator_first_line = min(decorator_start_lines)
        func_start_line = min(decorator_first_line, func_start_line)
    # decide end line from body
    if func_ast.body:
        # has body
        body_start_line = func_ast.body[0].lineno
        end_line = body_start_line - 1
    else:
        # no body
        end_line = func_ast.end_lineno
    assert end_line is not None
    return list(range(func_start_line, end_line + 1))


def extract_class_sig_from_ast(class_ast: ast.ClassDef) -> List[int]:
    """Extract the class signature from the AST.

    Args:
        class_ast (ast.ClassDef): AST of the class.

    Returns:
        The source line numbers that contains the class signature.
    """
    # STEP (1): extract the class signature
    sig_start_line = class_ast.lineno
    if class_ast.body:
        # has body
        body_start_line = class_ast.body[0].lineno
        sig_end_line = body_start_line - 1
    else:
        # no body
        sig_end_line = class_ast.end_lineno
    assert sig_end_line is not None
    sig_lines = list(range(sig_start_line, sig_end_line + 1))

    # STEP (2): extract the function signatures and assign signatures
    for stmt in class_ast.body:
        if isinstance(stmt, ast.FunctionDef):
            sig_lines.extend(extract_func_sig_from_ast(stmt))
        elif isinstance(stmt, ast.Assign):
            # for Assign, skip some useless cases where the assignment is to create docs
            stmt_str_format = ast.dump(stmt)
            if "__doc__" in stmt_str_format:
                continue
            # otherwise, Assign is easy to handle
            assert stmt.end_lineno is not None
            assign_range = list(range(stmt.lineno, stmt.end_lineno + 1))
            sig_lines.extend(assign_range)

    return sig_lines


def get_class_signature(file_full_path: str, class_name: str) -> str:
    """Get the class signature.

    Args:
        file_path (str): Path to the file.
        class_name (str): Name of the class.
    """
    with open(file_full_path, "r", encoding="utf-8-sig") as f:
        file_content = f.read()

    # tree = ast.parse(file_content)
    try: 
        tree = ast.parse(file_content)
    except:
        new_file_content = convert_py2_to_py3(file_content)
        tree = ast.parse(new_file_content)
        
    relevant_lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            # we reached the target class
            relevant_lines = extract_class_sig_from_ast(node)
            break
    if not relevant_lines:
        return ""
    else:
        with open(file_full_path, "r") as f:
            file_content = f.readlines()
        result = ""
        for line in relevant_lines:
            line_content: str = file_content[line - 1]
            if line_content.strip().startswith("#"):
                # this kind of comment could be left until this stage.
                # reason: # comments are not part of func body if they appear at beginning of func
                continue
            result += line_content
        return result
