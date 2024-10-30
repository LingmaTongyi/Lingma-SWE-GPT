# reproduce inference
import inspect
from pathlib import Path
import json
import re
from os.path import join as pjoin
import os
from termcolor import colored

from app import globals
from app.api.manage import ProjectApiManager
from app.data_structures import (
    FunctionCallIntent,
    MessageThread,
)
import subprocess
from app.log import log_and_cprint, log_and_print
from app.model.gpt import call_gpt
from app.search.search_manage import SearchManager
from app.search.search_utils import get_all_py_files, get_top_level_functions, get_all_classes_in_file, get_all_funcs_in_class_in_file, get_code_snippets
from app.utils import parse_function_invocation
import app.utils as apputils
import glob
from app.api.agent_locate_file_class_func import get_top_files_from_bm25, get_top_files_from_llm_prompt, is_valid_location_json, get_top_content_from_llm_prompt, SUMMARY_PROMPT


SYSTEM_PROMPT = """You are a senior software developer maintaining a large project.
You are working on an issue submitted to your project.
The issue contains a description marked between <issue> and </issue>.
Your task is to invoke a few search API calls to gather buggy information, then locate the code fragments that need to be modified based on the collected information, and finally write patches to solve the issues.
Please note that the problem-solving process needs to be carried out carefully and systematically, ensuring that you correctly understand the content of the issue, ensuring that you collect enough information to help you locate the bug location, and finally ensuring that the patch you generate is reliable and effective. It is very important that your fixes are correct, otherwise may affect the normal operation of the software.
"""

REVIEW_REPRODUCE_CODE_PROMPT = """You are a senior quality assurance engineer. Your main responsibility is to review whether the reproduction code reproduces the bug in the issue.
Given an issue, the reproduction code written by the developer, and the log information you get when running the reproduction code, read the issue and output log carefully, and determine whether the given reproduction code reproduces the issue. 
If the program runs normally and the output log includes the output of “before” and “after”, then the reproduction code is likely to reproduce the bug.
Regardless of whether it reproduces or not, please inform the developer of your review results at the end. 
Please think step by step first, output detailed reasoning steps in the review, and then draw a conclusion in the last line of the review "Does the given code reproduce the bug: str", ONLY ANSWER YES/NO.

Issue: {issue}

Reproduction code:
```
{code}
```

Output log:
```
{log}
```

Let's review step by step now.
"""


REVIEW_RESOLVED_PROMPT = """You are a senior quality assurance engineer, and your main responsibility is to review whether the patch solves the issue.
Given an issue, a patch written by a developer, and the log information when you run the reproduce code in the issue before and after applying the patch. 
Please read the issue and the output log carefully, and determine whether the given patch solves the problem. 
Regardless of whether it solves the problem or not, please inform the developer of your review results at the end. 
Please think step by step first, output detailed reasoning steps in the review, and then draw a conclusion in the last line of the review "Does the given patch resolve the bug: str", ONLY ANSWER YES/NO.

Issue: {issue}

Patch:
```
{code}
```

Reproduce code:
```
{rcode}
```

Output log before apply patch:
```
{blog}
```

Output log after apply patch:
```
{alog}
```

Let's review step by step now.
"""

REVIEW_RESOLVED_PROMPT_short = """Let's review the run logs when you run the reproduce code before and after applying the patch carefully to determine whether the patch solves the issue.
Issue: {issue}

Patch:
```
{code}
```

Reproduce code:
```
{rcode}
```

Output log before apply patch:
```
{blog}
```

Output log after apply patch:
```
{alog}
```
"""

REVIEW_PROMPT = """You are a senior quality assurance engineer, and your main responsibility is to review patch codes submitted by developers.
Given an issue, a developer's analysis of collected context, and a developer-submitted patch in the repository, carefully read and understand the issue and determine whether the given patch correctly addresses the issue. If it is solved, please pass this patch; otherwise, please provide detailed reasons.
Please refer to the above responsibilities and provide detailed reasoning and analysis. Then draw the conclusion "Code Review Result: str" on the last line, ONLY ANSWER ACCEPT/REJECT.

# Example
## Issue: --disable_progress_bar Flag Broken for Fix\n### What Happened\n\nI ran `sqlfluff fix $target --dialect ansi --disable_progress_bar --force` on version 1.4.0 and got an error with exit code 2. Running with `--disable-progress-bar` appears to work fine, but it appears that compatibility with underscores was broken in version 1.4.0.\n\n### Expected Behaviour\n\nShould run as expected, with no error and no progress bar.\n\n### Observed Behaviour\n\nExit code 2 and stderr:\r\n```\r\nUsage: sqlfluff fix [OPTIONS] [PATHS]...\r\n        Try 'sqlfluff fix -h' for help.\r\n\r\n        Error: No such option: --disable_progress_bar (Possible options: --disable-noqa, --disable-progress-bar)\r\n```\n\n### How to reproduce\n\nSql file:\r\n```\r\nSELECT foo FROM bar;\r\n```\r\n\r\nCommand:\r\n```\r\nsqlfluff fix $target --dialect ansi --disable_progress_bar --force\r\n```\n\n### Dialect\n\nansi\n\n### Version\n\npython 3.10.3\r\nsqlfluff 1.4.0 and up appears to have this problem (tested through 1.4.2)\n\n### Configuration\n\nNo special configuration. Ran hermetically with `trunk`.\n\n### Are you willing to work on and submit a PR to address the issue?\n\n- [ ] Yes I am willing to submit a PR!\n\n### Code of Conduct\n\n- [X] I agree to follow this project's Code of Conduct\n\n
## Patch to be Reviewed:
```Code
diff --git a/src/sqlfluff/cli/commands.py b/src/sqlfluff/cli/commands.py
--- a/src/sqlfluff/cli/commands.py
+++ b/src/sqlfluff/cli/commands.py
@@ -680,7 +680,7 @@ def do_fixes(lnt, result, formatter=None, **kwargs):
     return False  # pragma: no cover


-@cli.command()
+@cli.command(cls=DeprecatedOptionsCommand)
 @common_options
 @core_options
 @click.option(
@@ -710,9 +710,12 @@ def do_fixes(lnt, result, formatter=None, **kwargs):
     ),
 )
 @click.option(
+    "--disable_progress_bar",
     "--disable-progress-bar",
     is_flag=True,
     help="Disables progress bars.",
+    cls=DeprecatedOption,
+    deprecated=["--disable_progress_bar"],
 )
 @click.option(
     "--FIX-EVEN-UNPARSABLE",
```

## Analysis of collected context:
### Analysis of the Issue:
The user is experiencing an issue with the sqlfluff tool, specifically when running the fix command with the --disable_progress_bar flag. In the context provided, the user mentions that while using the flag with underscores (--disable_progress_bar), they encounter an error with exit code 2. However, when they switch to using hyphens (--disable-progress-bar), the command works as expected. The error message suggests that the flag with underscores is no longer recognized by sqlfluff in versions 1.4.0 and above.
### Proposed Solution:
Identify the exact commit or change where the --disable_progress_bar option stopped accepting the underscore variant.
Determine whether this was an intentional change or a regression.
If it was an unintentional regression, fix the command-line interface to accept both hyphen and underscore variants for the --disable_progress_bar flag.
Include unit tests to ensure that both flag variants work and to prevent future regressions.
Update documentation and help messages to reflect the supported flag formats.
The maintainers of the sqlfluff project will need to assess whether to restore compatibility with the underscore variant of the flag or to update their documentation to clearly specify that the hyphenated flag is the only supported format moving forward. It's also possible that the issue arises from a third-party dependency or a change in how command-line arguments are parsed, which may require changes in how sqlfluff handles its command-line interface.

## Thought:
The patch submitted introduces the use of the DeprecatedOptionsCommand class and the DeprecatedOption decorator, which are applied to handle both the --disable_progress_bar and --disable-progress-bar options. Specifically, it allows the command to recognize --disable_progress_bar as a deprecated format, redirecting it to the correct --disable-progress-bar option. This change effectively addresses the reported issue by allowing backward compatibility while guiding users towards the preferred flag format with hyphens.

Detailed Analysis:
The patch adds support for the --disable_progress_bar option explicitly, marked as deprecated, which is crucial for maintaining backward compatibility with scripts that still use the underscore version.
By implementing the DeprecatedOption decorator, the patch provides a mechanism to alert users that --disable_progress_bar is deprecated, encouraging them to switch to the hyphenated version, --disable-progress-bar.
The introduction of DeprecatedOptionsCommand ensures that both versions of the flag trigger the same functionality without causing an error, thus aligning with user expectations for functionality regardless of flag format.
This solution not only resolves the immediate error encountered by users of version 1.4.0 and later but also enhances the robustness of the command-line interface by gracefully handling deprecated options.
The patch effectively resolves the problem described in the issue by restoring functionality to the --disable_progress_bar flag while maintaining clear communication about the deprecation of this format.

## Code Review Result: ACCEPT

## Issue: pvlib.irradiance.reindl() model generates NaNs when GHI = 0\n**Describe the bug**\r\nThe reindl function should give zero sky diffuse when GHI is zero. Instead it generates NaN or Inf values due to \"term3\" having a quotient that divides by GHI.  \r\n\r\n**Expected behavior**\r\nThe reindl function should result in zero sky diffuse when GHI is zero.\r\n\r\n\npvlib.irradiance.reindl() model generates NaNs when GHI = 0\n

## Patch to be Reviewed:
```Code
@@ -886,8 +886,9 @@ def reindl(surface_tilt, surface_azimuth, dhi, dni, ghi, dni_extra,
     # these are the () and [] sub-terms of the second term of eqn 8
     term1 = 1 - AI
     term2 = 0.5 * (1 + tools.cosd(surface_tilt))
-    term3 = 1 + np.sqrt(HB / ghi) * (tools.sind(0.5 * surface_tilt) ** 3)
-
+    if ghi == 0:
+        term3 = 1 + np.sqrt(HB) * (tools.sind(0.5 * surface_tilt)**3)
+    else:
+        term3 = 1 + np.sqrt(HB / ghi) * (tools.sind(0.5 * surface_tilt)**3)
     sky_diffuse = dhi * (AI * Rb + term1 * term2 * term3)
     sky_diffuse = np.maximum(sky_diffuse, 0)
```

## Analysis of collected context:
Based on the provided context, the issue with the reindl function in the pvlib library is that it generates NaNs (Not a Number) or Inf (Infinity) values when the Global Horizontal Irradiance (GHI) is zero. This problem arises because the equation for term3 involves a division by GHI, leading to undefined values when GHI is zero.

The expected behavior is that the sky diffuse component should be zero when GHI is zero, as there is no available solar radiation to diffuse.

### Proposed Fix
To fix this issue, we need to ensure that when ghi is zero, term3 should be set in a way that avoids division by zero. A common approach is to handle this case explicitly by setting term3 to a value that logically makes sense when ghi is zero, typically resulting in zero sky diffuse radiation.

## Thought:
### Issue Identification
Incorrect Handling of term3 When ghi is Zero:

The patch sets term3 to 1 + np.sqrt(HB) * (tools.sind(0.5 * surface_tilt)**3) when ghi is zero.
This is problematic because term3 should logically be zero when ghi is zero, meaning there is no global horizontal irradiance to contribute to the diffuse irradiance.
### Mismatch in Expected Behavior:

When ghi is zero, term3 should be zero to avoid contributing any value to sky_diffuse. The current patch fails to achieve this.

## Code Review Result: REJECT

# Now the issue is:
## Issue: {issue}

## Patch to be Reviewed:
```
{patch}
```
## Analysis of collected context:
{analysis}
## Thought:
"""
REVIEW_PROMPT = """You are a senior quality assurance engineer, and your main responsibility is to review patch codes submitted by developers.
Given an issue, a developer-submitted patch in the repository, carefully read and understand the issue and determine whether the given patch correctly addresses the issue. If it is solved, please accept this patch; otherwise, please provide detailed reasons.
Please refer to the above responsibilities and provide detailed reasoning and analysis. Then draw the conclusion "Code Review Result: (ACCEPT|REJECT)" on the last line, ONLY ANSWER ACCEPT or REJECT.

# Now the issue is:
## Issue: {issue}

## Patch to be Reviewed:
```
{patch}
```
"""

SUMMARY = """Based on the issue and possible buggy codes, develop a comprehensive solution strategy. Your plan should include the following sections:

1. **Precise Fault Location**:
   - Infer the exact location that needs modification based on issue and possible fault locations.
   - Ensure the location is as specific and minimal as as possible to minimize the risk of new errors and facilitate easy review and testing.

2. **Solution Description and Rationale**:
   - Outline the steps necessary to resolve the issue.
   - Explain why this solution is appropriate, including relevant details and supporting evidence.

Note: You do not need to write a specific code patch or describe a test plan. Senior developers will implement the solution and perform testing using the existing test suite.

Now, let's think through this step by step.
"""

SUMMARY_SHORT = """Please plan your solution and describe its rationale based on the issue and possible buggy codes"""
def get_review_result(text):
    import re
    pattern = r"Does the given code reproduce the bug: (YES/NO)"
    try:
        matches = re.findall(pattern, text)
        result = matches[0]
    except:
        if 'YES' in text:
            result = 'YES'
        else:
            result = 'NO'
    return result


def prepare_issue_prompt(problem_stmt: str) -> str:
    """
    Given the raw problem statement, sanitize it and prepare the issue prompt.
    Args:
        problem_stmt (str): The raw problem statement.
            Assumption: the problem statement is the content of a markdown file.
    Returns:
        str: The issue prompt.
    """
    # remove markdown comments
    problem_wo_comments = re.sub(r"<!--.*?-->", "", problem_stmt, flags=re.DOTALL)
    content_lines = problem_wo_comments.split("\n")
    # remove spaces and empty lines
    content_lines = [x.strip() for x in content_lines]
    content_lines = [x for x in content_lines if x != ""]
    problem_stripped = "\n".join(content_lines)
    # add tags
    result = "<issue>" + problem_stripped + "\n</issue>"
    return result


def add_step_trigger(orig_prompt: str, is_first: bool = False) -> str:
    """
    Given the original prompt, add the trigger question for the next step.
    Args:
        orig_prompt (str): The original prompt.
        is_first (bool): Whether the trigger is for the first step.
    Returns:
        str: The prompt with trigger question.
    """
    if is_first:
        trigger = "What is the first step?"
    else:
        trigger = "What's the next step to complete the task? Be reminded that you are solving the initial issue."
    return orig_prompt + "\n" + trigger


def file_in_directory(directory, file_name):
    for filename in os.listdir(directory):
        if file_name in filename:
            return os.path.join(directory, filename)
    return None


def cal_location_rate(bug_locations, oracle_locations):
    """
    1. 总体计算
        - 文件定位精度
        - 函数定位精度
        - 片段定位精度
    2. 返回类型: {"file:" %, "function": %, "line": %, "function_line": %}
    """
    def cal_line_rate(line_content, oracle_line_content):
        line_content = line_content.replace('<code_snippets>', '').replace('</code_snippets>', '')
        oracle_line_content = oracle_line_content.replace('<code_snippets>', '').replace('</code_snippets>', '')
        line_content = line_content.split('\n')
        oracle_line_content = oracle_line_content.split('\n')

        for oracle_line in oracle_line_content:
            if oracle_line == '' or len(oracle_line) < 5 or oracle_line == '\n':
                continue
            else:
                if oracle_line in line_content:
                    return 1
        return 0

    file_rate, function_rate, line_rate = 0, 0, 0
    update_file, update_function, update_line = [], [], []
    for orcale_location in oracle_locations:
        oracle_file_name = orcale_location['file_name']
        update_file.append(oracle_file_name)
        oracle_bug_locations = orcale_location['bug_locations']
        for oracle_bug_location in oracle_bug_locations:
            if "<function>" in oracle_bug_location:
                update_function.append(oracle_bug_location)
            elif "<class>" in oracle_bug_location:
                update_function.append(oracle_bug_location)
            elif "<code_snippets>" in oracle_bug_location:
                update_line.append(oracle_bug_location)

    for bug_location in bug_locations:
        file_name = bug_location['file_name']
        if file_name in update_file or file_name == update_file:
            file_rate += 1
        bug_locations_item = bug_location['bug_locations']
        for bug_location_item in bug_locations_item:
            if "<function>" in bug_location_item and bug_location_item in update_function:
                function_rate += 1
            elif "<class>" in bug_location_item and bug_location_item in update_function:
                function_rate += 1
            elif "<code_snippets>" in bug_location_item:
                for update_line_item in update_line:
                    if cal_line_rate(bug_location_item, update_line_item):
                        line_rate += 1
                        break

    file_rate_percent = file_rate / len(update_file) if len(update_file) > 0 else -1
    function_rate_percent = function_rate / len(update_function) if len(update_function) > 0 else -1
    line_rate_percent = line_rate / len(update_line) if len(update_line) > 0 else -1
    function_line_rate_percent = (function_rate + line_rate) / (len(update_function) + len(update_line)) if (len(update_function) + len(update_line)) > 0 else -1
    return {"file": file_rate_percent, "function": function_rate_percent, "line": line_rate_percent,
             "function_line": function_line_rate_percent}


def get_location_from_agent_proxy(output_dir):
    final_locations = []
    temp_map = {}
    files = [f for f in os.listdir(output_dir) if f.startswith("agent_proxy_")]
    if files:
        max_file = max(files, key=lambda x: int(x.split("_")[2].split(".")[0]))
        with open(os.path.join(output_dir, max_file), "r") as f:
            data = json.load(f)
            if data[0][-1]['role'] == 'assistant':
                try:
                    locations = json.loads(data[0][-1]['content'])['bug_locations']
                except:
                    return final_locations
            else:
                return final_locations
            
            for location in locations:
                if 'method' in location.keys():
                    bug_location_str = f"<function>{location['method']}</function>"
                elif 'class' in location.keys():
                    bug_location_str = f"<class>{location['class']}</class>"
                else:
                    bug_location_str = "none"

                if 'file' in location.keys():
                    file_name = location['file']
                else:
                    file_name = 'unknown'

                if file_name not in temp_map.keys():
                    temp_map[file_name] = []
                temp_map[file_name].append(bug_location_str)

            for key, value in temp_map.items():
                final_locations.append({"file_name": key, "bug_locations": value})
            
            return final_locations
        
    else:
        return final_locations


def get_location_from_agent_repo_review(output_dir):
    final_locations = []
    temp_map = {}
    with open(os.path.join(output_dir, "agent_specific_content_location.json"), "r") as f:
        data = json.load(f)
        if data[-1]['role'] == 'assistant':
            locations = json.loads(data[-1]['content'])['bug_locations']
        else:
            return final_locations
            
        for location in locations:
            if 'method' in location.keys():
                bug_location_str = f"<function>{location['method']}</function>"
            elif 'class' in location.keys():
                bug_location_str = f"<class>{location['class']}</class>"
            elif "code_snippets" in location.keys():
                bug_location_str = f"<code_snippets>{location['code_snippets']}</code_snippets>"
            else:
                bug_location_str = "none"

            if 'file' in location.keys():
                file_name = location['file']
            else:
                file_name = 'unknown'

            if file_name not in temp_map.keys():
                temp_map[file_name] = []
            temp_map[file_name].append(bug_location_str)

        for key, value in temp_map.items():
            final_locations.append({"file_name": key, "bug_locations": value})
        
        return final_locations
        


def get_locations_from_patch(patch_content, repo_root_path):
    def split_patch(patch_content):
        file_patches = re.split(r'(?=diff --git)', patch_content.strip())

        result = {}
        for file_patch in file_patches:
            if file_patch.strip():
                match = re.search(r'diff --git a/(.*?) b/', file_patch)
                if match:
                    file_path = match.group(1)
                    result[file_path] = file_patch.strip()

        return result

    def parse_patch(patch, swebench_style=True):
        lines = patch.split('\n')

        subpatches = []
        line_numbers = []
        current_subpatch = []

        # 匹配@@ -x,y +a,b @@格式的正则表达式
        header_pattern = re.compile(r'^@@ -(\d+),(\d+) \+(\d+),(\d+) @@')

        for line in lines:
            match = header_pattern.match(line)
            if match:
                if current_subpatch:
                    subpatches.append('\n'.join(current_subpatch))
                    current_subpatch = []

                old_start, old_count, new_start, new_count = map(int, match.groups())
                line_numbers.append([old_start, old_count])

            current_subpatch.append(line)

        if current_subpatch:
            subpatches.append('\n'.join(current_subpatch))

        if swebench_style:
            subpatches = subpatches[1:]

        if len(line_numbers) != len(subpatches):
            print(f'line_numbers and subpatches should have same length! Lines: {len(line_numbers)}')
            return [], []
        return line_numbers, subpatches
    
    def extract_edit_lines(numbers, subpatch):
        edit_lines = []
        original_start = numbers[0]
        original_count = numbers[1]
        current_pos = original_start
        subpatches = subpatch.split('\n')
        for i, line in enumerate(subpatches):
            if line.startswith('-'):
                current_pos += 1
                edit_lines.append(current_pos)
            elif line.startswith('+'):
                edit_lines.append(current_pos)
            elif line.startswith(' ') or len(line) == 0:
                current_pos += 1
            elif line.startswith('@'):
                current_pos -= 1
            else:
                print(len(line))
                print('Invalid line: {}'.format(line))
                # raise ValueError('Invalid line: {}'.format(line))
        edit_lines = list(set(edit_lines))
        return edit_lines
    
    sub_file_patch = split_patch(patch_content)
    file_and_numbers = []

    for file_path, sub_patch in sub_file_patch.items():
        line_numbers, subpatches = parse_patch(sub_patch)
        edit_lines = []
        for i in range(len(line_numbers)):
            edit_lines.extend(extract_edit_lines(line_numbers[i], subpatches[i]))
        file_and_numbers.append([file_path, line_numbers, subpatches, edit_lines])

    if not os.path.exists(repo_root_path):
        return []

    py_files = get_all_py_files(repo_root_path)
    not_found_nums = 0
    bug_locations = []

    # 遍历每个修改位置
    for file_and_number in file_and_numbers:
        abs_file_path = ''
        file_name = file_and_number[0]
        subpatches = file_and_number[2]
        edit_lines = file_and_number[3]
        
        for py_file in py_files:
            if file_name in py_file:
                abs_file_path = py_file
                break
        if abs_file_path == '':
            continue

        try:
            classes = get_all_classes_in_file(abs_file_path)
            top_functions = get_top_level_functions(abs_file_path)
            class_functions = []

            for class_name in classes:
                class_funcs = get_all_funcs_in_class_in_file(abs_file_path, class_name[0])
                class_functions.extend(class_funcs)
            all_functions = []
            all_functions.extend(top_functions)
            all_functions.extend(class_functions)
        except:
            classes = []
            all_functions = []

        # 根据edit lineno找到具体的故障位置；
        bug_locations_in_file = []
        for edit_line in edit_lines:
            flag = False
            # 先在函数范围内找，找到包围lineno的函数；
            for function in all_functions:
                function_name = function[0]
                start_lineno = function[1]
                end_lineno = function[2]
                if edit_line >= start_lineno and edit_line <= end_lineno:
                    temp_bug_location = f"<function>{function_name}</function>"
                    flag = True
                    if temp_bug_location not in bug_locations_in_file:
                        bug_locations_in_file.append(temp_bug_location)
                    break

            if flag:
                continue
            # 如果没有找到，再在类范围内找；
            for class_item in classes:
                class_name = class_item[0]
                start_lineno = class_item[1]
                end_lineno = class_item[2]
                if edit_line >= start_lineno and edit_line <= end_lineno:
                    temp_bug_location = f"<class>{class_name}</class>"
                    flag = True
                    if temp_bug_location not in bug_locations_in_file:
                        bug_locations_in_file.append(temp_bug_location)
                    break

            if flag:
                continue
            # 如果都没有找到，需要在全局里去搜索；
            edit_line_before = edit_line - 1
            edit_line_after = edit_line + 1
            # 防止超过范围；
            edit_line_before = max(edit_line_before, 1)
            try:
                with open(abs_file_path, 'r') as f:
                    lines = f.readlines()
                    edit_line_after = min(edit_line_after, len(lines))
            except:
                edit_line_after = edit_line
            code_snippets = get_code_snippets(abs_file_path, edit_line_before, edit_line_after)
            if code_snippets:
                temp_bug_location = f"<code_snippets>{code_snippets}</code_snippets>"
                if temp_bug_location not in bug_locations_in_file:
                    bug_locations_in_file.append(temp_bug_location)
            else:
                print('[error]: no code_snippets found!')
                print(edit_line_before, edit_line, edit_line_after, abs_file_path)
                not_found_nums += 1

        # 保存每个file里的bug locations
        bug_locations_in_file_json = {
            'file_name': file_name,
            'abs_file_path': abs_file_path,
            'bug_locations': bug_locations_in_file,
            'bug_locations_edit_lineno': edit_lines
        }
        bug_locations.append(bug_locations_in_file_json)

    return bug_locations


def get_location_from_extracted_patch(output_dir, repo_root_path):
    final_locations = []
    files = [f for f in os.listdir(output_dir) if f.startswith("extracted_patch_")]
    if files:
        max_file = max(files, key=lambda x: int(x.split("_")[2].split(".")[0]))
        with open(os.path.join(output_dir, max_file), "r") as f:
            diff_patch = f.read()
        final_locations = get_locations_from_patch(diff_patch, repo_root_path)
        return final_locations
    else:
        return final_locations
    
    
def start_conversation_round_stratified(
        output_dir: str,
        msg_thread: MessageThread,
        api_manager: ProjectApiManager,
        issue: str = '',
        start_round_no: int = 0,
) -> bool:
    """
    This version uses json data to process API calls, instead of using the OpenAI function calling.
    Advantage is that multiple API calls can be made in a single round.
    """
    logger = api_manager.logger
    msg_thread.add_user(
        """Based on the files, classes, methods, code statements from the issue that related to the bug, you can use below search APIs to get more context of the project.
        search_class(class_name: str): Search for a class in the codebase.
        search_method_in_file(method_name: str, file_path: str): Search for a method in a given file.
        search_method_in_class(method_name: str, class_name: str): Search for a method in a given class.
        search_method(method_name: str): Search for a method in the entire codebase.
        search_code(code_str: str): Search for a code snippet in the entire codebase.
        search_code_in_file(code_str: str, file_path: str): Search for a code snippet in a given file file.
        Note that you can use multiple search APIs in one round.
        Now analyze the issue and select necessary APIs to get more context of the project, each API call must have concrete arguments as inputs.
        """
    )
    record_analyze = ""
    round_no = start_round_no
    for round_no in range(start_round_no, globals.conv_round_limit + 1):
        api_manager.start_new_tool_call_layer()

        conversation_file = pjoin(output_dir, f"conversation_round_{round_no}.json")
        # save current state before starting a new round
        msg_thread.save_to_file(conversation_file)

        log_and_cprint(
            logger,
            f"\n========== Conversation Round {round_no} ==========",
            "red",
            attrs=["bold"],
        )
        log_and_print(
            logger, f"{colored('Current message thread:', 'green')}\n{msg_thread}"
        )

        res_text, _, _, cost, input_tokens, output_tokens = call_gpt(
            logger, msg_thread.to_msg()
        )
        api_manager.accumulate_cost_and_tokens(cost, input_tokens, output_tokens)

        # print("raw API selection output", res_text)

        msg_thread.add_model(res_text, tools=[])

        selected_apis, _, proxy_threads = api_manager.proxy_apis(res_text)

        proxy_log = Path(output_dir, f"agent_proxy_{round_no}.json")
        proxy_messages = [thread.to_msg() for thread in proxy_threads]
        proxy_log.write_text(json.dumps(proxy_messages, indent=4))

        if selected_apis is None:
            msg_thread.add_user(
                "The search API calls seem not valid. Please check the arguments you give carefully and try again."
            )
            continue

        selected_apis_json = json.loads(selected_apis)
        json_api_calls = selected_apis_json.get("API_calls", [])
        buggy_locations = selected_apis_json.get("bug_locations", [])

        # collected enough information to write patch
        if buggy_locations and (not json_api_calls):
            collated_tool_response = "Here are the code in buggy locations:\n"
            # provide the buggy locations to the model
            for bug_location in buggy_locations:
                tool_output, *_ = search_for_bug_location(
                    api_manager, msg_thread, bug_location
                )
                collated_tool_response += f"{tool_output}\n"

            if (
                    "Unknown function" not in collated_tool_response
                    and "Could not" not in collated_tool_response
            ):
                msg_thread.add_user(collated_tool_response)
                break

            msg_thread.add_user(
                "The buggy locations is not precise. You may need to check whether the arguments are correct and search more information."
            )
            continue

        # prepare response from tools
        collated_tool_response = ""

        for api_call in json_api_calls:
            func_name, func_args = parse_function_invocation(api_call, logger)

            arg_spec = inspect.getfullargspec(getattr(SearchManager, func_name))
            arg_names = arg_spec.args[1:]  # first parameter is self

            assert len(func_args) == len(
                arg_names
            ), f"Number of argument is wrong in API call: {api_call}"

            kwargs = dict(zip(arg_names, func_args))
            intent = FunctionCallIntent(func_name, kwargs, None)
            tool_output, _, _ = api_manager.dispatch_intent(intent, msg_thread)

            collated_tool_response += f"Result of {api_call}:\n"
            collated_tool_response += tool_output + "\n\n"

        msg_thread.add_user(collated_tool_response)
        msg_thread.add_user(
            "To help diagnose and fix issues in software repositories, let's systematically analyze the collected context step by step first. "
            "If the context collected is not sufficient to resolve the issue, please clearly state the lack of necessary information and plan how to collect in next steps.")

        res_text, _, _, cost, input_tokens, output_tokens = call_gpt(
            logger, msg_thread.to_msg()
        )
        # record_analyze = res_text
        api_manager.accumulate_cost_and_tokens(cost, input_tokens, output_tokens)
        msg_thread.add_model(res_text, tools=[])

        if round_no < globals.conv_round_limit:
            msg_thread.add_user(
                "Based on your analysis, please answer below questions carefully:\n"
                "  - do we need more context: construct search API calls to get more context of the project. (leave it empty if you don't need more context)\n"
                "  - where are bug locations: buggy files and methods. (leave it empty if you don't have enough information)\n"
                "Be sure to provide detailed explanations and arguments when answering questions so the team can understand your reasoning and the basis for your decisions."
            )

    else:
        log_and_print(logger, "Too many rounds. Try writing patch anyway.")

    round_no += 1

    api_manager.start_new_tool_call_layer()

    api_manager.logger.info("Gathered enough information. Invoking write_patch.")

    review_round_no = 0
    early_stop = False
    while review_round_no < 1:
        review_round_no += 1
        if early_stop:
            break
        
        print(f'get_location_from_agent_proxy begin')
        bug_locations_from_agent_proxy = get_location_from_agent_proxy(api_manager.output_dir)
        print(f'get_location_from_agent_proxy: {len(bug_locations_from_agent_proxy)}')
        with open(os.path.join(api_manager.output_dir, "locations_from_agent_proxy.json"), 'w') as f:
            json.dump(bug_locations_from_agent_proxy, f, indent=4)
            
        
        write_patch_intent = FunctionCallIntent("write_patch", {}, None)
        api_manager.dispatch_intent(write_patch_intent, msg_thread)

        conversation_file = pjoin(output_dir, f"conversation_round_{round_no}.json")
        msg_thread.save_to_file(conversation_file)

        print(f'get_location_from_extracted_patch begin')
        bug_locations_from_extracted_patch = get_location_from_extracted_patch(api_manager.output_dir, api_manager.project_path)
        print(f'get_location_from_extracted_patch: {len(bug_locations_from_extracted_patch)}')
        with open(os.path.join(api_manager.output_dir, "locations_from_extracted_patch.json"), 'w') as f:
            json.dump(bug_locations_from_extracted_patch, f, indent=4)
        
        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        location_map_path = os.path.join(script_dir, '..', 'SWE-bench', 'setup_result', 'location_map.json')
        location_map_path = os.path.normpath(location_map_path)

        with open(location_map_path, 'r') as f:
            location_map = json.load(f)
        if api_manager.task_id in location_map.keys():
            oracle_location = location_map[api_manager.task_id]
        else:
            oracle_location = []
        
        location_rate_from_patch = cal_location_rate(bug_locations_from_extracted_patch, oracle_location)
        location_rate_from_agent_proxy = cal_location_rate(bug_locations_from_agent_proxy, oracle_location)  
        with open(os.path.join(api_manager.output_dir, "locations_rate.json"), 'w') as f:
            json.dump({'from_patch': location_rate_from_patch, 'from_agent_proxy': location_rate_from_agent_proxy}, f, indent=4)

        #sample patch
        if globals.multi_patch_sample:
            source_temperature = globals.model_temperature
            globals.model_temperature = 0.8
            for sample_iter in range(globals.sample_size):
                pattern = os.path.join(api_manager.output_dir, 'extracted_patch_*.diff')
                files = glob.glob(pattern)
                if len(files)>0:
                    break
                    
                write_patch_intent = FunctionCallIntent("write_patch", {}, None)
                api_manager.dispatch_intent(write_patch_intent, msg_thread)
                print(f'Write patch sample_iter: {sample_iter}')
            globals.model_temperature = source_temperature
        
    log_and_print(logger, "Invoked write_patch. Ending workflow.")
    return True


def list_directories(path):
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    directories_short = [dir.split("_2024")[0] for dir in directories]
    return directories, directories_short


def start_conversation_round_from_cache(
        output_dir: str,
        msg_thread: MessageThread,
        api_manager: ProjectApiManager,
        issue: str = '',
        start_round_no: int = 0,
) -> bool:
    """
    Temporarily unavailable
    """
    
    return True

def search_for_bug_location_path_summary(bug_location):
    file_name = bug_location.get("file")
    method_name = bug_location.get("method")
    class_name = bug_location.get("class")
    
    summary = ""
    
    if method_name and class_name and file_name:
        summary = f"<file>{file_name}</file> <class>{class_name}</class> <method>{method_name}</method>"
        return summary

    if method_name and file_name:
        summary = f"<file>{file_name}</file> <method>{method_name}</method>"
        return summary

    if class_name and file_name:
        summary = f"<file>{file_name}</file> <class>{class_name}</class>"
        return summary

    return summary        

def search_for_bug_location(
        api_manager: ProjectApiManager,
        msg_thread: MessageThread,
        bug_location: dict[str, str],
) -> tuple[str, str, bool]:
    found = False

    file_name = bug_location.get("file")
    method_name = bug_location.get("method")
    class_name = bug_location.get("class")

    assert method_name or class_name, f"Invalid bug location: {bug_location}"

    call_result = None

    def call_function(func_name: str, kwargs: dict[str, str]) -> None:
        nonlocal found, call_result

        intent = FunctionCallIntent(func_name, kwargs, None)
        call_result = api_manager.dispatch_intent(intent, msg_thread)
        _, _, call_is_ok = call_result
        found |= call_is_ok

    if (not found) and method_name and class_name:
        kwargs = {
            "method_name": method_name,
            "class_name": class_name,
        }
        call_function("search_method_in_class", kwargs)

    if (not found) and method_name and file_name:
        kwargs = {
            "method_name": method_name,
            "file_name": file_name,
        }
        call_function("search_method_in_file", kwargs)

    if (not found) and class_name and file_name:
        kwargs = {
            "class_name": class_name,
            "file_name": file_name,
        }
        call_function("search_class_in_file", kwargs)

    if (not found) and class_name:
        kwargs = {"class_name": class_name}
        call_function("get_class_full_snippet", kwargs)

    if (not found) and method_name:
        kwargs = {"method_name": method_name}
        call_function("search_method", kwargs)

    assert call_result

    return call_result


def dump_tool_call_layers_to_file(
        tool_call_layers: list[dict], output_dir: str
) -> None:
    """Dump the layers of tool calls to a file."""
    tool_call_file = pjoin(output_dir, "tool_call_layers.json")
    with open(tool_call_file, "w") as f:
        json.dump(tool_call_layers, f, indent=4)


def start_conversation_round_state_machine(
        output_dir: str,
        msg_thread: MessageThread,
        api_manager: ProjectApiManager,
        start_round_no: int = 0,
) -> bool:
    """
    Start the actual rounds of conversations with model.

    Args:
        output_dir (str): Path to the output directory.
        msg_thread (MessageThread): The message thread to be used.
        api_manager (ProjectApiManager): The API manager to be used.
        start_round_no (int): The round number to start with.
    """
    logger = api_manager.logger

    round_no = start_round_no
    for round_no in range(start_round_no, globals.conv_round_limit + 1):
        conversation_file = pjoin(output_dir, f"conversation_round_{round_no}.json")
        # save current state before starting a new round
        msg_thread.save_to_file(conversation_file)
        log_and_cprint(
            logger,
            f"\n========== Conversation Round {round_no} ==========",
            "red",
            attrs=["bold"],
        )
        log_and_print(
            logger, f"{colored('Current message thread:', 'green')}\n{msg_thread}"
        )

        allowed_tools = api_manager.next_tools()
        # TODO: configure the list of tools based on state machine
        tools = ProjectApiManager.get_full_funcs_for_openai(allowed_tools)

        log_and_cprint(logger, f"Current tool state: {api_manager.curr_tool}", "yellow")
        log_and_cprint(logger, f"Allowed next tool states: {allowed_tools}", "yellow")

        # create a new iteration of conversation
        (
            res_text,
            raw_tool_calls,
            func_call_intents,
            cost,
            input_tokens,
            output_tokens,
        ) = call_gpt(logger, msg_thread.to_msg(), tools=tools)
        api_manager.accumulate_cost_and_tokens(cost, input_tokens, output_tokens)
        log_and_print(
            logger, f"{colored('This roud model response (text):', 'blue')} {res_text}"
        )
        # model can decide whether to create a function call
        if len(func_call_intents) == 1:
            # good case in which we can check function call
            func_call_intent: FunctionCallIntent = func_call_intents[0]
            log_and_print(
                logger,
                f"{colored('This round model response (function call):', 'blue')} {func_call_intent}",
            )
            # dispatch this function call
            this_model_response = res_text
            this_model_tools = raw_tool_calls
            # add previous call information to user message
            tool_output, summary, _ = api_manager.dispatch_intent(
                func_call_intent, msg_thread
            )
        else:
            # no function call, let's force the model to make one
            this_model_tools = []
            this_model_response = res_text
            tool_output = ""
            summary = "There is no function call in your previous response. Make sure you include one function call. "

        next_user_message = add_step_trigger(summary)

        log_and_cprint(
            logger, f"Cost - current: {cost}; total: {api_manager.cost}", "yellow"
        )
        # form message thread for next round. should include what the model said as well
        msg_thread.add_model(this_model_response, this_model_tools)
        if this_model_tools:
            tool_call_id = this_model_tools[0].id
            msg_thread.add_tool(tool_output, tool_call_id)
            msg_thread.add_user(next_user_message)
        else:
            msg_thread.add_user(next_user_message)

        if len(func_call_intents) == 1:
            func_call_name = func_call_intents[0].func_name
            if func_call_name == "write_patch":
                log_and_print(logger, "Ending workflow. write_patch has been invoked.")
                break

        log_and_print(logger, "Going to next round ..........")
    else:
        log_and_print(logger, "Too many rounds. Try writing patch anyway.")
        write_patch_intent = FunctionCallIntent("write_patch", {}, None)
        api_manager.dispatch_intent(write_patch_intent, msg_thread)

    round_no += 1

    # if we end the workflow normally, there is one more round of conversation to store
    conversation_file = pjoin(output_dir, f"conversation_round_{round_no}.json")
    msg_thread.save_to_file(conversation_file)
    return True



def review_repo_summary_from_llm(output_dir, api_manager, issue, code_contents):
    prompt = SUMMARY_PROMPT.format(issue_content=issue, collected_content=code_contents)
    ask_summary_msg_thread = MessageThread()
    ask_summary_msg_thread.add_user(prompt)
    logger = api_manager.logger
    
    res_text, _, _, cost, input_tokens, output_tokens = call_gpt(
        logger, ask_summary_msg_thread.to_msg(),
    )
    
    ask_summary_msg_thread.add_model(res_text, [])
    api_manager.accumulate_cost_and_tokens(cost, input_tokens, output_tokens)
    
    summary_conversation_file = pjoin(output_dir, f"agent_review_repo_summary.json")
    ask_summary_msg_thread.save_to_file(summary_conversation_file)
    
    return res_text

def review_repo_from_llm(output_dir, api_manager, issue, retries=3, top_files_num=5):
    """
    review repo files, class, function and specific code snippet.
    output_format:
    "bug_locations": [
        {
            "file": "src/rez/util.py",
            "class": "PackageResourceHelper",
            "method": "create_forwarding_script"
        },
        {
            "file": "src/rez/cli/_entry_points.py",
            "class": "run_rez_fwd"
        }
        {
            "file": "src/rez/cli/_entry_points.py",
            "code_snippet": "...code..."
        }
    ]
    """
    
    logger = api_manager.logger
    
    repo_path = api_manager.project_path
    
    # STEP (1) select files
    # STEP (1.1) bm25
    top_files_from_bm25 = get_top_files_from_bm25(issue, repo_path)
    prompt_ask_files_from_gpt = get_top_files_from_llm_prompt(issue, repo_path)
    
    # STEP (1.1) llm    
    ask_location_msg_thread = MessageThread()
    ask_location_msg_thread.add_user(prompt_ask_files_from_gpt)
    for i in range(retries):
        res_text, _, _, cost, input_tokens, output_tokens = call_gpt(
                logger, ask_location_msg_thread.to_msg(), response_format="json_object",
            )
        api_manager.accumulate_cost_and_tokens(cost, input_tokens, output_tokens)
                
        if is_valid_location_json(res_text):
            ask_location_msg_thread.add_model(res_text, tools=[])
            break
        else:    
            log_and_print(logger, f"Retry {i+1} in locate files")
    
    if is_valid_location_json(res_text):
        top_files_from_llm = json.loads(res_text)['files_to_modify']
        top_files_from_llm = [
            os.path.join(repo_path, file_path[1:]) if file_path.startswith('/') else os.path.join(repo_path, file_path)
            for file_path in top_files_from_llm
        ]
    else:
        top_files_from_llm = []
    # print(f"top_files_from_bm25: {top_files_from_bm25}")
    # print(f"top_files_from_llm: {top_files_from_llm}")
    file_location_conversation_file = pjoin(output_dir, f"agent_file_location.json")
    ask_location_msg_thread.save_to_file(file_location_conversation_file)

    # STEP (1.3) concat top files
    top_files = top_files_from_llm + [file for file in top_files_from_bm25 if file not in top_files_from_llm]
    top_files = top_files[:top_files_num]
    # print(f"top_files: {top_files}")
    
    # STEP (2) select classes / functions / lines    
    prompt_ask_specific_location_from_gpt = get_top_content_from_llm_prompt(issue, top_files, repo_path)
    ask_specific_location_msg_thread = MessageThread()
    ask_specific_location_msg_thread.add_user(prompt_ask_specific_location_from_gpt)
    
    for i in range(retries):
        try:
            res_text, _, _, cost, input_tokens, output_tokens = call_gpt(
                    logger, ask_specific_location_msg_thread.to_msg(), response_format="json_object",
                )
            api_manager.accumulate_cost_and_tokens(cost, input_tokens, output_tokens)
        except Exception as e:
            log_and_print(logger, f"{e}, Retry {i+1} in locate specific contents")
            continue
        if is_valid_location_json(res_text, key_content="bug_locations"):
            ask_specific_location_msg_thread.add_model(res_text, tools=[])
            break
        else:    
            log_and_print(logger, f"Retry {i+1} in locate specific contents")
    
    top_locations_from_llm = []
    if is_valid_location_json(res_text, key_content="bug_locations"):
        top_locations_from_llm = json.loads(res_text)['bug_locations']
        
    content_location_conversation_file = pjoin(output_dir, "agent_specific_content_location.json")
    ask_specific_location_msg_thread.save_to_file(content_location_conversation_file)
    
    final_locations = get_location_from_agent_repo_review(output_dir)
    with open(os.path.join(output_dir, "locations_from_agent_review_repo.json"), 'w') as f:
        json.dump(final_locations, f, indent=4)


    # STEP (3.1) 读取oracle location文件
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    location_map_path = os.path.join(script_dir, '..', 'SWE-bench', 'setup_result', 'location_map.json')
    location_map_path = os.path.normpath(location_map_path)

    with open(location_map_path, 'r') as f:
        location_map = json.load(f)
    if api_manager.task_id in location_map.keys():
        oracle_location = location_map[api_manager.task_id]
    else:
        oracle_location = []
    # STEP (3.2) 计算定位率，写入locations_rate_from_repo_review.json文件
    location_rate_from_repo_review = cal_location_rate(final_locations, oracle_location)
    with open(os.path.join(api_manager.output_dir, "locations_rate_from_repo_review.json"), 'w') as f:
        json.dump({'from_repo_review': location_rate_from_repo_review}, f, indent=4)

    return top_locations_from_llm


def search_code_snippets_from_locations(top_locations_from_llm, api_manager, msg_thread):
    collated_tool_response = ""
    collated_path_summary = ""
    for bug_location in top_locations_from_llm:
        if not bug_location.get("code_snippets"):
            try:
                tool_output, *_ = search_for_bug_location(
                    api_manager, msg_thread, bug_location
                )
                collated_tool_response += f"{tool_output}\n"
                collated_path_summary += f"{search_for_bug_location_path_summary(bug_location)}\n" 
            except Exception as e:
                print('search_for_bug_location error!')
                print(e)
                print(f'{bug_location}')
                continue
        else:
            rel_path = bug_location.get("file")
            if not rel_path:
                continue
            file_part = f"<file>{rel_path}</file>"
            code_part = f" <code_snippets>{bug_location['code_snippets']}</code_snippets>"
            collated_tool_response += f"Found 1 code snippet in file {rel_path}.\nSearch result 1: {file_part+code_part}\n"
            collated_path_summary += f"{file_part}{code_part}\n"

    return collated_tool_response, collated_path_summary


def run_one_task(
        output_dir: str, api_manager: ProjectApiManager, problem_stmt: str
) -> bool:
    """
    Main entry point to run inference on one task.
    Args:
        output_dir (str): Path to the output directory.
        api_manager (ProjectApiManager): The already-initialized API manager.
        problem_stmt (str): The original problem statement submitted to the task issue.
    """
    msg_thread = MessageThread()

    system_prompt = SYSTEM_PROMPT
    if (
            not globals.enable_layered
    ) and globals.model in globals.PARALLEL_TOOL_CALL_MODELS:
        # these models support parallel tool calls, let's try to make them not do it
        system_prompt += " In your response, DO NOT make more than one tool call."

    msg_thread.add_system(system_prompt)
    original_prompt = prepare_issue_prompt(problem_stmt)
    msg_thread.add_user(original_prompt)

    logger = api_manager.logger
    log_and_print(
        logger,
        f"{colored('myw test:', 'blue')} {globals.enable_mtcs}",
    )
    # Add mtcs code search
    if globals.enable_mtcs:
        search_result, mcts_cost = api_manager.mcts_code_search(original_prompt)

        mcts_cost['cost'] = globals.MODEL_COST_PER_INPUT[globals.model] * mcts_cost['input_tokens'] + globals.MODEL_COST_PER_OUTPUT[globals.model] * mcts_cost['output_tokens']
        

        if not search_result:
            log_and_print(
                logger,
                f"{colored('search_result:', 'red')} 用于生成图, 提前返回.",
            )
        log_and_print(
            logger,
            f"{colored('search_result:', 'red')} {search_result}",
        )
        msg_thread.add_user(search_result)
        
        log_and_print(
            logger,
            f"{colored('MCTS cost:', 'red')} {mcts_cost}",
        )
        api_manager.accumulate_cost_and_tokens(mcts_cost['cost'], mcts_cost['input_tokens'], mcts_cost['output_tokens'])

    if globals.only_mcts:
        print('****** only_mcts, workflow end ******')
        return True
    
    # Add another user message about fault localization
    if globals.enable_sbfl:
        localization_result, _, _ = api_manager.fault_localization()
        localization_prompt = "An external analysis tool has been deployed to identify the suspicious code to be fixed. You can choose to use the results from this tool, if you think they are useful."
        localization_prompt += "The tool output is as follows:\n"
        localization_prompt += localization_result
        msg_thread.add_user(localization_prompt)

    if globals.review_repo:
        print("begin review_repo process...")
        top_locations_from_llm = review_repo_from_llm(output_dir, api_manager, issue=original_prompt)
        
        if len(top_locations_from_llm) > 0:
            top_locations_codes, top_locations_codes_summary = search_code_snippets_from_locations(top_locations_from_llm, api_manager, msg_thread)
            try:
                summary_response = review_repo_summary_from_llm(output_dir, api_manager, issue=original_prompt, code_contents=top_locations_codes)
            except:
                summary_response = ""
                
            review_prompt = "An external analysis tool has been deployed to identify the suspicious code to be fixed. You can choose to use the results from this tool, if you think they are useful."
            review_prompt += "The tool output is as follows:\n"
            review_prompt += top_locations_codes_summary + "\n"
            review_prompt += summary_response
            msg_thread.add_user(review_prompt)        
        
    
    if globals.continue_task_from_cache:
        print('****** start from cache!!! ******')
        return start_conversation_round_from_cache(output_dir, msg_thread, api_manager, issue=original_prompt)
    elif globals.enable_layered:
        return start_conversation_round_stratified(output_dir, msg_thread, api_manager, issue=original_prompt)
    else:
        return start_conversation_round_state_machine(
            output_dir, msg_thread, api_manager
        )


# NOTE: deprecated
def continue_task_from_cache(
        cache_path: str, output_dir: str, api_manager: ProjectApiManager
) -> bool:
    """
    Run inference on one task, but load conversation history from cache.
    Args:
        cache_path (str): Path to the old conversation history file.
        output_dir (str): Path to the output directory.
        api_manager (ProjectApiManager): The already-initialized API manager.
    """
    # (1) load the existing message thread
    msg_thread = MessageThread.load_from_file(cache_path)
    completed_round_no = msg_thread.get_round_number()

    # (2) start the actual workflow
    return start_conversation_round_state_machine(
        output_dir, msg_thread, api_manager, start_round_no=completed_round_no
    )
