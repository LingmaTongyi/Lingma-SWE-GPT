import re
import os
from app.search.search_utils import get_all_py_files, get_all_classes_in_file, get_top_level_functions_signatures, get_code_snippets, get_class_signature
from app.search.bm25 import BM25Retriever
import json

CONTEXT_TEMPLATE = """
You are a programming assistant who helps users solve issue regarding their workspace code. 
Your responsibility is to use the description or logs in the issue to locate the code files that need to be modified.
The user will provide you with potentially relevant information from the workspace.
DO NOT ask the user for additional information or clarification.
DO NOT try to answer the user's question directly.

# Additional Rules

Think step by step:
1. Read the user's question to understand what they are asking about their workspace.
2. If there is traceback information in the console logs, then the file where the error reporting function is located is most likely the file to be modified.
3. Please note that the absolute path to the file can be different from the workspace, as long as the file name is the same.
4. OUTPUT AT MOST 5 FILES that need to be modified in their workspace and sort them by modification priority.

# Examples
I am working in a workspace that has the following structure:
```
- /src/base64.py
- /src/base64_test.py
- /src/model.py
- /setup.py
```
User: Where's the code for base64 encoding?

Response:
{{"files_to_modify": ["/src/base64.py", "/src/base64_test.py"]}}

# Now the workspace is:
{workspace_metainfo}

User's question (issue)
{question}

Response (according to the examples json_format):
"""


file_content_in_block_template = """
### File: {file_name} ###
```python
{file_content}
```
"""


LOCATION_PROMPT_TEMPLATE = """
Please look through the following GitHub Problem Description and the Skeleton of Relevant Files.
Identify all locations that need inspection or editing to fix the problem, including directly related areas as well as any potentially related functions, classes, and global code snippets.
For each location you provide, either give the name of the class, the name of a function, or the global code snippets from a region in the original text.

## GitHub Problem Description ##
{problem_statement}

## Skeleton of Relevant Files ##
{file_contents}

##

Please provide the complete set of locations as either a class name, a function name, or a global code snippet from a region in the original text.
Note that if you include a class, you do not need to list its specific methods. Please note that the global code snippet usually does not exceed three lines.
You can include either the entire class or don't include the class name and instead include specific methods in the class.
OUTPUT AT MOST 10 LOCATIONS that need to be modified in their workspace and sort them by modification priority.

Please respond according to the JSON format in Response Example.
## Response Example:
{{
    "reason": "To resloved this issue, xxx. (Here please analyze possible fault locations step by step and output your chain-of-thought)",
    "bug_locations": [
        {{"file": "file path", "class": "class name", "method": "function name"}},
        {{"file": "file path", "class": "class name"}},
        {{"file": "file path", "code_snippets": "code snippet from the original text"}}
    ]
}}

Return just in JSON format.
"""

SUMMARY_PROMPT = """You are a senior software developer maintaining a large project.
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

def get_top_files_from_llm_prompt(query, repo_path):
    py_files = get_all_py_files(repo_path)
    workspace_metainfo = ""
    # print('file nums:', len(py_files))
    for py_file in py_files:
        rel_path = py_file.replace(repo_path, '')
        workspace_metainfo += f"- {rel_path}\n"
    prompt = CONTEXT_TEMPLATE.format(workspace_metainfo=workspace_metainfo, question=query)
    return prompt

    
def get_top_files_from_bm25(query, repo_path, predict_list_num=10, use_gpt=True):
    py_files = get_all_py_files(repo_path)
    codes = []
    paths = []
    only_paths = []
    workspace_metainfo = ""
    # print('file nums:', len(py_files))
    for py_file in py_files:
        # print('file_name:', py_file)
        with open(py_file, 'r') as f:
            code = f.read()
        rel_path = py_file.replace(repo_path, '/')
        codes.append(code)
        paths.append({"source": rel_path})
        only_paths.append(rel_path)
        workspace_metainfo += f"- {rel_path}\n"

    if not codes:
        raise Exception('No code!')

    predicted_list = []
    bm25_retriever = BM25Retriever.from_texts(
        codes, metadatas=paths
    )
    bm25_retriever.k = predict_list_num

    results = bm25_retriever.get_relevant_documents(query)
    predicted_list_bm25 = []
    for res in results:
        path = res.metadata['source']
        predicted_list_bm25.append(path)

    union_list = predicted_list + [item for item in predicted_list_bm25 if item not in predicted_list]
    union_list = union_list[:predict_list_num]
    ret_list = []
    
    for path_item in union_list:
        if path_item.startswith('//'):
            path_item = path_item[2:]
        elif path_item.startswith('/'):
            path_item = path_item[1:]
        new_path = os.path.join(repo_path, path_item)
        ret_list.append(new_path)
    return ret_list


def is_valid_location_json(json_str: str, key_content="files_to_modify"):
    """
    Check whether a json string is valid.
    """
    try:
        datas = json.loads(json_str).get(key_content, [])
        if len(datas) <= 0:
            return False
        return True
    except:
        return False



def get_omitted_full_content(file_full_path: str) -> str:
    with open(file_full_path, "r") as f:
        file_content = f.read()

    # print("***src file_content***")
    # print(file_content)
    
    classes = get_all_classes_in_file(file_full_path)
    top_funcs, top_funcs_sigs = get_top_level_functions_signatures(file_full_path)

    for idx, top_func in enumerate(top_funcs):
        file_content = file_content.replace(top_func, top_funcs_sigs[idx])

    for idx, class_item in enumerate(classes):
        class_name, b, e = class_item[0], class_item[1], class_item[2]
        class_full_code = get_code_snippets(file_full_path, b, e)
        class_omit_code = get_class_signature(file_full_path, class_name)
        file_content = file_content.replace(class_full_code, class_omit_code)

    return file_content

def get_top_content_from_llm_prompt(query, files_path, repo_path):
    
    omitted_file_contents = []
    for file_path in files_path:
        if not os.path.exists(file_path):
            print(f"File {file_path} not exists!")
            continue
        try:
            file_content = get_omitted_full_content(file_path)
        except:
            # raise Exception('File omitted error!')
            file_content = open(file_path, 'r').read()
            print('File omitted error! Use full file content!')
        
        rel_path = file_path.replace(repo_path, '')
        if rel_path.startswith('/'):
            rel_path = rel_path[1:]
        omitted_file_contents.append(file_content_in_block_template.format(file_name=rel_path, file_content=file_content))
    
    omitted_file_content = "\n".join(omitted_file_contents)
    
    # print("***omitted file_content***")
    # print(omitted_file_content)
    
    prompt = LOCATION_PROMPT_TEMPLATE.format(file_contents=omitted_file_content, problem_statement=query)
    # print(prompt)
    return prompt



if __name__ == '__main__':
    repo_path = "/opt/temp_acr/Repos/Repos/testbed/AcademySoftwareFoundation/rez/edb5e31fd9291fdfeffeda98acda33b0e5136107/"
    query = "get the code for base64 encoding; please output the absolute path of the file; please sort them by modification priority; please output the absolute path of the file; please sort them by modification priority; please output the absolute path of the file; please sort them by modification priority; please output the absolute path of the file; please sort them by modification priority; please output the absolute path of the file; please sort them by modification priority; please output the absolute path of the file; please sort them by modificationpriority; please output the absolute path of the file; please sort them by modification"
    ret_list = get_top_files_from_bm25(query, repo_path)
    print(ret_list)