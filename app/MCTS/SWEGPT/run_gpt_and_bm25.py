import os
import json
import random
import requests
from tqdm import tqdm
import concurrent.futures
import argparse
import tiktoken
import numpy as np
import glob
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
from MCTS.SWEGPT.bm25 import BM25Retriever
from MCTS.SWEGPT.ask_llm_location_file import run_FilesExtraction, get_RewardValue, get_SummaryResults
random.seed(1234)


issue = """
3.0: DateTime fields cannot be used as inner field for List or Tuple fields
Between releases 3.0.0rc8 and 3.0.0rc9, `DateTime` fields have started throwing an error when being instantiated as inner fields of container fields like `List` or `Tuple`. The snippet below works in <=3.0.0rc8 and throws the error below in >=3.0.0rc9 (and, worryingly, 3.0.0):

```python
from marshmallow import fields, Schema
class MySchema(Schema):
    times = fields.List(fields.DateTime())
    s = MySchema()
```
Traceback:
```
Traceback (most recent call last):
  File "test-mm.py", line 8, in <module>
      s = MySchema()
  File "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/schema.py", line 383, in __init__
    self.fields = self._init_fields()
  File "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/schema.py", line 913, in _init_fields
    self._bind_field(field_name, field_obj)
  File "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/schema.py", line 969, in _bind_field
    field_obj._bind_to_schema(field_name, self)
  File "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/fields.py", line 636, in _bind_to_schema
    self.inner._bind_to_schema(field_name, self)
  File "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/fields.py", line 1117, in _bind_to_schema
    or getattr(schema.opts, self.SCHEMA_OPTS_VAR_NAME)
AttributeError: 'List' object has no attribute 'opts'
```
It seems like it's treating the parent field as a Schema without checking that it is indeed a schema, so the `schema.opts` statement fails as fields don't have an `opts` attribute.

"""


token = 'your_key'
url = "request url"

def get_all_py_files(dir_path: str):
    """Get all .py files recursively from a directory.

    Skips files that are obviously not from the source code, such third-party library code.

    Args:
        dir_path (str): Path to the directory.
    Returns:
        List[str]: List of .py file paths. These paths are ABSOLUTE path!
    """

    py_files = glob.glob(os.path.join(dir_path, "**/*.py"), recursive=True)
    res = []
    for file in py_files:
        rel_path = file[len(dir_path) + 1:]
        if rel_path.startswith("build"):
            continue
        if rel_path.startswith("doc"):
            # discovered this issue in 'pytest-dev__pytest'
            continue
        if rel_path.startswith("requests/packages"):
            # to walkaround issue in 'psf__requests'
            continue
        # 新增，过滤tests文件夹
        if "test" in rel_path:
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
        res.append(file)
    return res


class ChatGPT:
    def __init__(self, args):
        self.fin = args.fin
        self.fout = args.fout
        self.n_workers = args.n_workers
        self.multithread = args.multithread
        self.n_samples = args.n_samples
        self.model_name = args.model_name
        self.logprobs = False
        self.basenames = []
        if self.fin:
            self.get_basename()

    def get_embedding(self, query):
        payload = {
            'input': self.num_tokens_from_string(query)[0],
            'model': self.model_name
        }
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        resp = requests.request("POST", url, headers=headers, json=payload)
        # print(resp.json())
        if resp and resp.json().get('data').get('response').get('data'):
            data = resp.json()['data']['response']['data'][0]['embedding']
            return np.array(data).reshape(1, -1)
        else:
            raise Exception('chatgpt get None')

    def process_batch(self, batch):
        batch_results = []
        for query in batch:
            result = self.get_embedding(query[1])
            batch_results.append([query[0], result])
        return batch_results
    

    def get_batch_embeddings(self, queries, batch_size=10):
        batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
        results = []

        # 使用线程池处理每个批次
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_batch, batch) for batch in batches]
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())

        return results

    def num_tokens_from_string(self, string: str, encoding_name: str = 'cl100k_base'):
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(string)
        num_tokens = len(tokens)
        if num_tokens < 8191:
            return string, num_tokens
        else:
            return encoding.decode(tokens[:8191]), 8191

    def sim_score(self, vec1, vec2):
        return cosine_similarity(vec1, vec2)[0]

def test_openai_embedding(chat_gpt, predict_list_num, with_code=True):
    json_path = '../dev/swe-bench-dev-search.json'
    repo_path = '../'

    with open(json_path, 'r') as f:
        datas = json.load(f)

    random.shuffle(datas)
    avg_recall = 0
    data_len = 1000
    flag = 0
    for data in tqdm(datas[:data_len]):
        score_dict = {}
        instance_id = data['instance_id']
        query = data['requirement']

        # 无代码情况
        if not with_code:
            if "reproduce" in query or "Reproduce" in query or "```" in query:
                continue

        sub_repo_path = data['repo_path']
        new_repo_path = os.path.join(repo_path, sub_repo_path)
        orcal_files = data['orcal_files']
        query_embedding = chat_gpt.get_embedding(query)
        py_files = get_all_py_files(new_repo_path)
        codes = []
        print('file nums:', len(py_files))
        # print('query:', query)
        for py_file in py_files:
            with open(py_file, 'r') as f:
                code = f.read()
            rel_path = py_file.replace(new_repo_path, '/')
            codes.append([rel_path, code])
        batch_code_embedding = chat_gpt.get_batch_embeddings(codes)
        # print('batch_code_embedding', len(batch_code_embedding))
        for code_embedding in batch_code_embedding:
            rel_path = code_embedding[0]
            embedding = code_embedding[1]
            score_dict[rel_path] = chat_gpt.sim_score(embedding, query_embedding)

        sorted_dict = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)[:predict_list_num]
        sorted_name = [item[0] for item in sorted_dict]
        intersection_list = [item for item in sorted_name if item in orcal_files]
        recall = len(intersection_list)/len(orcal_files)
        print('recall', recall)
        avg_recall += recall
        flag += 1
        print('avg_recall', avg_recall/flag)


def test_bm25_reviewer(predict_list_num, with_code=True):
    json_path = '../dev/swe-bench-dev-search.json'
    repo_path = '../'

    with open(json_path, 'r') as f:
        datas = json.load(f)

    random.shuffle(datas)
    avg_recall = 0
    data_len = 1000
    flag = 0
    for data in tqdm(datas[:data_len]):
        instance_id = data['instance_id']
        query = data['requirement']
        # 无代码情况
        if not with_code:
            if "reproduce" in query or "Reproduce" in query or "```" in query:
                continue

        sub_repo_path = data['repo_path']
        new_repo_path = os.path.join(repo_path, sub_repo_path)
        orcal_files = data['orcal_files']
        py_files = get_all_py_files(new_repo_path)
        codes = []
        paths = []
        print('file nums:', len(py_files))
        for py_file in py_files:
            with open(py_file, 'r') as f:
                code = f.read()
            rel_path = py_file.replace(new_repo_path, '/')
            codes.append(code)
            paths.append({"source": rel_path})

        if not codes:
            continue

        bm25_retriever = BM25Retriever.from_texts(
            codes, metadatas=paths
        )
        bm25_retriever.k = predict_list_num

        results = bm25_retriever.get_relevant_documents(query)
        predicted_list = []
        for res in results:
            path = res.metadata['source']
            content = res.page_content
            predicted_list.append(path)

        intersection_list = [item for item in predicted_list if item in orcal_files]
        recall = len(intersection_list) / len(orcal_files)
        print('recall', recall)
        avg_recall += recall
        flag += 1
    print('avg_recall', avg_recall / flag)


def test_bm25_faiss_reviewer(predict_list_num, with_code=True):
    json_path = '../dev/swe-bench-dev-search.json'
    repo_path = '../'

    with open(json_path, 'r') as f:
        datas = json.load(f)

    random.shuffle(datas)
    avg_recall = 0
    data_len = 1000
    flag = 0
    for data in tqdm(datas[:data_len]):
        instance_id = data['instance_id']
        query = data['requirement']
        # 无代码情况
        if not with_code:
            if "reproduce" in query or "Reproduce" in query or "```" in query:
                continue

        sub_repo_path = data['repo_path']
        new_repo_path = os.path.join(repo_path, sub_repo_path)
        orcal_files = data['orcal_files']
        py_files = get_all_py_files(new_repo_path)
        codes = []
        paths = []
        print('file nums:', len(py_files))
        for py_file in py_files:
            with open(py_file, 'r') as f:
                code = f.read()
            rel_path = py_file.replace(new_repo_path, '/')
            codes.append(code)
            paths.append({"source": rel_path})

        if not codes:
            continue

        bm25_retriever = BM25Retriever.from_texts(
            codes, metadatas=paths
        )
        bm25_retriever.k = predict_list_num

        results = bm25_retriever.get_relevant_documents(query)
        predicted_list = []
        for res in results:
            path = res.metadata['source']
            content = res.page_content
            predicted_list.append(path)

        intersection_list = [item for item in predicted_list if item in orcal_files]
        recall = len(intersection_list) / len(orcal_files)
        print('recall', recall)
        avg_recall += recall
        flag += 1
    print('avg_recall', avg_recall / flag)


def get_bm25_retriever(repo_path):
    py_files = get_all_py_files(repo_path)
    codes, paths = [], []
    for py_file in py_files:
        with open(py_file, 'r') as f:
            code = f.read()
        rel_path = py_file.replace(repo_path, '')
        codes.append(code)
        paths.append({"source": rel_path})

    bm25_retriever = BM25Retriever.from_texts(
        codes, metadatas=paths
    )
    return bm25_retriever


def get_bm25_score(issue, code_content, retriever):
    # bm25_retriever.k = predict_list_num
    score = retriever._get_scores_query_code(issue, code_content)
    return score


def get_gpt4_and_bm25_results(query, repo_path, predict_list_num=10, use_gpt=True):
    py_files = get_all_py_files(repo_path)
    codes = []
    paths = []
    only_paths = []
    workspace_metainfo = ""
    print('file nums:', len(py_files))
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

    # use gpt-4，gpt-4有时候不通.
    if use_gpt:
        # print('gpt:', use_gpt)
        predicted_list = run_FilesExtraction(workspace_metainfo, query)
        print('predict:', predicted_list)
        # gpt4 有时候会出现幻觉，需要避免
        predicted_list = [item for item in predicted_list if item in only_paths]
    else:
        predicted_list = []
    # 集成bm25。
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
    union_list = union_list[:10]
    ret_list = []
    for path_item in union_list:
        if repo_path[-1] == "/":
            # rel_path里有"/",去除,避免重复
            new_path = repo_path + path_item[1:]
        else:
            new_path = repo_path + path_item
        ret_list.append(new_path)
    return ret_list


def get_reward_value(issue: str, code_content: str, method_type:str, method_name:str, rel_file_path:str):
    return get_RewardValue(issue, code_content, method_type, method_name, rel_file_path)


def get_summary_results(prompt: str):
    return get_SummaryResults(prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fin", type=str, default=None, help="Input file path")
    parser.add_argument("--fout", type=str, default=None, help="Output file path")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt")
    parser.add_argument("--code", type=str, default=None, help="Input code file path")
    parser.add_argument("--n-workers", type=int, default=60, help="Number of workers")
    parser.add_argument("--model-name", type=str, default="text-embedding-3-small", help="model name")  # gpt-4-0125-preview
    parser.add_argument(
        "--multithread",
        action="store_true",
        help="Use multiple threads to process the input file",
    )
    parser.add_argument(
        "--n-samples", type=int, default=0, help="Number of samples for test run"
    )
    args = parser.parse_args()

    chat_gpt = ChatGPT(args)

    predict_list_num = 10
    with_code = True
    # test_openai_embedding(chat_gpt, predict_list_num, with_code)
    # test_bm25_reviewer(predict_list_num, with_code)
    # test_gpt4_reviewer(predict_list_num, with_code)

