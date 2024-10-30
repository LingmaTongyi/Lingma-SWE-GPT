#!/usr/bin/env python
# -*- coding: utf-8 -*-

from metagpt.logs import log_llm_stream, logger
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_incrementing
from metagpt.utils.common import general_after_log
import re
from http import HTTPStatus
import dashscope
import argparse

def my_retry_error_callback(retry_state):
    # return "retry error callback", None
    return -1, None


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
4. OUTPUT ONE TO FIVE FILE that most need to be modified in their workspace and sort them by modification priority.

# Examples
I am working in a workspace that has the following structure:
```
- /src/base64.py
- /src/base64_test.py
- README.md
- requirements.txt
```
User: Where's the code for base64 encoding?

Response:
Files to modify: 
- /src/base64.py
- /src/base64_test.py

# Now the workspace is:
{workspace_metainfo}

User's question (issue)
{question}

Response (according to the examples format):
"""

REWARD_TEMPLATE = """
You are a programming assistant who helps users solve issue regarding their workspace code. 
Your main responsibilities include examining issue information to analyze possible causes of the issue and determine the code that needs to be fixed.
Given an issue and a piece of code in the repository, please read and understand the issue carefully, and determine whether the given code is a possible cause of the issue. If the issue contains a bug, please determine whether the code is a buggy location.
Please refer to the above responsibilities and provide detailed reasoning and analysis. Then at the last line conclude "Thus the probability score that this code needs to be modified to solve this issue is s", where s is an integer between 1 and 10.

# Examples
Issue: ModelChain.prepare_inputs can succeed with missing dhi\nFrom the docstring for `ModelChain.prepare_inputs()` I believe the method should fail if `weather` does not have a `dhi` column.\r\n\r\nThe validation checks for `'ghi'` twice, but not `'dhi`.
Code: 
```
# class-function method prepare_inputs_from_poa in pvlib/modelchain.py
def prepare_inputs_from_poa(self, data):
    \"\"\"
    Prepare the solar position, irradiance and weather inputs to
    the model, starting with plane-of-array irradiance.

    Parameters
    ----------
    data : DataFrame
        Contains plane-of-array irradiance data. Required column names
        include ``'poa_global'``, ``'poa_direct'`` and ``'poa_diffuse'``.
        Columns with weather-related data are ssigned to the
        ``weather`` attribute.  If columns for ``'temp_air'`` and
        ``'wind_speed'`` are not provided, air temperature of 20 C and wind
        speed of 0 m/s are assumed.

    Notes
    -----
    Assigns attributes: ``weather``, ``total_irrad``, ``solar_position``,
    ``airmass``, ``aoi``.

    See also
    --------
    pvlib.modelchain.ModelChain.prepare_inputs
    \"\"\"

    self._assign_weather(data)

    self._verify_df(data, required=['poa_global', 'poa_direct',
                                        'poa_diffuse'])
    self._assign_total_irrad(data)

    self._prep_inputs_solar_pos()
    self._prep_inputs_airmass()

    if isinstance(self.system, SingleAxisTracker):
        self._prep_inputs_tracking()
    else:
        self._prep_inputs_fixed()

    return self
```
Thought: To solve the verification problem in the ModelChain.prepare_inputs() method mentioned in the issue, we do not need to modify the prepare_inputs_from_poa function, but should focus on ModelChain.prepare_inputs() itself. This is because the prepare_inputs_from_poa function and prepare_inputs() function handle different input data and verification logic.
Result: Thus the probability score that this code needs to be modified to solve this issue is 1.

Issue: Enable quiet mode/no-verbose in CLI for use in pre-commit hook\nThere seems to be only an option to increase the level of verbosity when using SQLFluff [CLI], not to limit it further.\r\n\r\nIt would be great to have an option to further limit the amount of prints when running `sqlfluff fix`, especially in combination with deployment using a pre-commit hook. For example, only print the return status and the number of fixes applied, similar to how it is when using `black` in a pre-commit hook:\r\n\r\nThis hides the potentially long list of fixes that are being applied to the SQL files, which can get quite verbose.
Code:
```
# top-level method do_fixed in src/sqlfluff/cli/commands.py file
def do_fixes(lnt, result, formatter=None, **kwargs):
    \"\"\"Actually do the fixes.\"\"\"
    click.echo("Persisting Changes...")
    res = result.persist_changes(formatter=formatter, **kwargs)
    if all(res.values()):
        click.echo("Done. Please check your files to confirm.")
        return True
    # If some failed then return false
    click.echo(
        "Done. Some operations failed. Please check your files to confirm."
    )  # pragma: no cover
    click.echo(
        "Some errors cannot be fixed or there is another error blocking it."
    )  # pragma: no cover
    return False  # pragma: no cover
```
Thought: Users hope to have an option to limit the amount of output information when using the sqlfluff fix command, similar to the output behavior when using black in the pre-commit hook. The purpose is to hide the long list of repairs applied when repairing SQL files, making the output more concise. In the do_fixes function, there are multiple click.echo() calls used to output progress and result messages on the command line. In response to the issue's request, which is to implement a silent mode with less or no output in the CLI, it is necessary to modify this function so that the output can be controlled and the information will only be displayed when necessary. It should be noted that other related functions may need to be modified.
Result: Thus the probability score that this code needs to be modified to solve this issue is 8.

Issue: Bug within scaling.py wavelet calculation methodology\n**Describe the bug**\r\nMathematical error within the wavelet computation for the scaling.py WVM implementation. Error arises from the methodology, as opposed to just a software bug. \r\n\r\n**To Reproduce**\r\nSteps to reproduce the behavior:\r\n```\r\nimport numpy as np\r\nfrom pvlib import scaling\r\ncs = np.random.rand(2**14)\r\nw, ts = scaling._compute_wavelet(cs,1)\r\nprint(np.all( (sum(w)-cs) < 1e-8 ))  # Returns False, expect True\r\n```\r\n\r\n**Expected behavior**\r\nFor a discrete wavelet transform (DWT) the sum of all wavelet modes should equate to the original data. \r\n\r\n**Versions:**\r\n - ``pvlib.__version__``: 0.7.2\r\n - ``pandas.__version__``: 1.2.3\r\n - python: 3.8.8\r\n\r\n**Additional context**\r\nThis bug is also present in the [PV_LIB](https://pvpmc.sandia.gov/applications/wavelet-variability-model/) Matlab version that was used as the basis for this code (I did reach out to them using the PVLIB MATLAB email form, but don't know who actually wrote that code). Essentially, the existing code throws away the highest level of Detail Coefficient in the transform and keeps an extra level of Approximation coefficient. The impact on the calculation is small, but leads to an incorrect DWT and reconstruction. I have a fix that makes the code pass the theoretical test about the DWT proposed under 'To Reproduce' but there may be some question as to whether this should be corrected or left alone to match the MATLAB code it was based on.
Code:
```
# top-level method _compute_wavelet in pvlib/scaling.py file
def _compute_wavelet(clearsky_index, dt=None):
    \"\"\"
    Compute the wavelet transform on the input clear_sky time series.

    Parameters
    ----------
    clearsky_index : numeric or pandas.Series
        Clear Sky Index time series that will be smoothed.

    dt : float, default None
        The time series time delta. By default, is inferred from the
        clearsky_index. Must be specified for a time series that doesn't
        include an index. Units of seconds [s].

    Returns
    -------
    wavelet: numeric
        The individual wavelets for the time series

    tmscales: numeric
        The timescales associated with the wavelets in seconds [s]
    \"\"\"

    try:  # Assume it's a pandas type
        vals = clearsky_index.values.flatten()
    except AttributeError:  # Assume it's a numpy type
        vals = clearsky_index.flatten()
        if dt is None:
            raise ValueError("dt must be specified for numpy type inputs.")
    else:  # flatten() succeeded, thus it's a pandas type, so get its dt
        try:  # Assume it's a time series type index
            dt = (clearsky_index.index[1] - clearsky_index.index[0]).seconds
        except AttributeError:  # It must just be a numeric index
            dt = (clearsky_index.index[1] - clearsky_index.index[0])

    # Pad the series on both ends in time and place in a dataframe
    cs_long = np.pad(vals, (len(vals), len(vals)), 'symmetric')
    cs_long = pd.DataFrame(cs_long)

    # Compute wavelet time scales
    min_tmscale = np.ceil(np.log(dt)/np.log(2))  # Minimum wavelet timescale
    max_tmscale = int(12 - min_tmscale)  # maximum wavelet timescale

    tmscales = np.zeros(max_tmscale)
    csi_mean = np.zeros([max_tmscale, len(cs_long)])
    # Loop for all time scales we will consider
    for i in np.arange(0, max_tmscale):
        j = i+1
        tmscales[i] = 2**j * dt  # Wavelet integration time scale
        intvlen = 2**j  # Wavelet integration time series interval
        # Rolling average, retains only lower frequencies than interval
        df = cs_long.rolling(window=intvlen, center=True, min_periods=1).mean()
        # Fill nan's in both directions
        df = df.fillna(method='bfill').fillna(method='ffill')
        # Pop values back out of the dataframe and store
        csi_mean[i, :] = df.values.flatten()

    # Calculate the wavelets by isolating the rolling mean frequency ranges
    wavelet_long = np.zeros(csi_mean.shape)
    for i in np.arange(0, max_tmscale-1):
        wavelet_long[i, :] = csi_mean[i, :] - csi_mean[i+1, :]
    wavelet_long[max_tmscale-1, :] = csi_mean[max_tmscale-1, :]  # Lowest freq

    # Clip off the padding and just return the original time window
    wavelet = np.zeros([max_tmscale, len(vals)])
    for i in np.arange(0, max_tmscale):
        wavelet[i, :] = wavelet_long[i, len(vals)+1: 2*len(vals)+1]

    return wavelet, tmscales
```
Thought: The original issue pointed out that _compute_wavelet, as a buggy function, cannot correctly reconstruct the input signal when performing wavelet transform. Theoretically, for discrete wavelet transform (DWT), the sum of all wavelet modes should be equal to the original data. However, during testing it was found that this reconstruction condition was not met. To resolve this issue, a fundamental change in the implementation of the _compute_wavelet function is required.
Result: Thus the probability score that this code needs to be modified to solve this issue is 10.

# Now the issue is:
{issue}
Code:
```
# {method_type} method {method_name} in {rel_file_path} file
{code_content}
```
Thought:
"""


token = 'your key'
url = "request url"


class ChatGPT:
    def __init__(self, args):
        self.fin = args['fin']
        self.fout = args['fout']
        self.n_workers = args['n_workers']
        self.n_samples = args['n_samples']
        self.model_name = args['model_name']
        self.logprobs = False
        self.basenames = []
        if self.fin:
            self.get_basename()

    def parse_response(self, rsp):
        files_to_modify = None
        try:
            files_to_modify = []
            for line in rsp.split("Files to modify")[1].strip().split("\n"):
                if line.startswith('-'):
                    files_to_modify.append(line[2:].strip().strip('`'))
        except Exception as e:
            logger.info(f'Extract files to modify error: {e}')

        return files_to_modify

    def parse_value_score(self, rsp):
        pattern = r"Thus the probability score that this code needs to be modified to solve this issue is (10|[1-9])(,|\.|\s|$)"
        match = re.search(pattern, rsp)
        if match:
            score_s = match.group(1)
        else:
            pattern = r"is (\d+)"
            match = re.search(pattern, rsp)
            if match:
                score_s = match.group(1)
            else:
                raise ValueError(f"no score found for {rsp}")
        print(f'response:{rsp}')
        return int(score_s)

    @retry(
        wait=wait_random_exponential(min=50, max=100),
        stop=stop_after_attempt(5),
        after=general_after_log(logger),
        retry_error_callback=my_retry_error_callback
    )
    def chat(self, prompt, history=None, verbose=False):
        if history is None:
            history = []

        messages = history + [{"role": "user", "content": prompt}]
        history = messages
        # print('###:', self.model_name)
        payload = {
            'model': self.model_name,
            'messages': messages,
            'logprobs': self.logprobs,
            # 'top_logprobs' : 2,
            # 'ask': False
        }

        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        resp = requests.request("POST", url, headers=headers, json=payload)
        if resp and resp.json().get('data').get('response'):
            data = resp.json()['data']['response']
            if data.get('choices'):
                reply = data['choices'][0]['message']
                history.append({'role': 'assistant', 'content': reply['content']})
                if self.logprobs:
                    return self.parse_response(reply['content']), data['choices'][0]['logprobs']
                else:
                    return self.parse_response(reply['content']), history
            else:
                raise Exception("请求失败。\nMessage: {}".format(resp.text))
        else:
            raise Exception("请求失败。\nMessage: {}".format(resp.text))

    @retry(
        wait=wait_random_exponential(min=10, max=50),
        stop=stop_after_attempt(5),
        after=general_after_log(logger),
        retry_error_callback=my_retry_error_callback
    )
    def chat_with_Qwen(self, prompt, history=None):
        if history is None:
            history = []

        messages = history + [{"role": "user", "content": prompt}]
        history = messages

        dashscope.api_key = 'your_key'
        response = dashscope.Generation.call(
            model='qwen-max-longcontext',  # llama3-8b-instruct qwen-max-longcontext
            messages=messages,
            result_format='message',  # set the result to be "message" format.
        )

        if response.status_code != HTTPStatus.OK:
            raise Exception('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))

        if response and response.get('output'):
            data = response['output']
            if data.get('choices'):
                reply = data['choices'][0]['message']
                history.append({'role': 'assistant', 'content': reply['content']})
                return self.parse_response(reply['content']), history
            else:
                raise Exception("请求失败。\nMessage: {}".format(response))
        else:
            raise Exception("请求失败。\nMessage: {}".format(response))

    @retry(
        wait=wait_random_exponential(min=10, max=50),
        stop=stop_after_attempt(5),
        after=general_after_log(logger),
        retry_error_callback=my_retry_error_callback
    )
    def get_reward(self, prompt, history=None):
        if history is None:
            history = []

        messages = history + [{"role": "user", "content": prompt}]
        history = messages
        payload = {
            'model': self.model_name,
            'messages': messages,
            'logprobs': self.logprobs,
            # 'top_logprobs' : 2,
            # 'ask': False
        }

        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        resp = requests.request("POST", url, headers=headers, json=payload)
        if resp and resp.json().get('data').get('response'):
            data = resp.json()['data']['response']
            if data.get('choices'):
                reply = data['choices'][0]['message']
                history.append({'role': 'assistant', 'content': reply['content']})
                input_tokens = int(resp.json()['data']['prompt_tokens'])
                output_tokens = int(resp.json()['data']['completion_tokens'])
                print(input_tokens, output_tokens)
                if self.logprobs:
                    return self.parse_value_score(reply['content']), data['choices'][0]['logprobs'], input_tokens, output_tokens
                else:
                    return self.parse_value_score(reply['content']), history, input_tokens, output_tokens
            else:
                raise Exception("请求失败。\nMessage: {}".format(resp.text))
        else:
            raise Exception("请求失败。\nMessage: {}".format(resp.text))

    @retry(
        wait=wait_random_exponential(min=10, max=50),
        stop=stop_after_attempt(5),
        after=general_after_log(logger),
        retry_error_callback=my_retry_error_callback
    )
    def ask_summary_results(self, prompt, history=None):
        if history is None:
            history = []

        messages = history + [{"role": "user", "content": prompt}]
        history = messages
        payload = {
            'model': self.model_name,
            'messages': messages,
            'logprobs': self.logprobs,
            # 'top_logprobs' : 2,
            # 'ask': False
        }

        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        resp = requests.request("POST", url, headers=headers, json=payload)
        if resp and resp.json().get('data').get('response'):
            data = resp.json()['data']['response']
            if data.get('choices'):
                reply = data['choices'][0]['message']
                history.append({'role': 'assistant', 'content': reply['content']})
                if self.logprobs:
                    return reply['content'], data['choices'][0]['logprobs']
                else:
                    return reply['content'], history
            else:
                raise Exception("请求失败。\nMessage: {}".format(resp.text))
        else:
            raise Exception("请求失败。\nMessage: {}".format(resp.text))

    @retry(
        wait=wait_random_exponential(min=10, max=50),
        stop=stop_after_attempt(5),
        after=general_after_log(logger),
        retry_error_callback=my_retry_error_callback
    )
    def get_reward_with_Qwen(self, prompt, history=None):
        if history is None:
            history = []

        messages = history + [{"role": "user", "content": prompt}]
        history = messages

        dashscope.api_key = 'sk-your key'
        response = dashscope.Generation.call(
            model='qwen-max-longcontext',  # llama3-8b-instruct qwen-max-longcontext
            messages=messages,
            result_format='message',  # set the result to be "message" format.
        )

        if response.status_code != HTTPStatus.OK:
            raise Exception('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))

        if response and response.get('output'):
            data = response['output']
            if data.get('choices'):
                reply = data['choices'][0]['message']
                history.append({'role': 'assistant', 'content': reply['content']})
                return self.parse_value_score(reply['content']), history
            else:
                raise Exception("请求失败。\nMessage: {}".format(response))
        else:
            raise Exception("请求失败。\nMessage: {}".format(response))


def run_FilesExtraction(workspace_metainfo: str, question: str):
    print('workspace:', workspace_metainfo)
    print('question:', question)

    # : gpt-4-0125-x-preview, gpt-4o-2024-05-x-13, gpt-4-1106-x-preview
    argsdict = {'fin':None, "fout":None, "prompt":None, 'code':None, "n_workers":60, "model_name":"gpt-4o-2024-05-13",
                "n_samples":0}

    chat_gpt = ChatGPT(argsdict)
    prompt = CONTEXT_TEMPLATE.format(workspace_metainfo=workspace_metainfo, question=question)
    # res, _ = chat_gpt.chat_with_Qwen(prompt)
    res, _ = chat_gpt.chat(prompt)
    return res

def get_RewardValue(issue: str, code_content: str, method_type:str, method_name:str, rel_file_path:str):
    argsdict = {'fin': None, "fout": None, "prompt": None, 'code': None, "n_workers": 60,
                "model_name": "gpt-4o-2024-05-13",
                "n_samples": 0}
    chat_gpt = ChatGPT(argsdict)
    # # {method_type} method {method_name} in {rel_file_path} file
    prompt = REWARD_TEMPLATE.format(issue=issue, method_type=method_type, method_name=method_name,
                                    rel_file_path=rel_file_path, code_content=code_content)
    # res, _ = chat_gpt.get_reward_with_Qwen(prompt)
    res, _, input_tokens, output_tokens = chat_gpt.get_reward(prompt)
    return res, input_tokens, output_tokens

def get_SummaryResults(prompt, history=None):
    argsdict = {'fin': None, "fout": None, "prompt": None, 'code': None, "n_workers": 60,
                "model_name": "gpt-4o-2024-05-13",  
                "n_samples": 0}
    chat_gpt = ChatGPT(argsdict)
    res, _ = chat_gpt.ask_summary_results(prompt)
    return res


if __name__ == '__main__':
    workspace_metainfo = """
    - src/marshmallow/__init__.py
    - src/marshmallow/base.py
    - src/marshmallow/class_registry.py
    - src/marshmallow/decorators.py
    - src/marshmallow/error_store.py
    - src/marshmallow/exceptions.py
    - src/marshmallow/fields.py
    - src/marshmallow/orderedset.py
    - src/marshmallow/schema.py
    - src/marshmallow/utils.py
    - src/marshmallow/validate.py
    - performance/benchmark.py
    - examples/flask_example.py
    - reporduce_bug.py
    - setup.py
    """
    question = """
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
    logs = """
    Traceback (most recent call last):
  File "/Users/mayingwei/Documents/code/QRepoAgent/examples/devin_swe/dev/marshmallow-code__marshmallow/3.0/codebase/marshmallow-code__marshmallow__3.0/reproduce_bug.py", line 7, in <module>
    s = MySchema()
  File "/Users/mayingwei/anaconda3/envs/swe-bench/lib/python3.9/site-packages/marshmallow-3.0.0-py3.9.egg/marshmallow/schema.py", line 383, in __init__
    self.fields = self._init_fields()
  File "/Users/mayingwei/anaconda3/envs/swe-bench/lib/python3.9/site-packages/marshmallow-3.0.0-py3.9.egg/marshmallow/schema.py", line 913, in _init_fields
    self._bind_field(field_name, field_obj)
  File "/Users/mayingwei/anaconda3/envs/swe-bench/lib/python3.9/site-packages/marshmallow-3.0.0-py3.9.egg/marshmallow/schema.py", line 969, in _bind_field
    field_obj._bind_to_schema(field_name, self)
  File "/Users/mayingwei/anaconda3/envs/swe-bench/lib/python3.9/site-packages/marshmallow-3.0.0-py3.9.egg/marshmallow/fields.py", line 636, in _bind_to_schema
    self.inner._bind_to_schema(field_name, self)
  File "/Users/mayingwei/anaconda3/envs/swe-bench/lib/python3.9/site-packages/marshmallow-3.0.0-py3.9.egg/marshmallow/fields.py", line 1117, in _bind_to_schema
    or getattr(schema.opts, self.SCHEMA_OPTS_VAR_NAME)
AttributeError: 'List' object has no attribute 'opts'
    """
    # import asyncio
    # asyncio.run(test_FilesExtraction(workspace_metainfo, question))
    res = run_FilesExtraction(workspace_metainfo, question)
    print(res)
