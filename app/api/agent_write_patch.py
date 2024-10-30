"""
An agent, which is only responsible for the write_patch tool call.
"""

import os
import json
from copy import deepcopy
from os.path import join as pjoin
from typing import Tuple
from pathlib import Path
import datetime

from app import globals
from app.analysis.sbfl import MethodId
from app.api import agent_common, validation
from app.data_structures import MessageThread
from app.log import log_and_print
from app.model.gpt import call_gpt
from app.post_process import (
    ExtractStatus,
    extract_diff_one_instance,
    record_extract_status,
)

SYSTEM_PROMPT = """You are a software developer maintaining a large project.
You are working on an issue submitted to your project.
You ultimate goal is to write a patch that resolves this issue.
"""

# refined to swe-gpt, don't omit.
USER_PROMPT_INIT = """Write a patch for the issue, based on the retrieved context. You can import necessary libraries.
Return the patch in the format below. Within <file></file>, replace "..." with actual file path. Within <original></original>, replace "..." with the original code snippet from the program. If the original code snippet is too long, you can modify it in multiple sections. Be careful not to modify too long original code at one time to avoid errors. In addition, the original code snippet must not be abbreviated (such as ...), otherwise it cannot be matched to the original location. Within <patched></patched>, replace "..." with the fixed version of the original code. When adding orignal code and updated code, pay attention to indentation, as the code is in Python.
You can write multiple modifications if needed. 

# modification 1
```python
<file>...</file>
<original>...</original>
<patched>...</patched>
```

# modification 2
```python
<file>...</file>
<original>...</original>
<patched>...</patched>
```

# modification 3
...
"""


def run_with_retries(
    logger,
    message_thread: MessageThread,
    output_dir: str,
    project_path,
    test_cmd,
    pre_test_cmd,
    repo_name,
    env_name,
    task_id: str,
    testcases_passing,
    testcases_failing,
    is_pre_test,
    retries=3,
) -> Tuple[str, float, int, int]:
    """
    Since the agent may not always write an applicable patch, we allow for retries.
    This is a wrapper around the actual run.
    """
    # (1) replace system prompt
    messages = deepcopy(message_thread.messages)
    new_thread: MessageThread = MessageThread(messages=messages)

    # (2) add the initial user prompt
    new_thread.add_user(USER_PROMPT_INIT)

    can_stop = False
    result_msg = ""

    all_cost = 0.0
    all_input_tokens = 0
    all_output_tokens = 0

    now = datetime.datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d-%H-%M-%S")

    valid_first_flag = False
    for i in range(1, retries + 2):
        if i > 1:
            debug_file = pjoin(output_dir, f"debug_agent_write_patch_{i - 1}.json")
            if globals.multi_patch_sample:
                if os.path.exists(pjoin(output_dir, f"debug_agent_write_patch_1.json")) or os.path.exists(pjoin(output_dir, f"debug_agent_write_patch_2.json")) or os.path.exists(pjoin(output_dir, f"debug_agent_write_patch_3.json")):
                    debug_file = pjoin(output_dir, f"debug_agent_write_patch_{i - 1}_{formatted_datetime}.json")
            with open(debug_file, "w") as f:
                json.dump(new_thread.to_msg(), f, indent=4)

        if can_stop or i > retries:
            break

        log_and_print(logger, f"Trying to write a patch. Try {i} of {retries}.")

        raw_patch_file = pjoin(output_dir, f"agent_patch_raw_{i}")
        if globals.multi_patch_sample:
            if os.path.exists(pjoin(output_dir, f"agent_patch_raw_1")) or os.path.exists(pjoin(output_dir, f"agent_patch_raw_2")) or os.path.exists(pjoin(output_dir, f"agent_patch_raw_3")):
                raw_patch_file = pjoin(output_dir, f"agent_patch_raw_{i}_{formatted_datetime}")
        
        # actually calling gpt
        res_text, _, _, cost, input_tokens, output_tokens = call_gpt(
            logger, new_thread.to_msg()
        )
        if len(res_text) <= 5:
            log_and_print(logger, f"network error in write patch {i}. Trying again.")
            res_text, _, _, cost, input_tokens, output_tokens = call_gpt(
                logger, new_thread.to_msg()
            )

        all_cost += cost
        all_input_tokens += input_tokens
        all_output_tokens += output_tokens

        new_thread.add_model(res_text, [])  # no tools

        log_and_print(
            logger, f"Raw patch produced in try {i}. Writing patch into file."
                    f"\nRaw_patch_file: {raw_patch_file}\noutput_dir: {output_dir}\nproject_path: {project_path}"
        )

        with open(raw_patch_file, "w") as f:
            f.write(res_text)

        # Attemp to extract a real patch from the raw patch
        diff_file = pjoin(output_dir, f"extracted_patch_{i}.diff")
        if globals.multi_patch_sample:
            if os.path.exists(pjoin(output_dir, f"extracted_patch_1.diff")) or os.path.exists(pjoin(output_dir, f"extracted_patch_2.diff")) or os.path.exists(pjoin(output_dir, f"extracted_patch_3.diff")):
                diff_file = pjoin(output_dir, f"extracted_patch_{i}_{formatted_datetime}.diff")
            
        extract_status, extract_msg = extract_diff_one_instance(
            raw_patch_file, diff_file
        )

        # record the extract status. This is for classifying the task at the end of workflow
        record_extract_status(output_dir, extract_status)

        if extract_status == ExtractStatus.APPLICABLE_PATCH:
            # patch generated is applicable and all edits are ok, so we can think about validation
            patch_content = Path(diff_file).read_text()
            log_and_print(
                logger,
                f"```diff\n{patch_content}\n```",
            )

            if globals.enable_validation:
                # pass2pass
                log_and_print(
                    logger,
                    f"**enable_validation**: true.",
                )
                # if 第一次valid: 先预加一个null的patch进行测试;
                if not valid_first_flag:
                    valid_first_flag = True
                    import sys
                    sys.path.append("/opt/SWE-bench/metrics")
                    from getters import get_logs_eval
                    # 传入null的model_patch
                    NOOP_PATCH = (
                        "diff --git a/empty.file.{nonce}.ignore b/empty.file.{nonce}.ignore\n"
                        "new file mode 100644\n"
                        "index 0000000..e69de29\n"
                    )
                    temp_model_patch = NOOP_PATCH.format(nonce="model_patch")
                    temp_diff_path = pjoin(output_dir, f"temp_null.diff")
                    with open(temp_diff_path, 'w') as f:
                        f.write(temp_model_patch)
                    run_test_suite_log_file = pjoin(output_dir, f"{task_id}_run_pre_test_suite.log")

                    patch_is_correct, err_message, all_test_passed = validation.validate(
                        temp_diff_path,
                        repo_name,
                        output_dir,
                        project_path,
                        test_cmd,
                        pre_test_cmd,
                        env_name,
                        testcases_passing,
                        testcases_failing,
                        run_test_suite_log_file,
                        logger,
                        is_pre_test,
                    )
                    before_eval_sm, pretest_valid = get_logs_eval(run_test_suite_log_file)
                    log_and_print(
                        logger,
                        f"Pre-test validation result: {patch_is_correct}",
                    )
                # end

                # if we have a patch extracted, apply it and validate
                run_test_suite_log_file = pjoin(output_dir, f"{task_id}_run_test_suite_{i}.log")
                patch_is_correct, err_message, all_test_passed = validation.validate(
                    diff_file,
                    repo_name,
                    output_dir,
                    project_path,
                    test_cmd,
                    pre_test_cmd,
                    env_name,
                    testcases_passing,
                    testcases_failing,
                    run_test_suite_log_file,
                    logger,
                    is_pre_test,
                )
                after_eval_sm, _ = get_logs_eval(run_test_suite_log_file)
                some_true_passed = True
                from log_parsers import TestStatus
                for case, status in before_eval_sm.items():
                    if status == TestStatus.PASSED.value:
                        if after_eval_sm[case] != TestStatus.PASSED.value:
                            log_and_print(
                                logger,
                                f"Patch does not pass the prepassed tests: {case}",
                            )
                            some_true_passed = False

                # if patch_is_correct:
                if all_test_passed or some_true_passed:
                    result_msg = (
                        "Written a patch that passed pre_test. Congratulations!"
                    )
                    new_thread.add_user(result_msg)  # just for logging
                    print('**pre_test** passed!')
                    log_and_print(
                        logger,
                        '**pre_test** passed!',
                    )
                    can_stop = True
                # the following two branches cannot be swapped, because
                # --enable-perfect-angelic is meant to override --enable-angelic
                elif globals.enable_perfect_angelic:
                    msg = (
                        f"Written an applicable patch, but it did not resolve the issue. Error message: {err_message}.",
                    )

                    incorrect_locations = validation.perfect_angelic_debug(
                        task_id, diff_file, project_path
                    )
                    angelic_msg = angelic_debugging_message(incorrect_locations)

                    result_msg = f"{msg}\n{angelic_msg}"
                    new_thread.add_user(result_msg)
                    continue
                elif globals.enable_angelic:
                    raise NotImplementedError(
                        "Angelic debugging has not been integrated"
                    )
                else:
                    new_thread.messages.pop()
                    print('Pass2Pass error, again!')
                    continue
            elif globals.enable_perfect_angelic:
                incorrect_locations = validation.perfect_angelic_debug(
                    task_id, diff_file, project_path
                )

                msg = "Extracted a patch."
                if angelic_msg := angelic_debugging_message(incorrect_locations):
                    result_msg = f"{msg}\n{angelic_msg}"
                else:
                    result_msg = msg

                new_thread.add_user(result_msg)
                continue
            elif globals.enable_angelic:
                raise NotImplementedError("Angelic debugging has not been integrated")
            else:
                result_msg = "Extracted a patch. You should review the patch for correctness later on."
                new_thread.add_user(result_msg)  # just for logging
                can_stop = True
                print(result_msg)

        else:
            # we dont have a valid patch
            new_prompt = (
                "Your edit could not be applied to the program. "
                + extract_msg
                + " Please try again."
            )
            new_thread.add_user(new_prompt)
            result_msg = "Failed to write a valid patch."

    return result_msg, all_cost, all_input_tokens, all_output_tokens


def angelic_debugging_message(incorrect_locations: list[tuple[str, MethodId]]) -> str:
    msg = []

    if incorrect_locations:
        msg.append("The following methods should not have been changed:")
        msg.extend(
            f"    {filename}: {method_id!s}"
            for filename, method_id in incorrect_locations
        )

    return "\n".join(msg)
