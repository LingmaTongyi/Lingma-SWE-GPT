"""
The main driver.
"""

import argparse
import datetime
import json
import subprocess
from multiprocessing import Pool
from os.path import join as pjoin
from subprocess import CalledProcessError
from typing import Dict, List, Mapping, Optional, Tuple
import os
import shutil

from app import globals, globals_mut, inference, log
from app import utils as apputils
from app.api.manage import ProjectApiManager
from app.post_process import (
    extract_organize_and_form_input,
    organize_and_form_input,
    reextract_organize_and_form_inputs,
)


def parse_task_list_file(task_list_file: str) -> List[str]:
    """
    Parse the task list file.
    The file should contain one task/instance id per line, without other characters.
    """
    with open(task_list_file, "r") as f:
        task_ids = f.readlines()
    return [x.strip() for x in task_ids]


class Task:
    """
    Encapsulate everything required to run one task.
    """

    def __init__(
        self, task_counter: str, task_id: str, setup_info: Dict, task_info: Dict
    ):
        # a counter str, format "1/150", which means first task out of 150
        self.task_counter = task_counter
        # id from the benchmark
        self.task_id = task_id
        # setup_info (Dict): keys: ['repo_path', 'env_name', 'pre_install', 'install','test_cmd']
        self.setup_info = setup_info
        # task_info (Dict): keys: ['base_commit', 'hints_text', 'created_at',
        # 'test_patch', 'repo', 'problem_statement', 'version', 'instance_id',
        # 'FAIL_TO_PASS', 'PASS_TO_PASS', 'environment_setup_commit']
        self.task_info = task_info

def remove_directory_if_exists(directory_path):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)
        print(f"Removed directory: {directory_path}")
    else:
        print(f"Directory does not exist: {directory_path}")


def run_one_task(task: Task, loop_times=1) -> bool:
    """
    High-level entry for running one task.

    Args:
        - task: The Task instance to run.

    Returns:
        Whether the task completed successfully.
    """
    source_temperature = globals.model_temperature
    for iter_location in range(loop_times):
        task_id = task.task_id
        setup_info = task.setup_info
        task_info = task.task_info
        src_setup_info_repo_path = setup_info["repo_path"]
        repo_path = setup_info["repo_path"]
        env_name = setup_info["env_name"]
        pre_install_cmds = setup_info["pre_install"]
        install_cmd = setup_info["install"]
        # command to run the relevant tests
        test_cmd = setup_info["test_cmd"]
        if "pre_test_cmd" in setup_info:
            pre_test_cmd = setup_info["pre_test_cmd"]
        else:
            pre_test_cmd = []
        is_pre_test = globals.is_pre_test
        model_name = globals.model

        base_commit = task_info["base_commit"]
        problem_stmt = task_info["problem_statement"]
        repo_name = task_info["repo"]
        # modifications to the test suite for this task instance
        test_patch = task_info["test_patch"]
        testcases_passing = task_info["PASS_TO_PASS"]
        testcases_failing = task_info["FAIL_TO_PASS"]

        # use time as part of folder name so it's always unique
        start_time = datetime.datetime.now()
        start_time_s = start_time.strftime("%Y-%m-%d_%H-%M-%S")
        #if 'vanilla-lite' not in globals.output_dir:
        #    globals.output_dir = global_output_dir
        task_output_dir = pjoin(globals.output_dir, task_id + "_" + start_time_s)
        print(f'task_output_dir: {task_output_dir}')

        apputils.create_dir_if_not_exists(task_output_dir)
        
        # swebench 上的测评
        if 'SWE-bench/repos/testbed' in repo_path:
            print('model_name judge temp dir:', model_name)
            if 'gpt' in model_name:
                temp_repo_path = repo_path.replace('SWE-bench/repos/testbed', 'SWE-bench/repos/testbed_temp')
            elif 'claude' in model_name:
                temp_repo_path = repo_path.replace('SWE-bench/repos/testbed', 'SWE-bench/repos/testbed_temp_1')
            elif 'gemini' in model_name:
                temp_repo_path = repo_path.replace('SWE-bench/repos/testbed', 'SWE-bench/repos/testbed_temp_2')
            else:
                temp_repo_path = repo_path.replace('SWE-bench/repos/testbed', 'SWE-bench/repos/testbed_temp_3')

            temp_repo_path = os.path.dirname(temp_repo_path)
            # if temp_path exists, delete
            remove_directory_if_exists(temp_repo_path)
            apputils.create_dir_if_not_exists(temp_repo_path)
            copy_cmd = ["cp", "-R", repo_path, temp_repo_path]
            print('temp_repo_path', temp_repo_path)
            try:
                if not os.path.exists(repo_path):
                    print(f"Source directory does not exist: {repo_path}")
                result = subprocess.run(copy_cmd, capture_output=True, text=True, check=True)
                print(f"Copy operation completed successfully.\nsrc_path is {repo_path}\n Path is {temp_repo_path}.\n{result.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"Copy operation failed with return code {e.returncode}.")
                print(f"Error output: {e.stderr}")
            except FileNotFoundError as e:
                print(f"Command not found: {e}")

            temp_repo_path = repo_path.replace('SWE-bench/repos/testbed', 'SWE-bench/repos/testbed_temp')
            if 'gpt' in model_name:
                temp_repo_path = repo_path.replace('SWE-bench/repos/testbed', 'SWE-bench/repos/testbed_temp')
            elif 'claude' in model_name:
                temp_repo_path = repo_path.replace('SWE-bench/repos/testbed', 'SWE-bench/repos/testbed_temp_1')
            elif 'gemini' in model_name:
                temp_repo_path = repo_path.replace('SWE-bench/repos/testbed', 'SWE-bench/repos/testbed_temp_2')
            else:
                temp_repo_path = repo_path.replace('SWE-bench/repos/testbed', 'SWE-bench/repos/testbed_temp_3')

            if os.path.exists(temp_repo_path):
                print(f"Congras! Temporary directory exists: {temp_repo_path}")
            else:
                print(f"Temporary directory was not created: {temp_repo_path}")
            repo_path = temp_repo_path
            setup_info["repo_path"] = repo_path
        else:
            print(f"No testbed error in repo_path: {repo_path}")
        # end

        # commit_hash = get_current_commit_hash()
        commit_hash = '1111111'
        # save some meta data and other files for convenience
        meta = {
            "task_id": task_id,
            "setup_info": setup_info,
            "task_info": task_info,
        }
        with open(pjoin(task_output_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)
        with open(pjoin(task_output_dir, "problem_statement.txt"), "w") as f:
            f.write(problem_stmt)
        with open(pjoin(task_output_dir, "developer_patch.diff"), "w") as f:
            f.write(task_info["patch"])

        logger = log.create_new_logger(task_id, task_output_dir)
        log.log_and_always_print(
            logger,
            f"============= Running task {task_id} =============",
        )

        try:
            api_manager = ProjectApiManager(
                task_id,
                repo_path,
                base_commit,
                env_name,
                repo_name,
                pre_install_cmds,
                install_cmd,
                test_cmd,
                pre_test_cmd,
                is_pre_test,
                test_patch,
                testcases_passing,
                testcases_failing,
                task_output_dir,
                do_install=globals.do_install,
            )
        except Exception as e:
            log.log_exception(logger, e)
            run_status_message = f"Task {task_id} failed with exception: {e}."
            print(run_status_message)
            
            if 'testbed_temp' in repo_path:
                apputils.remove_temp_dir(repo_path)
            return False
        finally:
            # 复原
            setup_info["repo_path"] = src_setup_info_repo_path
            
        # run inference and catch error
        run_ok = False
        run_status_message = ""
        DEBUG = False
        if not DEBUG:
            try:
                # create api manager and run project initialization routine in its init
                if globals.load_cache is not None:
                    # NOTE: although we start from a history state, still creating a new
                    # output folder to store results from this run
                    run_ok = inference.continue_task_from_cache(
                        globals.load_cache, task_output_dir, api_manager
                    )
                else:
                    run_ok = inference.run_one_task(task_output_dir, api_manager, problem_stmt)
                if run_ok:
                    run_status_message = f"Task {task_id} completed successfully."
                else:
                    run_status_message = f"Task {task_id} failed without exception."
            except Exception as e:
                log.log_exception(logger, e)
                run_status_message = f"Task {task_id} failed with exception: {e}."
                run_ok = False
                raise e
            finally:
                # dump recorded tool call sequence into a file
                end_time = datetime.datetime.now()

                api_manager.dump_tool_call_sequence_to_file()
                api_manager.dump_tool_call_layers_to_file()

                input_cost_per_token = globals.MODEL_COST_PER_INPUT[globals.model]
                output_cost_per_token = globals.MODEL_COST_PER_INPUT[globals.model]
                with open(pjoin(task_output_dir, "cost.json"), "w") as f:
                    json.dump(
                        {
                            "model": globals.model,
                            "commit": commit_hash,
                            "input_cost_per_token": input_cost_per_token,
                            "output_cost_per_token": output_cost_per_token,
                            "total_input_tokens": api_manager.input_tokens,
                            "total_output_tokens": api_manager.output_tokens,
                            "total_tokens": api_manager.input_tokens
                            + api_manager.output_tokens,
                            "total_cost": api_manager.cost,
                            "start_epoch": start_time.timestamp(),
                            "end_epoch": end_time.timestamp(),
                            "elapsed_seconds": (end_time - start_time).total_seconds(),
                        },
                        f,
                        indent=4,
                    )

                # at the end of each task, reset everything in the task repo to clean state
                with apputils.cd(repo_path):
                    apputils.repo_reset_and_clean_checkout(base_commit, logger)

                # delete temp_path (myw, mingwei);
                if 'testbed_temp' in repo_path:
                    apputils.remove_temp_dir(repo_path)

                log.log_and_always_print(logger, run_status_message)


        location_file = "locations_rate.json"
        location_file_path = os.path.join(task_output_dir, location_file)
        if os.path.exists(location_file_path):
            with open(location_file_path, 'r') as f:
                locations_rate = json.load(f)
                from_patch = locations_rate.get("from_patch", False)
                from_agent_proxy = locations_rate.get("from_agent_proxy", False)
                if from_patch and from_agent_proxy:
                    if from_patch['function_line'] == 1 or from_agent_proxy['function_line'] == 1:
                        print(f"## location successful in iter {iter_location}! ## from_patch: {from_patch['function_line']}, from_agent_proxy: {from_agent_proxy['function_line']}")
                        log.log_and_cprint(logger, f"## location successful in iter {iter_location}! ## from_patch: {from_patch['function_line']}, from_agent_proxy: {from_agent_proxy['function_line']}")

        if loop_times > 1:
            globals.model_temperature = 0.8
    globals.model_temperature = source_temperature
    return run_ok

def get_current_commit_hash() -> str:
    command = ["git", "rev-parse", "HEAD"]
    cp = subprocess.run(command, text=True, capture_output=True)
    try:
        cp.check_returncode()
        return cp.stdout.strip()
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to get SHA-1 of HEAD: {cp.stderr}") from e


def run_task_group(task_group_id: str, task_group_items: List[Task]) -> None:
    """
    Run all tasks in a task group sequentially.
    Main entry to parallel processing.
    """
    log.print_with_time(
        f"Starting process for task group {task_group_id}. Number of tasks: {len(task_group_items)}."
    )
    for task in task_group_items:
        # within a group, the runs are always sequential
        run_one_task(task, loop_times=1)
        log.print_with_time(globals_mut.incre_task_return_msg())

    log.print_with_time(
        f"{globals_mut.incre_task_group_return_msg()} Finished task group {task_group_id}."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setup-map",
        type=str,
        help="Path to json file that contains the setup information of the projects.",
    )
    parser.add_argument(
        "--tasks-map",
        type=str,
        help="Path to json file that contains the tasks information.",
    )
    ## where to store run results
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to the directory that stores the run results.",
    )
    ## which tasks to be run
    parser.add_argument(
        "--task-list-file",
        type=str,
        help="Path to the file that contains all tasks ids to be run.",
    )
    parser.add_argument("--task", type=str, help="Task id to be run.")
    parser.add_argument(
        "--num-processes",
        type=str,
        default=1,
        help="Number of processes to run the tasks in parallel.",
    )
    parser.add_argument(
        "--load-cache",
        type=str,
        help="(Deprecated) Point to a json file which contains past conversation history. "
        "Restart conversation from this file instead of starting from scratch. "
        "Only available when running a single task.",
    )
    parser.add_argument(
        "--enable-sbfl", action="store_true", default=False, help="Enable SBFL."
    )
    parser.add_argument(
        "--enable-layered",
        action="store_true",
        default=False,
        help="Enable layered code search.",
    )
    parser.add_argument(
        "--enable-validation",
        action="store_true",
        default=False,
        help="Enable validation in our workflow.",
    )
    parser.add_argument(
        "--enable-angelic",
        action="store_true",
        default=False,
        help="(Experimental) Enable angelic debugging",
    )
    parser.add_argument(
        "--enable-perfect-angelic",
        action="store_true",
        default=False,
        help="(Experimental) Enable perfect angelic debugging; overrides --enable-angelic",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        default=False,
        help="Do not print most messages to stdout.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo-0125",
        choices=globals.MODELS,
        help="The model to use. Currently only OpenAI models are supported.",
    )
    parser.add_argument(
        "--model-temperature",
        type=float,
        default=0.0,
        help="The model temperature to use, for OpenAI models.",
    )
    parser.add_argument(
        "--conv-round-limit",
        type=int,
        default=15,
        help="Conversation round limit for the main agent.",
    )
    parser.add_argument(
        "--extract-patches",
        type=str,
        help="Only extract patches from the raw results dir. Voids all other arguments if this is used.",
    )
    parser.add_argument(
        "--re-extract-patches",
        type=str,
        help="same as --extract-patches, except that individual dirs are moved out of their categories first",
    )
    parser.add_argument(
        "--save-sbfl-result",
        action="store_true",
        default=False,
        help="Special mode to only save SBFL results for future runs.",
    )
    parser.add_argument(
        "--multi-patch-sample",
        action="store_true",
        default=False,
        help="(Experimental) Enable multi-patch sampling",
    )
    args = parser.parse_args()
    setup_map_file = args.setup_map
    tasks_map_file = args.tasks_map
    globals.output_dir = args.output_dir
    if globals.output_dir is not None:
        globals.output_dir = apputils.convert_dir_to_absolute(globals.output_dir)
    task_list_file = args.task_list_file
    task_id = args.task
    num_processes: int = int(args.num_processes)
    globals.load_cache = args.load_cache
    globals.model = args.model
    globals.model_temperature = args.model_temperature
    # set whether brief or verbose log
    print_stdout: bool = not args.no_print
    log.print_stdout = print_stdout
    globals.enable_sbfl = args.enable_sbfl
    globals.enable_layered = args.enable_layered
    globals.enable_validation = args.enable_validation
    globals.enable_angelic = args.enable_angelic
    globals.enable_perfect_angelic = args.enable_perfect_angelic
    globals.conv_round_limit = args.conv_round_limit
    globals.multi_patch_sample = args.multi_patch_sample

    # special modes
    extract_patches: Optional[str] = args.extract_patches
    globals.only_save_sbfl_result = args.save_sbfl_result

    if globals.only_save_sbfl_result and extract_patches is not None:
        raise ValueError(
            "Cannot save SBFL result and extract patches at the same time."
        )

    # special mode 1: extract patch, for this we can early exit
    if args.re_extract_patches is not None:
        extract_patches = apputils.convert_dir_to_absolute(args.re_extract_patches)
        reextract_organize_and_form_inputs(args.re_extract_patches)
        return

    if extract_patches is not None:
        extract_patches = apputils.convert_dir_to_absolute(extract_patches)
        extract_organize_and_form_input(extract_patches)
        return

    globals.do_install = (
        globals.enable_sbfl
        or globals.enable_validation
        or globals.only_save_sbfl_result
    )

    # check parameters
    if task_id is not None and task_list_file is not None:
        raise ValueError("Cannot specify both task and task-list.")

    if globals.load_cache is not None and task_id is None:
        raise ValueError("Cannot load cache when not in single-task mode.")

    all_task_ids = []
    if task_list_file is not None:
        all_task_ids = parse_task_list_file(task_list_file)
    if task_id is not None:
        all_task_ids = [task_id]
    if len(all_task_ids) == 0:
        raise ValueError("No task ids to run.")

    with open(setup_map_file, "r") as f:
        setup_map = json.load(f)
    with open(tasks_map_file, "r") as f:
        tasks_map = json.load(f)

    apputils.create_dir_if_not_exists(globals.output_dir)
    print(f'globals.output_dir: {globals.output_dir}')
    num_tasks = len(all_task_ids)
    globals_mut.init_total_num_tasks(num_tasks)

    # for each task in the list to run, create a Task instance
    all_tasks = []
    all_task_ids = sorted(all_task_ids)
    for idx, task_id in enumerate(all_task_ids):
        setup_info = setup_map[task_id]
        task_info = tasks_map[task_id]
        task = Task(f"{idx+1}/{num_tasks}", task_id, setup_info, task_info)
        all_tasks.append(task)

    # group tasks based on repo-version; tasks in one group should
    # be executed in one thread
    # key: env_name (a combination of repo+version), value: list of tasks
    task_groups: Mapping[str, List[Task]] = dict()
    task: Task
    for idx, task in enumerate(all_tasks):
        # key = task.setup_info["env_name"]
        key = str(idx)
        if key not in task_groups:
            task_groups[key] = []
        task_groups[key].append(task)

    # print some info about task
    log.print_with_time(f"Total number of tasks: {num_tasks}")
    log.print_with_time(f"Total number of processes: {num_processes}")
    log.print_with_time(f"Task group info: (number of groups: {len(task_groups)})")
    for key, tasks in task_groups.items():
        log.print_with_time(f"\t{key}: {len(tasks)} tasks")

    # single process mode
    if num_processes == 1:
        log.print_with_time("Running in single process mode.")
        for task in all_tasks:
            run_one_task(task, loop_times=1)
            # break
        log.print_with_time("Finished all tasks sequentially.")

    # multi process mode
    else:
        # prepare for parallel processing
        num_task_groups = len(task_groups)
        globals_mut.init_total_num_task_groups(num_task_groups)
        num_processes = min(num_processes, num_task_groups)
        # If the function for Pool.map accepts multiple arguments, each argument should
        # be prepared in the form of a list for multiple processes.
        task_group_ids_items: List[Tuple[str, List[Task]]] = list(task_groups.items())
        task_group_ids_items = sorted(
            task_group_ids_items, key=lambda x: len(x[1]), reverse=True
        )
        log.print_with_time(
            f"Sorted task groups: {[x[0] for x in task_group_ids_items]}"
        )
        try:
            pool = Pool(processes=num_processes)
            pool.starmap(run_task_group, task_group_ids_items)
            pool.close()
            pool.join()
        finally:
            log.print_with_time("Finishing all tasks in the pool.")

    if globals.only_save_sbfl_result:
        log.print_with_time("Only saving SBFL results. Exiting.")
        return

    # post-process completed experiments to get input file to SWE-bench
    log.print_with_time("Post-processing completed experiment results.")
    swe_input_file = organize_and_form_input(globals.output_dir)
    log.print_with_time("SWE-Bench input file created: " + swe_input_file)


if __name__ == "__main__":
    main()
