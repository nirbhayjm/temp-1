import re
import sys
from subprocess import Popen
from visualize import VisdomLogger
from utilities import confirm

if __name__ == '__main__':
    assert len(sys.argv) == 2
    env_regex = sys.argv[1]
    print(f"Finding visdom envs with regex: {env_regex}")

    viz = VisdomLogger(env_name='main',
                        server='localhost',
                        port=8896)
    env_list = viz.viz.get_env_list()
    regex = re.compile(env_regex)

    matched_list = [env for env in env_list if regex.match(env)]
    jobid_list = []

    print(f"Found {len(matched_list)} matches:")
    null = None
    false = False
    true = True
    for env in matched_list:
        win_data = viz.viz.get_window_data(env=env, win=None)
        win_dict = eval(win_data)
        text_window_key = None
        for key in win_dict.keys():
            if 'window' in key:
                text_window_key = key
        assert text_window_key is not None
        html_content = win_dict[text_window_key]['content']

        jobid_raw = re.split('jobid', html_content)[1][:30]
        jobid = re.compile("[0-9]+").search(jobid_raw).group()
        jobid_list.append(jobid)
        print(f" [{jobid}] > {env}")

    if len(matched_list) > 0:
        if confirm(f"Confirm killing of {len(matched_list)} slurm jobs?"):
            for jobid in jobid_list:
                Popen(f"scancel {jobid}", shell=True)
            print("Successfully killed slurm jobs")
        else:
            print("Skipping job kill")


        if confirm(f"Confirm delete of {len(matched_list)} envs?"):
            for env in matched_list:
                viz.viz.delete_env(env)
            print("Successfully deleted visdom envs")
        else:
            print("Skipping delete")
    else:
        print("No match found.")
