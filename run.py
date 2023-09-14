import time
import os
import run_GPT
import openai
from threading import Thread

filepath = '/Users/zhangchi/Desktop/attack_LLM/result/few_shot_prompt_defense_all_wd_idk/GPT_result.csv'

def rerun_program():
    # idx = get_first_term_of_last_line(filepath)
    with open(filepath, 'r') as f:
        line_count = sum(1 for _ in f)
    if line_count > 0:
        start_idx = line_count % 500
        bag = line_count // 500
    else:
        bag = 0
        start_idx = 0
    print(f"Rerunning program with start_idx={start_idx} and bag={bag}")
    run_GPT.main(bag, start_idx)
    
def get_first_term_of_last_line(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        if lines:
            last_line = lines[-1]
            first_term = last_line.split(', ')[0]
            return int(first_term)+1
        else:
            return None

def main():
    while True:        
        with open(filepath, 'r') as f:
            line_count = sum(1 for _ in f)
        
        if line_count >= 2700:
            print("The file has reached 2700 lines. Exiting...")
            break
        try:
            thread = Thread(target=rerun_program)
            thread.start()
            thread.join(timeout=80)
            if thread.is_alive():
                print("Execution took too long. Restarting...")
        except (openai.error.RateLimitError, openai.error.ServiceUnavailableError):
            print("Rate limit reached. Retrying with last line of file...")
            print("Sleeping for 10 seconds when error...")
            time.sleep(10)
            continue
        print("Sleeping for 10 seconds before nest loop...")
        time.sleep(10)


if __name__ == "__main__":
    main()
