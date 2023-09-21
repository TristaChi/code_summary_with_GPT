import time
import os
import run_GPT
import openai
from threading import Thread

file_name = '/Users/zhangchi/Desktop/attack_LLM/result/GPT4/GPT_result'
data_dir = '/Users/zhangchi/Desktop/attack_LLM/data/v2-92-z_o_5-pgd_3_smooth-asr45/tokens/sri/py150/gradient-targeting/test.tsv'
use_GPT = True

filepath = file_name+'.csv'
def rerun_program():
    with open(filepath, 'r') as f:
        line_count = sum(1 for _ in f)
    if line_count > 0:
        start_idx = line_count % 500
        bag = line_count // 500
    else:
        bag = 0
        start_idx = 0
    print(f"Rerunning program with start_idx={start_idx} and bag={bag}")
    run_GPT.main(bag, 
                start_idx,
                data_dir = data_dir,
                randomly_pick = False,
                bag_size = 500,
                file_name = file_name,
                new_sample = True,
                use_GPT = True
                )
    
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
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write('index, src_response, adv_response, src_w_fsd_response, adv_w_fsd_response, src_w_pd_response, adv_w_pd_response, src_w_d_response, adv_w_d_response, src_idk_response, adv_idk_response, src_w_fsd_idk_response, adv_w_fsd_idk_response, src_w_pd_idk_response, adv_w_pd_idk_response, src_w_d_idk_response, adv_w_d_idk_response, tgt \n')
    while True:        
        with open(filepath, 'r') as f:
            line_count = sum(1 for _ in f)
        
        if line_count >= 2500:
            print("The file has reached 2500 lines. Exiting...")
            break
        try:
            thread = Thread(target=rerun_program)
            thread.start()
            thread.join(timeout=80)
            if thread.is_alive():
                print("Execution took too long. Restarting...")
        except (openai.error.RateLimitError, openai.error.ServiceUnavailableError):
            print("Rate limit reached. Retrying with last line of file...")
            print("Sleeping for 20 seconds when error...")
            time.sleep(20)
            continue
        print("Sleeping for 20 seconds before nest loop...")
        time.sleep(20)


if __name__ == "__main__":
    main()
