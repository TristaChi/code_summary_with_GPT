import time
import os
import run_GPT
import openai
import threading
from threading import Thread


file_name = '[direction to result file]'
data_dir = '[directoin to data]'
use_GPT = True

def rerun_program(stop_event):
    try:
        with open(file_name, 'r') as f:
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
        # Periodically check if stop_event is set, and exit if it is
        if stop_event.is_set():
            return
    except (openai.error.RateLimitError, openai.error.ServiceUnavailableError) as e:
        print(f"Error encountered: {e}")


def main():
    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            f.write('index, src_response, adv_response, src_w_fsd_response, adv_w_fsd_response, src_w_pd_response, adv_w_pd_response, src_w_d_response, adv_w_d_response, src_idk_response, adv_idk_response, src_w_fsd_idk_response, adv_w_fsd_idk_response, src_w_pd_idk_response, adv_w_pd_idk_response, src_w_d_idk_response, adv_w_d_idk_response, tgt \n')
    stop_event = threading.Event()
    while True:
        with open(file_name, 'r') as f:
            line_count = sum(1 for _ in f)
            print('line count:', line_count)
        
        if line_count >= 1000:
            print("The file has reached 1000 lines. Exiting...")
            break

        try:
            stop_event.clear()
            thread = threading.Thread(target=rerun_program, args=(stop_event,))
            thread.start()
            thread.join(timeout=80)
            if thread.is_alive():
                print("Execution took too long. Restarting...")
                stop_event.set()  # Signal the thread to stop
        except Exception as e:
            print("Sleeping for 30 seconds before next loop...")
            time.sleep(30)
            print(f"An error occurred: {e}")
            continue

        print("Sleeping for 30 seconds before next loop...")
        time.sleep(30)

if __name__ == "__main__":
    main()
