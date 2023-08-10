from myGPT import ChatGPT
import csv
import numpy as np
import re
import json
import argparse 

dir = '/Users/zhangchi/Desktop/code_robustness_of_LLM/dataset/seq2seq_z_rand_5-pgd_3_no_transforms_combined/test_adv.tsv'
# data_count = 50
randomly_pick = False
dict_size = 300
# start_idx = 0
# bag = 0
# file_name = 'result/rand/GPT_result_prompt_defence_'+str(bag)
file_name = 'result/few_shot_prompt_defense_all_wd_idk/GPT_result'
new_sample = True
run_single_example = False

prompt = "Choose one word from the provided dictionary to summarize the given piece of code. \n code:"
prompt_w_d = "Choose one word from the provided dictionary to summarize the given piece of code. Remove the if false statement and the print statement in the code before the summarization. \n code:"
prompt_idk = "Choose one word from the provided dictionary to summarize the given piece of code. Reply 'I don't know' if not sure. \n code:"
prompt_idk_w_d = "Choose one word from the provided dictionary to summarize the given piece of code. Remove the if false statement and the print statement in the code before the summarization. Reply 'I don't know' if not sure. \n code:"

# prompt = "Choose one word from the provided dictionary to summarize the given piece of code. The dictionary and code are as follows:\n dictionary: {} \n code:"
# prompt_w_d = "Choose one word from the provided dictionary to summarize the given piece of code. Remove the dead code and the print statement in the code before the summarization. The dictionary and code are as follows:\n dictionary: {} \n code:"

# prompt = "Use one word from the set {} to describe the following piece of code. Also, output your confidence in your choice as a probability between 0 and 1. Don’t provide any other description. The reply form should be \"word (confidence)\". \n"
# prompt_w_d = "Use one word from the set {} to describe the following piece of code.Please remove the dead code and the print statement in the code before the description. Also, output your confidence in your choice as a probability between 0 and 1. Don’t provide any other description. The reply form should be \"word (confidence)\" \n"

def get_data(index,src,tgt,adv,dict_size,bag=100000,picked=[]):
    if picked != []:
        print('using the selected samples...')
        rand_pick = picked
    elif (bag+1)*dict_size > len(src):
        print("randomly pick data from src")
        # randomly pick data
        rand_pick = np.random.choice(range(len(index)),size=dict_size) # type: ignore
    else:
        rand_pick = [num + bag*dict_size for num in range(dict_size)]
        # range(dict_size)+bag*dict_size
        print("pick data from ", rand_pick[0],"to ", rand_pick[-1])

    # index_r = [index[i] for i in rand_pick]
    src_r = [src[i] for i in rand_pick]
    tgt_r = [tgt[i] for i in rand_pick]
    adv_r = [adv[i] for i in rand_pick]
    print('rand_pick',rand_pick)
    return rand_pick,src_r,tgt_r,adv_r

def get_response(chatbot, 
                input, 
                dict, 
                prompt,
                prompt_w_d,
                temperature=0,
                top_p=1.0,
                defence=False, 
                save_p=False, 
                file_name='result/'):
    prompt = prompt.format(dict)+input
    
    prompt_w_d = prompt_w_d.format(dict)+input
        
    if defence: 
        response = chatbot(prompt_w_d,dict=dict,defence=True,temperature=temperature,top_p=top_p,max_tokens=30)
    else: response = chatbot(prompt,dict=dict,defence=False,temperature=temperature,top_p=top_p,max_tokens=10)
    return response

def read_file(dir,randomly_pick,dict_size=100,bag=0,picked=[]):
    # read from file and store in list
    with open(dir, 'r') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        first_row = next(reader)
        index = []
        src = []
        tgt = []
        adv = []
        # Iterate over each row in the TSV file
        for row in reader:
            # Access the values in each column
            index.append(row[0])
            src.append(row[1])
            tgt.append(row[2])
            adv.append(row[3])
    if picked != []:
        index_n,src_n,tgt_n,adv_n = get_data(index,src,tgt,adv,dict_size,picked=picked)
    if randomly_pick:
        index_n,src_n,tgt_n,adv_n = get_data(index,src,tgt,adv,dict_size)
    else: 
        index_n,src_n,tgt_n,adv_n = get_data(index,src,tgt,adv,dict_size,bag=bag)
        
    # create a set of words as dictionary
    print("original words",set(tgt_n))
    tgt_dict = {word.replace(' ', '_') for word in set(tgt_n)}
    print('tgt_dict',tgt_dict)
    return index_n,src_n,tgt_n,adv_n,tgt_dict

def run_GPT_dif_cases(chatbot,src_r,adv_r,tgt_r,tgt_dict,prompt,prompt_w_d,temperature=0,):
    print('-------- src ---------')
    # print('src code:',src_r[i])
    src_response = get_response(chatbot, src_r, tgt_dict, prompt,
                prompt_w_d, temperature=temperature)
    print('src response',src_response)

    print('-------- adv ---------')
    # print('adv code:',adv_r[i])
    adv_response = get_response(chatbot, adv_r, tgt_dict, prompt,
                prompt_w_d, temperature=temperature)
    print('adv response',adv_response)
    print('-------- src w/ defence ---------')
    # print('adv code:',adv_r[i])
    src_wd_response = get_response(chatbot, src_r, tgt_dict, prompt,
                prompt_w_d, defence=True, temperature=temperature)
    print('src w/ defence response',src_wd_response)

    print('-------- adv w/ defence ---------')
    # print('adv code:',adv_r[i])
    adv_wd_response = get_response(chatbot, adv_r, tgt_dict, prompt,
                prompt_w_d, defence=True, temperature=temperature)
    print('adv w/ defence response',adv_wd_response)

    print('-------- src and idk ---------')
    # print('adv code:',adv_r[i])
    src_idk_response = get_response(chatbot, src_r, tgt_dict, prompt_idk,
                prompt_idk_w_d, temperature=temperature)
    print('src w/ idk response',src_idk_response)

    print('-------- adv and idk ---------')
    # print('adv code:',adv_r[i])
    adv_idk_response = get_response(chatbot, adv_r, tgt_dict, prompt_idk,
                prompt_idk_w_d, temperature=temperature)
    print('adv w/ idk response',adv_idk_response)

    print('-------- src w/ defence and idk ---------')
    # print('adv code:',adv_r[i])
    src_wd_idk_response = get_response(chatbot, src_r, tgt_dict, prompt_idk,
                prompt_idk_w_d, defence=True, temperature=temperature)
    print('src w/ defence and idk response',src_wd_idk_response)

    print('-------- adv w/ defence and idk ---------')
    # print('adv code:',adv_r[i])
    adv_wd_idk_response = get_response(chatbot, adv_r, tgt_dict, prompt_idk,
                prompt_idk_w_d, defence=True, temperature=temperature)
    print('adv w/ defence and idk response',adv_wd_idk_response)

    print('-------- tgt ---------')
    print('tgt code:',tgt_r.replace(' ', '_'))

    # return src_response, adv_response, adv_wd_response, tgt_r.replace(' ', '_')
    return src_response, adv_response, src_wd_response, adv_wd_response, src_idk_response, adv_idk_response, src_wd_idk_response, adv_wd_idk_response, tgt_r.replace(' ', '_')

def save_picked_idx(file_name,picked):
    np.save(file_name+'_idx', picked) # type: ignore
    print('Saving the sampled index......')

def read_picked_idx(file_name):
    picked = np.load(file_name+'_idx.npy') # type: ignore
    print('Reading the sampled index: ',picked)
    return picked

def save_dict(file_name,tgt_dict):
    with open(file_name+'.dict', 'w') as d:
        json.dump(list(tgt_dict), d)
        print('Saving the dictionary......')

def load_dict(file_name):
    with open(file_name+'.dict', 'r') as d:
        tgt_dict = json.load(d)
    return tgt_dict

def save_prompt(file_name,prompt,prompt_w_d):
    with open(file_name+'_prompt.txt', 'w') as d:
        print('Saving the prompt......')
        d.write('prompt:' + prompt)
        d.write('prompt_w_d' + prompt_w_d)
        print('prompt:', prompt)
        print('prompt_w_d', prompt_w_d)

def load_prompt(file_name):
    prompt = ''
    prompt_w_d = ''
    with open(file_name+'_prompt.txt', 'r') as d:
        lines = d.readlines()
        for line in lines:
            if line.startswith('prompt:'):
                prompt = line.replace('prompt:', '').strip()
            elif line.startswith('prompt_w_d'):
                prompt_w_d = line.replace('prompt_w_d', '').strip()
        return prompt, prompt_w_d

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('bag', type=int, default=0, help='bag')
    parser.add_argument('i', type=int, default=0, help='start index')
    args = parser.parse_args()
    start_idx = args.i
    
    if not run_single_example: 
        if new_sample:
            index_r,src_r,tgt_r,adv_r,tgt_dict = read_file(dir,randomly_pick,bag=args.bag,dict_size=dict_size)
            save_picked_idx(file_name,index_r)
            print(index_r)
        else:
            pick = read_picked_idx(file_name=file_name)
            index_r,src_r,tgt_r,adv_r,tgt_dict = read_file(dir,randomly_pick=False,dict_size=dict_size,picked=pick)
            print(pick)
            print(len(pick))
        print(len(src_r))

        with open (file_name+'.csv', 'a') as f:
            print('Writing in file',file_name, '........')
            writer = csv.writer(f)
            
            # run GPT
            chatbot = ChatGPT()
            for i in range(len(src_r)):
                if i < start_idx:
                    continue
                elif i == 0:
                    # f.write('tgt,result_src,reault_adv,result_adv_defence,index,src,adv\n')
                    f.write('tgt,src_response, adv_response, src_wd_response, adv_wd_response, src_idk_response, adv_idk_response, src_wd_idk_response, adv_wd_idk_response, index,src,adv\n')
                    save_dict(file_name,tgt_dict)
                    save_prompt(file_name,prompt,prompt_w_d)
                print('-------- index ',i,' ---------')
                # src_response, adv_response, adv_wd_response, tgt = run_GPT_dif_cases(chatbot,src_r[i],adv_r[i],tgt_r[i],tgt_dict,prompt,prompt_w_d,temperature=0)
                src_response, adv_response, src_wd_response, adv_wd_response, src_idk_response, adv_idk_response, src_wd_idk_response, adv_wd_idk_response, tgt = run_GPT_dif_cases(chatbot,src_r[i],adv_r[i],tgt_r[i],tgt_dict,prompt,prompt_w_d,temperature=0)
                
                # f.write(str(tgt)+','+str(src_response)+','+str(adv_response)+','+str(adv_wd_response)+','+src_r[i]+','+adv_r[i]+'\n')
                # f.write(str(tgt)+','+str(src_wd_response)+','+str(adv_wd_response)+','+str(i)+','+src_r[i]+','+adv_r[i]+'\n')
                f.write(str(tgt)+','+str(src_response)+','+str(adv_response)+','+str(src_wd_response)+','+str(adv_wd_response)+','+str(src_idk_response)+','+str(adv_idk_response)+','+str(src_wd_idk_response)+','+str(adv_wd_idk_response)+','+str(i)+','+src_r[i]+','+adv_r[i]+'\n')
    else:
        # run single example
        tgt_dict = load_dict(file_name)
        prompt,prompt_w_d = load_prompt(file_name)
        chatbot = ChatGPT()
        src_r = "( self size = default size alpha = default alpha rescale threshold = default rescale threshold clock = get clock ( ) ) : creates a new exponentially decaying reservoir of 1028 elements which offers a 99.9% confidence level with a 5% margin of error assuming a normal distribution and an alpha factor of 0.015 which heavily biases the reservoir to the past 5 minutes of @param size the number of samples to keep in the sampling reservoir @param alpha the exponential decay factor; the higher this is the more biased the reservoir will be towards newer values @param rescale threshold the time period over which to decay super ( exponentially decaying reservoir self ) . init ( ) self . values = { } self . alpha = alpha self . size = size self . clock = clock self . rescale threshold = rescale threshold self . count = 0 self . start time = self . clock . time ( ) self . next scale time = self . clock . time ( ) + self . rescale threshold"
        adv_r = "( self size = default size alpha = default alpha rescale threshold = default rescale threshold clock = get clock ( ) ) :    creates a new exponentially decaying reservoir of 1028 elements which offers a 99.9% confidence level with a 5% margin of error assuming a normal distribution and an alpha factor of 0.015 which heavily biases the reservoir to the past 5 minutes of @param size the number of samples to keep in the sampling reservoir @param alpha the exponential decay factor; the higher this is the more biased the reservoir will be towards newer values @param rescale threshold the time period over which to decay   if false : hermit = 1 super ( exponentially decaying reservoir self ) . init ( )    self . values = { } if false : pypy = 1   self . alpha = alpha    self . size = size    self . clock = clock  print ( lte )  self . rescale threshold = rescale threshold   if false : reference = 1 self . count = 0    self . start time = self . clock . time ( )    self . next scale time = self . clock . time ( ) + self . rescale "
        tgt_r = "init"
        # src_response, adv_response, adv_wd_response, tgt = run_GPT_dif_cases(chatbot,src_r,adv_r,tgt_r,tgt_dict,prompt,prompt_w_d,temperature=0)
        src_response, adv_response, src_wd_response, adv_wd_response, src_idk_response, adv_idk_response, src_wd_idk_response, adv_wd_idk_response, tgt = run_GPT_dif_cases(chatbot,src_r[i],adv_r[i],tgt_r[i],tgt_dict,prompt,prompt_w_d,temperature=0)


