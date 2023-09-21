from myGPT import ChatGPT
import csv
import numpy as np
import re
import json
import argparse 


run_single_example = False
prompt = "Choose one word from the provided dictionary to summarize the given piece of code. \n code:"
prompt_w_d = "Choose one word from the provided dictionary to summarize the given piece of code. Remove the if false statement and the print statement in the code before the summarization. \n code:"
prompt_idk = "Choose one word from the provided dictionary to summarize the given piece of code. Reply 'I don't know' if not sure. \n code:"
prompt_idk_w_d = "Choose one word from the provided dictionary to summarize the given piece of code. Remove the if false statement and the print statement in the code before the summarization. Reply 'I don't know' if not sure. \n code:"

# prompt = "Choose one word from the provided dictionary to summarize the given piece of code. The dictionary and code are as follows:\n dictionary: {} \n code:"
# prompt_w_d = "Choose one word from the provided dictionary to summarize the given piece of code. Remove the dead code and the print statement in the code before the summarization. The dictionary and code are as follows:\n dictionary: {} \n code:"

# prompt = "Use one word from the set {} to describe the following piece of code. Also, output your confidence in your choice as a probability between 0 and 1. Don’t provide any other description. The reply form should be \"word (confidence)\". \n"
# prompt_w_d = "Use one word from the set {} to describe the following piece of code.Please remove the dead code and the print statement in the code before the description. Also, output your confidence in your choice as a probability between 0 and 1. Don’t provide any other description. The reply form should be \"word (confidence)\" \n"

def get_data(index,src,tgt,adv,dict_size,random_pick=False,bag=1,picked=[]):
    if picked != []:
        print('using the selected samples...')
        pick_idx = picked
    elif random_pick:
        print("randomly pick data from src")
        # randomly pick data
        pick_idx = np.random.choice(range(len(index)),size=dict_size) # type: ignore
    else:
        pick_idx = [num + bag*dict_size for num in range(dict_size)]
        # range(dict_size)+bag*dict_size
        print("pick data from ", pick_idx[0],"to ", pick_idx[-1])

    # index_r = [index[i] for i in rand_pick]
    src_r = [src[i] for i in pick_idx]
    tgt_r = [tgt[i] for i in pick_idx]
    adv_r = [adv[i] for i in pick_idx]
    print('pick_idx',pick_idx)
    return pick_idx,src_r,tgt_r,adv_r

def get_response(chatbot, 
                input, 
                dict, 
                prompt,
                temperature=0,
                top_p=1.0,
                few_shot_defense=False, 
                use_GPT=True,
                save_p=False, 
                file_name='result/'):
    prompt = prompt.format(dict)+input
        
    if few_shot_defense: 
        response = chatbot(prompt,dict=dict,defense=few_shot_defense,use_GPT=use_GPT,temperature=temperature,top_p=top_p,max_tokens=10)
    else: 
        response = chatbot(prompt,dict=dict,defense=few_shot_defense,use_GPT=use_GPT,temperature=temperature,top_p=top_p,max_tokens=10)
    return response

def read_file(data_dir,randomly_pick,dict_size=100,bag=0,picked=[]):
    # read from file and store in list
    with open(data_dir, 'r') as tsv_file:
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
    tgt_dict = {word.replace(' ', '_') for word in set(tgt_n)}
    print('tgt_dict',tgt_dict)
    print('dict_size',len(tgt_dict))
    return index_n,src_n,tgt_n,adv_n,tgt_dict

def run_GPT_dif_cases(chatbot,src_r,adv_r,tgt_r,tgt_dict,prompt,prompt_w_d,temperature=0,use_GPT=True):
    print('-------- src ---------')
    # print('src code:',src_r)
    src_response = get_response(chatbot, src_r, tgt_dict, prompt,
                few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)
    print('src response',src_response)

    print('-------- adv ---------')
    # print('adv code:',adv_r)
    adv_response = get_response(chatbot, adv_r, tgt_dict, prompt,
                few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)
    print('adv response',adv_response)

    # print('-------- src w/ few-shot defense ---------')
    # src_w_fsd_response = get_response(chatbot, src_r, tgt_dict, prompt,
    #             few_shot_defense=True, use_GPT=use_GPT, temperature=temperature)
    # print('src w/ few-shot defense',src_w_fsd_response)

    print('-------- adv w/ few-shot defense ---------')
    adv_w_fsd_response = get_response(chatbot, adv_r, tgt_dict, prompt,
                few_shot_defense=True, use_GPT=use_GPT, temperature=temperature)
    print('adv w/ few-shot defense',adv_w_fsd_response)

    # print('-------- src w/ prompt defense ---------')
    # src_w_pd_response = get_response(chatbot, src_r, tgt_dict, prompt_w_d,
    #             few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)
    # print('src w/ prompt defense',src_w_pd_response)

    print('-------- adv w/ prompt defense ---------')
    adv_w_pd_response = get_response(chatbot, adv_r, tgt_dict, prompt_w_d,
                few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)
    print('adv w/ prompt defense',adv_w_pd_response)

    # print('-------- src w/ both defense ---------')
    # src_w_d_response = get_response(chatbot, src_r, tgt_dict, prompt_w_d,
    #             few_shot_defense=True, use_GPT=use_GPT, temperature=temperature)
    # print('src w/ both defense',src_w_d_response)

    print('-------- adv w/ both defense ---------')
    adv_w_d_response = get_response(chatbot, adv_r, tgt_dict, prompt_w_d,
                few_shot_defense=True, use_GPT=use_GPT, temperature=temperature)
    print('adv w/ both defense',adv_w_d_response)

    # print('-------- src and idk ---------')
    # src_idk_response = get_response(chatbot, src_r, tgt_dict, prompt_idk,
    #             few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)
    # print('src w/ idk response',src_idk_response)

    print('-------- adv and idk ---------')
    adv_idk_response = get_response(chatbot, adv_r, tgt_dict, prompt_idk,
                few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)
    print('adv w/ idk response',adv_idk_response)

    # print('-------- src w/ few-shot defense and idk ---------')
    # src_w_fsd_idk_response = get_response(chatbot, src_r, tgt_dict, prompt_idk,
    #             few_shot_defense=True, use_GPT=use_GPT, temperature=temperature)
    # print('src w/ few-shot defense and idk',src_w_fsd_idk_response)

    # print('-------- adv w/ few-shot defense and idk ---------')
    # adv_w_fsd_idk_response = get_response(chatbot, adv_r, tgt_dict, prompt_idk,
    #             few_shot_defense=True, use_GPT=use_GPT, temperature=temperature)
    # print('adv w/ few-shot defense and idk',adv_w_fsd_idk_response)

    # print('-------- src w/ prompt defense and idk ---------')
    # src_w_pd_idk_response = get_response(chatbot, src_r, tgt_dict, prompt_idk_w_d,
    #             few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)
    # print('src w/ prompt defense and idk',src_w_pd_idk_response)

    # print('-------- adv w/ prompt defense and idk ---------')
    # adv_w_pd_idk_response = get_response(chatbot, adv_r, tgt_dict, prompt_idk_w_d,
    #             few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)
    # print('adv w/ prompt defense and idk',adv_w_pd_idk_response)

    # print('-------- src w/ both defense and idk ---------')
    # src_w_d_idk_response = get_response(chatbot, src_r, tgt_dict, prompt_idk_w_d,
    #             few_shot_defense=True, use_GPT=use_GPT, temperature=temperature)
    # print('src w/ both defense and idk',src_w_d_idk_response)

    print('-------- adv w/ both defense and idk ---------')
    adv_w_d_idk_response = get_response(chatbot, adv_r, tgt_dict, prompt_idk_w_d,
                few_shot_defense=True, use_GPT=use_GPT, temperature=temperature)
    print('adv w/ both defense and idk',adv_w_d_idk_response)

    print('-------- tgt ---------')
    print('tgt code:',tgt_r.replace(' ', '_'))

    # src_idk_response, adv_idk_response,src_w_fsd_idk_response, adv_w_fsd_idk_response, src_w_pd_idk_response, adv_w_pd_idk_response, src_w_d_idk_response, adv_w_d_idk_response = '','','','','','','',''
    # src_response, adv_response, src_w_fsd_response, adv_w_fsd_response, src_w_pd_response, adv_w_pd_response, src_w_d_response, adv_w_d_response = '','','','','','','',''
    src_w_fsd_response, src_w_pd_response, src_w_d_response, src_idk_response, src_w_fsd_idk_response, adv_w_fsd_idk_response, src_w_pd_idk_response, adv_w_pd_idk_response, src_w_d_idk_response = '','','','','','','','',''

    return src_response, adv_response, src_w_fsd_response, adv_w_fsd_response, src_w_pd_response, adv_w_pd_response, src_w_d_response, adv_w_d_response, src_idk_response, adv_idk_response, src_w_fsd_idk_response, adv_w_fsd_idk_response, src_w_pd_idk_response, adv_w_pd_idk_response, src_w_d_idk_response, adv_w_d_idk_response, tgt_r.replace(' ', '_')

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

def main(bag, start_idx,
        # prompt, 
        # prompt_w_d,
        # prompt_idk, 
        # prompt_idk_w_d,
        data_dir = '/Users/zhangchi/Desktop/attack_LLM/data/v2-92-z_o_5-pgd_3_smooth-asr45/tokens/sri/py150/gradient-targeting/test.tsv',
        randomly_pick = False,
        bag_size = 500,
        file_name = '/Users/zhangchi/Desktop/attack_LLM/result/GPT4/GPT_result',
        new_sample = True,
        use_GPT = True
        ):
    if new_sample:
        index_r,src_r,tgt_r,adv_r,tgt_dict = read_file(data_dir,randomly_pick,bag=bag,dict_size=bag_size)
        save_picked_idx(file_name,index_r)
        print(index_r)
    else:
        pick = read_picked_idx(file_name=file_name)
        index_r,src_r,tgt_r,adv_r,tgt_dict = read_file(data_dir,randomly_pick=False,dict_size=bag_size,picked=pick)
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
            elif i == 0 and bag == 0:
                # f.write('tgt,result_src,reault_adv,result_adv_defense,index,src,adv\n')
                # f.write('index, src_response, adv_response, src_w_fsd_response, adv_w_fsd_response, src_w_pd_response, adv_w_pd_response, src_w_d_response, adv_w_d_response, src_idk_response, adv_idk_response, src_w_fsd_idk_response, adv_w_fsd_idk_response, src_w_pd_idk_response, adv_w_pd_idk_response, src_w_d_idk_response, adv_w_d_idk_response, tgt \n')
                save_dict(file_name,tgt_dict)
                save_prompt(file_name,prompt,prompt_w_d)
            print('-------- index ',i,' ---------')
            # src_response, adv_response, adv_wd_response, tgt = run_GPT_dif_cases(chatbot,src_r[i],adv_r[i],tgt_r[i],tgt_dict,prompt,prompt_w_d,temperature=0)
            response = run_GPT_dif_cases(chatbot,src_r[i],adv_r[i],tgt_r[i],tgt_dict,prompt,prompt_w_d,temperature=0,use_GPT=use_GPT)

            response = (str(i),) + response
            formatted_response = ', '.join(map(str, response)) + '\n'

            f.write(formatted_response)
            f.flush()
            print(formatted_response)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('bag', type=int, default=0, help='bag')
    parser.add_argument('i', type=int, default=0, help='start index')
    args = parser.parse_args()
    start_idx = args.i
    
    

    main(args.bag, start_idx)
    

