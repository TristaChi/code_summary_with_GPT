from myGPT import ChatGPT
import csv
import numpy as np
import re
import json
import argparse 
import time
from myGPT_plain import ChatGPT_plain
from prompt import *

def get_data(index,src,tgt,adv,dict_size,random_pick=False,bag=1,picked=[]):
    if picked != []:
        print('using the selected samples...')
        pick_idx = picked
    elif random_pick:
        print("randomly pick data from src")
        pick_idx = np.random.choice(range(len(index)),size=dict_size) # type: ignore
    else:
        # pick data batch by batch
        pick_idx = [num + bag*dict_size for num in range(dict_size)]
        print("pick data from ", pick_idx[0],"to ", pick_idx[-1])

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
                use_GPT='gpt4',
                max_tokens=10):
    prompt = prompt.format(dict)+input
        
    if few_shot_defense: 
        response = chatbot(prompt,dict=dict,defense=few_shot_defense,use_GPT=use_GPT,temperature=temperature,top_p=top_p,max_tokens=max_tokens)
    else: 
        response = chatbot(prompt,dict=dict,defense=few_shot_defense,use_GPT=use_GPT,temperature=temperature,top_p=top_p,max_tokens=max_tokens)
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

def run_GPT_dif_cases(chatbot,src_r,adv_r,tgt_r,tgt_dict,temperature=0,use_GPT='gpt4'):
    # Please comment out the cases not need for experiment for efficiency

    src_response, adv_response, src_w_fsd_response, adv_w_fsd_response, src_w_pd_response, adv_w_pd_response, src_w_d_response, adv_w_d_response = '','','','','','','',''
    src_idk_response, adv_idk_response,src_w_fsd_idk_response, adv_w_fsd_idk_response, src_w_pd_idk_response, adv_w_pd_idk_response, src_w_d_idk_response, adv_w_d_idk_response = '','','','','','','',''

    print('-------- src ---------')
    # print('src code:',src_r)
    src_response = get_response(chatbot, src_r, tgt_dict, prompt,
                few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)[0]
    print('src response',src_response)

    print('-------- adv ---------')
    # print('adv code:',adv_r)
    adv_response = get_response(chatbot, adv_r, tgt_dict, prompt,
                few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)[0]
    print('adv response',adv_response)

    # only consider defense on inputs correctly been labeled
    if not tgt_r.replace("_", "") in src_response.replace("_", ""):
        return src_response, adv_response, src_w_fsd_response, adv_w_fsd_response, src_w_pd_response, adv_w_pd_response, src_w_d_response, adv_w_d_response, src_idk_response, adv_idk_response, src_w_fsd_idk_response, adv_w_fsd_idk_response, src_w_pd_idk_response, adv_w_pd_idk_response, src_w_d_idk_response, adv_w_d_idk_response, tgt_r.replace(' ', '_')

    print('-------- src w/ few-shot defense ---------')
    src_w_fsd_response = get_response(chatbot, src_r, tgt_dict, prompt,
                few_shot_defense=True, use_GPT=use_GPT, temperature=temperature)[0]
    print('src w/ few-shot defense',src_w_fsd_response)

    print('-------- adv w/ few-shot defense ---------')
    adv_w_fsd_response = get_response(chatbot, adv_r, tgt_dict, prompt,
                few_shot_defense=True, use_GPT=use_GPT, temperature=temperature)[0]
    print('adv w/ few-shot defense',adv_w_fsd_response)

    print('-------- src w/ prompt defense ---------')
    src_w_pd_response = get_response(chatbot, src_r, tgt_dict, prompt_w_d,
                few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)[0]
    print('src w/ prompt defense',src_w_pd_response)

    print('-------- adv w/ prompt defense ---------')
    adv_w_pd_response = get_response(chatbot, adv_r, tgt_dict, prompt_w_d,
                few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)[0]
    print('adv w/ prompt defense',adv_w_pd_response)

    print('-------- src w/ both defense ---------')
    src_w_d_response = get_response(chatbot, src_r, tgt_dict, prompt_w_d,
                few_shot_defense=True, use_GPT=use_GPT, temperature=temperature)[0]
    print('src w/ both defense',src_w_d_response)

    print('-------- adv w/ both defense ---------')
    adv_w_d_response = get_response(chatbot, adv_r, tgt_dict, prompt_w_d,
                few_shot_defense=True, use_GPT=use_GPT, temperature=temperature)[0]
    print('adv w/ both defense',adv_w_d_response)

    print('-------- src and idk ---------')
    src_idk_response = get_response(chatbot, src_r, tgt_dict, prompt_idk,
                few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)[0]
    print('src w/ idk response',src_idk_response)

    print('-------- adv and idk ---------')
    adv_idk_response = get_response(chatbot, adv_r, tgt_dict, prompt_idk,
                few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)[0]
    print('adv w/ idk response',adv_idk_response)

    print('-------- src w/ few-shot defense and idk ---------')
    src_w_fsd_idk_response = get_response(chatbot, src_r, tgt_dict, prompt_idk,
                few_shot_defense=True, use_GPT=use_GPT, temperature=temperature)[0]
    print('src w/ few-shot defense and idk',src_w_fsd_idk_response)

    print('-------- adv w/ few-shot defense and idk ---------')
    adv_w_fsd_idk_response = get_response(chatbot, adv_r, tgt_dict, prompt_idk,
                few_shot_defense=True, use_GPT=use_GPT, temperature=temperature)[0]
    print('adv w/ few-shot defense and idk',adv_w_fsd_idk_response)

    print('-------- src w/ prompt defense and idk ---------')
    src_w_pd_idk_response = get_response(chatbot, src_r, tgt_dict, prompt_idk_w_d,
                few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)[0]
    print('src w/ prompt defense and idk',src_w_pd_idk_response)

    print('-------- adv w/ prompt defense and idk ---------')
    adv_w_pd_idk_response = get_response(chatbot, adv_r, tgt_dict, prompt_idk_w_d,
                few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)[0]
    print('adv w/ prompt defense and idk',adv_w_pd_idk_response)

    print('-------- src w/ both defense and idk ---------')
    src_w_d_idk_response = get_response(chatbot, src_r, tgt_dict, prompt_idk_w_d,
                few_shot_defense=True, use_GPT=use_GPT, temperature=temperature)[0]
    print('src w/ both defense and idk',src_w_d_idk_response)

    print('-------- adv w/ both defense and idk ---------')
    adv_w_d_idk_response = get_response(chatbot, adv_r, tgt_dict, prompt_idk_w_d,
                few_shot_defense=True, use_GPT=use_GPT, temperature=temperature)[0]
    print('adv w/ both defense and idk',adv_w_d_idk_response)

    print('-------- tgt ---------')
    print('tgt code:',tgt_r.replace(' ', '_'))


    return src_response, adv_response, src_w_fsd_response, adv_w_fsd_response, src_w_pd_response, adv_w_pd_response, src_w_d_response, adv_w_d_response, src_idk_response, adv_idk_response, src_w_fsd_idk_response, adv_w_fsd_idk_response, src_w_pd_idk_response, adv_w_pd_idk_response, src_w_d_idk_response, adv_w_d_idk_response, tgt_r.replace(' ', '_')

def run_GPT_meta_d(chatbot,src_r,adv_r,tgt_r,tgt_dict,temperature=0,use_GPT='gp4'):
    # Please comment out the cases not need for experiment for efficiency

    src_response, adv_response, src_w_self_d_1_response, adv_w_self_d_1_response, src_w_self_d_2_response, adv_w_self_d_2_response = '','','','','',''
    print('-------- tgt ---------')
    tgt_r = tgt_r.replace(' ', '_')
    print('tgt code:',tgt_r)

    print('-------- src ---------')
    src_response = get_response(chatbot, src_r, tgt_dict, prompt,
                few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)[0]
    print('src response',src_response)

    print('-------- adv ---------')
    adv_response = get_response(chatbot, adv_r, tgt_dict, prompt,
                few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)[0]
    print('adv response',adv_response)

    # only consider defense on inputs correctly been labeled
    if not tgt_r.replace("_", "") in src_response.replace("_", ""):
        return src_response, adv_response, src_w_self_d_1_response, adv_w_self_d_1_response, src_w_self_d_2_response, adv_w_self_d_2_response, tgt_r


    print('-------- src w/ self_d_1 ---------')
    src_w_self_d_1_response = get_response(chatbot, src_r, tgt_dict, prompt_EBMP,
                few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)[0]
    print('src w/ self_d_1',src_w_self_d_1_response)

    print('-------- adv w/ self_d_1 ---------')
    adv_w_self_d_1_response = get_response(chatbot, src_r, tgt_dict, prompt_EBMP,
                few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)[0]
    print('adc w/ self_d_1',adv_w_self_d_1_response)

    print('-------- src w/ self_d_2 ---------')
    src_w_self_d_2_response = get_response(chatbot, src_r, tgt_dict, prompt_PAMP,
                few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)[0]
    print('src w/ self_d_2',src_w_self_d_2_response)

    print('-------- adv w/ self_d_2 ---------')
    adv_w_self_d_2_response = get_response(chatbot, src_r, tgt_dict, prompt_PAMP,
                few_shot_defense=False, use_GPT=use_GPT, temperature=temperature)[0]
    print('adc w/ self_d_2',adv_w_self_d_2_response)

    return src_response, adv_response, src_w_self_d_1_response, adv_w_self_d_1_response, src_w_self_d_2_response, adv_w_self_d_2_response, tgt_r

def save_picked_idx(file_name,picked):
    np.save(file_name+'_idx', picked)
    print('Saving the sampled index......')

def read_picked_idx(file_name):
    picked = np.load(file_name+'_idx.npy') 
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

def main(bag=0, 
        start_idx=0,
        data_dir = 'data/v2-92-z_o_5-pgd_3_smooth-asr45/tokens/sri/py150/gradient-targeting/test.tsv',
        randomly_pick = False,
        bag_size = 500,
        file_name = 'gpt-4_result/gpt4',
        new_sample = True,
        use_GPT = 'gpt4', 
        meta_d = False
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
        print('Writing in file',file_name,'.csv', '........')
        writer = csv.writer(f)
        
        # run GPT
        chatbot = ChatGPT()
        for i in range(216):
            if i < start_idx:
                continue
            elif i == 0 and bag == 0:
                save_dict(file_name,tgt_dict)
            print('-------- index ',index_r[i],' ---------')
            if meta_d:
                response = run_GPT_meta_d(chatbot,src_r[i],adv_r[i],tgt_r[i],tgt_dict,temperature=0,use_GPT=use_GPT)
            else: 
                response = run_GPT_dif_cases(chatbot,src_r[i],adv_r[i],tgt_r[i],tgt_dict,temperature=0,use_GPT=use_GPT)

            response = (str(index_r[i]),) + response
            formatted_response = ', '.join(map(str, response)) + '\n'

            f.write(formatted_response)
            f.flush()
            print(formatted_response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('--bag', type=int, default=0, help='bag')
    parser.add_argument('--i', type=int, default=0, help='start index')
    parser.add_argument('--plain', type=bool, default=False, help='add strategy to emprove accuracy such as dictionary, few-shot prompt') 
    parser.add_argument('--data_dir', type=str, default='data/data.tsv', help='data directory')
    parser.add_argument('--randomly_pick', type=bool, default=False, help='randomly pick data from src')
    parser.add_argument('--bag_size', type=int, default=500, help='bag size')
    parser.add_argument('--file_name', type=str, default='gpt4', help='result file name')
    parser.add_argument('--new_sample', type=bool, default=True, help='new sample or read from the result directory')
    parser.add_argument('--use_GPT', type=int, default=35, help='use GPT 4 or 3.5')

    args = parser.parse_args()

    main(bag=args.bag, 
        start_idx=args.i,
        data_dir = args.data_dir,
        randomly_pick = args.randomly_pick,
        bag_size = args.bag_size,
        file_name = args.file_name,
        new_sample = args.new_sample,
        use_GPT = args.use_GPT, 
        )
    

