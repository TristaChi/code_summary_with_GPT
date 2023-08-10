import csv
# from email.utils import collapse_rfc2231_value
import re
import argparse 

csv_file_path = '/Users/zhangchi/Desktop/code_robustness_of_LLM/GPT/result/few_shot_prompt_defense/GPT_result.csv'


def get_result(input_string):
    english_word = re.search(r'[a-zA-Z]+', input_string).group()
    return english_word

def evaluate(csv_file_path,print_code,print_detail=False):
    # Create an empty array to store the extracted data
    data = []
    # Open the CSV file
    with open(csv_file_path, 'r') as file:
        # Create a CSV reader object
        reader = csv.reader(file)

        # Skip the first row
        next(reader)

        # Read and process the remaining rows
        for row in reader:
            # Extract the first four columns (indexes 0 to 3)
            # columns = row[:4]
            columns = row

            # Add the extracted columns to the data array
            data.append(columns)
    src_correct = 0
    adv_correct = 0
    advd_correct = 0
    total = 0
    for row in data:

        tgt = row[0]
        src = row[1]
        adv = row[2]
        advd = row[3]
        # print("tgt,src,adv,advd: ", tgt,src,adv,advd)
        total += 1
        if tgt in src:
            src_correct+=1
            if tgt in adv: adv_correct+=1
            if tgt in advd: advd_correct+=1
        
        if tgt in src and tgt not in adv:
            if print_code:
                print("row:", row)
            else:
                # print('--------------- adv success -----------------')
                print( total+1,tgt,src,adv,advd)
        
        
        # if tgt in src and tgt in adv and tgt not in advd:
        #     if print_code:
        #         print("row:", row)
        #     else:
        #         # print('--------------- advd fail -----------------')
        #         print("tgt,src,adv,advd: ", total+1,tgt,src,adv,advd)
        
        
        
        # if src == tgt and adv == tgt and advd != tgt:
        #     if print_code:
        #         print("---- defence fail ----\nrow:", row)
        #     else:
        #         print("---- defence fail ----\nidx,tgt,src,adv,advd: ", total+1,tgt,src,adv,advd)
        # print("total,src_correct,adv_correct,advd_correct: ",total,src_correct,adv_correct,advd_correct)
            
    print("total,src_correct,adv_correct,advd_correct: ",total,src_correct,adv_correct,advd_correct)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('i', type=int, default=1)
    parser.add_argument('code', type=lambda x: (str(x).lower() == 'true'), default=False)
    args = parser.parse_args()
    print("evaluating file :",args.i)
    print("print_code: ",args.code)

    print("idx,tgt,src,adv,advd: ")
    # Specify the path to your CSV file 
    # csv_file_path = f"/Users/zhangchi/Desktop/code_robustness_of_LLM/GPT/result/top1_GPT_result_prompt_defence_seq_{args.i}.csv"
    evaluate(csv_file_path,args.code)
   
