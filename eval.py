import numpy as np
import matplotlib.pyplot as plt
import argparse 

def get_list(i0=None, i1=None, i2=None, i3=None, i4=None, i5=None, i6=None, i7=None,
            i8=None,i9=None, i10=None, i11=None, i12=None, i13=None, i14=None, i15=None, i16=None,i17=None):
    return [i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17]

def get_data(file='/Users/zhangchi/Desktop/attack_LLM/result/few_shot_prompt_defense_all_wd_idk/GPT_result.csv', ignore_first_line=False):
    data_str = np.genfromtxt(file, delimiter=',', dtype=str, encoding='utf-8')

    if ignore_first_line:
        data_str = data_str[1:]
    print(np.shape(data_str))

    bool_data = []
    idk_data = []
    for line in data_str:
        new_line = []
        new_line_idk = []
        # print(line)
        for i in range(len(line)-1):
            new_line.append((line[-1].replace("_", "") in line[i].replace("_", "")) and not ("I don't know" in line[i]))
            new_line_idk.append(("I don't know" in line[i]))
        bool_data.append(new_line)
        idk_data.append(new_line_idk)
    return bool_data, idk_data

def count_match(target, bool_data):
    count = 0
    index = 0
    for line in bool_data:
        match = all((t is None or t == l) for t, l in zip(target, line))
        if match:
            # print('index: %d' % index)
            count += 1
        index += 1
        # if index >= 200: break

    return count

def count_idk(target, bool_data, idk_data):
    count = 0
    src_correct = 0
    for i in range(len(idk_data)):
    # for line in idk_data:
        match = all((t is None or t == l) for t, l in zip(target, idk_data[i]))
        if bool_data[i][1]==True:
            src_correct += 1
        if match and bool_data[i][1]==True:
            count += 1
    # print("src correct: ",src_correct)
    return count

def plot(data, labels, title, name='src&adv'):
    # Define a custom color palette
    colors = [
    '#66b3ff',  
    '#99ff99', 
    '#ffcc99',  
    '#ff9999', 
    '#ccf2ff',  
    '#b3ffb3',  
    '#ffe0e0', 
    '#d9f2d9',  
    '#ffcc66',  
    '#99ccff',  
    ]
    colors = colors[:len(data)]

    # Create a pie chart
    plt.figure(figsize=(10, 9))
    plt.pie(data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140,textprops={'fontsize': 16})
    plt.title(title, fontsize=16)  # Increase title font size

    # Save the pie chart to a file
    plt.savefig('plot/'+name+'.png')

    # Display the pie chart
    plt.show()

def main(csv_file_path):
    bool_data, idk_data = get_data(file=csv_file_path,ignore_first_line=False)

    print('src	adv	src_w_fsd_response	adv_w_fsd_response	src_w_pd_response	adv_w_pd_response')
    print(count_match(get_list(i1=True),bool_data))
    print(count_match(get_list(i1=True,i2=True),bool_data))
    print(count_match(get_list(i1=True,i3=True),bool_data))
    print(count_match(get_list(i1=True,i4=True),bool_data))
    print(count_match(get_list(i1=True,i5=True),bool_data))
    print(count_match(get_list(i1=True,i6=True),bool_data))
    print('src_idk src_idk_correct	adv_idk  adv_idk_correct adv_w_fsd_idk  adv_w_fsd_idk_correct	adv_w_pd_idk  adv_w_pd_idk_correct')
    print(count_match(get_list(i9=True),idk_data))
    print(count_match(get_list(i9=True),bool_data))
    print(count_idk(get_list(i10=True), bool_data, idk_data))
    print(count_match(get_list(i1=True,i10=True),bool_data))
    print(count_idk(get_list(i12=True), bool_data, idk_data))
    print(count_match(get_list(i1=True,i12=True),bool_data))
    print(count_idk(get_list(i14=True), bool_data, idk_data))
    print(count_match(get_list(i1=True,i14=True),bool_data))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('--file', type=str, default='result/gpt4.csv', help='path to result file')
    args = parser.parse_args()

    csv_file_path = args.file
    main(csv_file_path)
