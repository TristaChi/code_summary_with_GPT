import numpy as np
import matplotlib.pyplot as plt

# ['index' ' src_response' ' adv_response' ' src_w_fsd_response'
#  ' adv_w_fsd_response' ' src_w_pd_response' ' adv_w_pd_response'
#  ' src_w_d_response' ' adv_w_d_response' ' src_idk_response'
#  ' adv_idk_response' ' src_w_fsd_idk_response' ' adv_w_fsd_idk_response'
#  ' src_w_pd_idk_response' ' adv_w_pd_idk_response' ' src_w_d_idk_response'
#  ' adv_w_d_idk_response' ' tgt']
src = [None,'True','False',None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]
adv = [None,'False','True',None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]
src_adv = [None,'True','True',None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]
nsrc_nadv = [None,'False','False',None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]

def target(i,j,src=False):
    only_i = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]
    only_j = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None] 
    both_ij = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None] 
    none_ij  = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]
    if src:
        only_i[1] = 'True'
        only_j[1] = 'True'
        both_ij[1] = 'True'
        none_ij[1] = 'True'
    only_i[i] = 'True'
    only_i[j] = 'False'
    only_j[j] = 'True'
    only_j[i] = 'False'
    both_ij[i] = 'True'
    both_ij[j] = 'True'
    none_ij[i] = 'False'
    none_ij[j] = 'False'
    # print(i,j,only_i,both_ij)
    
    return only_i, only_j, both_ij, none_ij

def get_list(i0=None, i1=None, i2=None, i3=None, i4=None, i5=None, i6=None, i7=None,
            i8=None,i9=None, i10=None, i11=None, i12=None, i13=None, i14=None, i15=None, i16=None,i17=None):
    return [i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17]

def get_data(file='/Users/zhangchi/Desktop/attack_LLM/result/few_shot_prompt_defense_all_wd_idk/GPT_result.csv', ignore_first_line=False):
    data_str = np.genfromtxt(file, delimiter=',', dtype=str, encoding='utf-8')
    # print('---> ', data_str[0])
    # print('---> ', data_str[1])
    # print('---> ', data_str[2])
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
            # new_line.append(line[-1] in line[i]) 
            # new_line.append((line[-1] in line[i]) or (line[i] in line[-1]))
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
    '#66b3ff',  # Light Blue
    '#99ff99',  # Light Green
    '#ffcc99',  # Peach
    '#ff9999',  # Light Red
    '#ccf2ff',  # Very light blue, works well with light blue and peach tones
    '#b3ffb3',  # Very light green, complements light green and light blue tones
    '#ffe0e0',  # Soft pink, complements light red and peach tones
    '#d9f2d9',  # Soft mint, works well with light green tones
    '#ffcc66',  # A golden tone, complements peach and light blue
    '#99ccff',  # Sky blue, works nicely with blue and green tones
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

if __name__ == '__main__':
    # ['index' 
    # i1-i8: ' src_response' ' adv_response' ' src_w_fsd_response'
    #  ' adv_w_fsd_response' ' src_w_pd_response' ' adv_w_pd_response'
    #  ' src_w_d_response' ' adv_w_d_response'
    #  i9-i17: ' src_idk_response' ' adv_idk_response' ' src_w_fsd_idk_response' ' adv_w_fsd_idk_response'
    #  ' src_w_pd_idk_response' ' adv_w_pd_idk_response' ' src_w_d_idk_response'
    #  ' adv_w_d_idk_response' ' tgt']

    # csv_file_path = 'merged-result/gpt-3.5-turbo.csv'
    csv_file_path = 'merged-result/gpt4_merge_1000.csv'
    # csv_file_path = 'merged-result/claude-instant-1.csv'    
    # csv_file_path = 'merged-result/claude-2.csv'
    # csv_file_path='merged-result/codellama.csv'

    # csv_file_path = 'merged-result/GPT35-meta-d.csv'
    # csv_file_path = 'merged-result/unique_meta_gpt4.csv'
    bool_data, idk_data = get_data(file=csv_file_path,ignore_first_line=False)
    
    # bool_data = get_data('modified_file.csv')
    # target = get_list(i1=True,i3=True)
    # target = get_list(i1=True)
    # print(target)
    # print('correct:',count_match(target,bool_data))
    # print('idk: ',count_match(target,idk_data))
    # print('idk for src correct: ',count_idk(target, bool_data, idk_data))


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



    # title = 'Src and Src_w_pd'
    # labels = ['Both Src & Src_w_pd Correct', 'Src Correct', 'Src_w_pd Correct', 'None Correct']
    # name='Src&Src_w_pd'
    # only_i, only_j, both_ij, none_ij = target(1,5,True)

    # data = [count_match(both_ij, bool_data), count_match(only_i, bool_data), count_match(only_j, bool_data), count_match(none_ij, bool_data)]
    # plot(data, labels, title, name=name)
