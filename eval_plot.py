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

def get_data():
    data_str = np.genfromtxt('/Users/zhangchi/Desktop/attack_LLM/result/few_shot_prompt_defense_all_wd_idk/GPT_result.csv', delimiter=',', dtype=str)
    print(np.shape(data_str))

    bool_data = []
    for line in data_str:
        new_line = line.copy()
        for i in range(len(line)):
            new_line[i] = (line[-1] in line[i])
        bool_data.append(new_line)
    return bool_data

def count_match(target, bool_data):
    count = 0
    for line in bool_data:
        match = all((t is None or t == l) for t, l in zip(target, line))
        if match:
            count += 1
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
    # ['index' ' src_response' ' adv_response' ' src_w_fsd_response'
    #  ' adv_w_fsd_response' ' src_w_pd_response' ' adv_w_pd_response'
    #  ' src_w_d_response' ' adv_w_d_response' ' src_idk_response'
    #  ' adv_idk_response' ' src_w_fsd_idk_response' ' adv_w_fsd_idk_response'
    #  ' src_w_pd_idk_response' ' adv_w_pd_idk_response' ' src_w_d_idk_response'
    #  ' adv_w_d_idk_response' ' tgt']
    bool_data = get_data()

    title = 'Src and Src_w_pd'
    labels = ['Both Src & Src_w_pd Correct', 'Src Correct', 'Src_w_pd Correct', 'None Correct']
    name='Src&Src_w_pd'
    only_i, only_j, both_ij, none_ij = target(1,5,True)

    data = [count_match(both_ij, bool_data), count_match(only_i, bool_data), count_match(only_j, bool_data), count_match(none_ij, bool_data)]
    plot(data, labels, title, name=name)