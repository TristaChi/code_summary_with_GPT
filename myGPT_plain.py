import openai
import requests
import json


openai.api_key = 'your openai key'

class ChatGPT_plain():

    def __call__(self, input_msg, model="gpt-4-0613",use_GPT=35, **kwargs):
        if use_GPT == 35:
            model = "gpt-3.5-turbo"

        msg = [{"role": "user", "content": input_msg}]
        raw_response = openai.ChatCompletion.create(
                model=model,
                messages=msg,
                **kwargs)
        return [str(m.message.content) for m in raw_response['choices']]
            
        # return raw_response['choices'][0]['text']




if __name__ == '__main__':
    chatbot = ChatGPT()
    attack = False
    prompt = "Choose one word from the provided dictionary to summarize the given piece of code. The code are as follows:\n code:"
    prompt_w_d = "Choose one word from the provided dictionary to summarize the given piece of code. Remove the if false statement and the print statement in the code before the summarization. \n code:"
    file_name = 'result/role_system/GPT_result'
