import openai
import requests
import json


openai.api_key = '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'
# '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'
# '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'
# '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'

class ChatGPT():
    def __init__(self) -> None:
        pass

    def __call__(self, msg, dict,defence=True,example=False, **kwargs):
        # raw_response = openai.ChatCompletion.create(
        #             model="gpt-3.5-turbo",
        #             # model="gpt-4",
        #             messages=[
        #                 {"role": "user", "content": msg}
        #             ],
        #             **kwargs)
        
        # raw_response = openai.Completion.create(
        #             model="text-davinci-003",
        #             prompt=msg,
        #             **kwargs)
        # self.raw_response = raw_response
        sys_msg = "the possible output should be one of the following dictionary: "+str(dict)
        if not example: 
            
            # raw_response = openai.ChatCompletion.create(
            #             model="gpt-3.5-turbo",
            #             messages=[
            #                 {"role": "system", "content": "act as a code summarization model that only outputs one word"},
            #                 {"role": "system", "content": sys_msg},
            #                 {"role": "user", "content": msg}
            #             ],
            #             **kwargs)
            raw_response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "act as a code summarization model that only outputs one word"},
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": "( self ) : self . res1 = adfuller ( self . x regression = ct autolag = none maxlag = 4 ) self . teststat = - 1.8566374 self . pvalue = 0.67682968 self . critvalues = [ - 4.007 - 3.437 - 3.137 ]"},
                            {"role": "assistant", "content": "init"},
                            {"role": "user", "content": "( self ) : if false : bool = 1   self . res1 = adfuller ( self . x regression = ct autolag = none maxlag = 4 )   if false : disassemble = 1 self . gpu = - 1.8566374    self . pvalue = 0.67682968 if false : field2 = 1   self . critvalues = [ - 4.007 - 3.437 - 3.137 ] if false : disasm = 1"},
                            {"role": "assistant", "content": "init"},
                            {"role": "user", "content": "( self other ) : if false : partitioning = 1 print ( set1 )  return ( self . pid == other . pid and self . sid == other . sid and self . doing == other . score and self . p start == other . p start and self . p end == other . p end and self . mor == other . p end and self . s start == other . s start and self . sy == other . s end and self . s len == other . s len )"},
                            {"role": "assistant", "content": "eq"},
                            {"role": "user", "content": "( kernel ) : result = weapon ( ) result . template = ( object/weapon/melee/sword/shared sword lightsaber ben.iff ) result . attribute template id = 10 result . stf name ( weapon name sword lightsaber ben ) return result"},
                            {"role": "assistant", "content": "create"},
                            {"role": "user", "content": "( kernel ) :    result = weapon ( )  print ( lwrite )  result . template = ( object/weapon/melee/sword/shared sword lightsaber ben.iff )  print ( stf ) if false : exported = 1 result . attribute template id = 10    result . stf name ( weapon name sword lightsaber ben )  print ( virtualname )  return result   if false : orm = 1"},
                            {"role": "assistant", "content": "create"},
                            {"role": "user", "content": msg}
                        ],
                        **kwargs)
               

            return [str(m.message.content) for m in raw_response['choices']]
        else: 
            # raw_response = openai.ChatCompletion.create(
            #             model="gpt-3.5-turbo",
            #             messages=[
            #                 {"role": "system", "content": "act as a code summarization model that only outputs one word"},
            #                 {"role": "system", "content": sys_msg},
            #                 {"role": "user", "content": msg}
            #             ],
            #             **kwargs)
            raw_response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "act as a code model"},
                            # {"role": "system", "content": sys_msg},
                            {"role": "user", "content": msg}
                        ],
                        **kwargs)
            return [str(m.message.content) for m in raw_response['choices']]
        # return raw_response['choices'][0]['text']




if __name__ == '__main__':
    chatbot = ChatGPT()
    attack = False
    prompt = "Choose one word from the provided dictionary to summarize the given piece of code. The code are as follows:\n code:"
    prompt_w_d = "Choose one word from the provided dictionary to summarize the given piece of code. Remove the if false statement and the print statement in the code before the summarization. \n code:"
    file_name = 'result/role_system/GPT_result'

    # # correct label: create, idx:39
    # src = "( kernel ) : result = creature ( ) result . template = object/mobile/shared piket longhorn.iff result . attribute template id = 9 result . stf name ( monster name piket ) return result"
    # adv = "( kernel ) : if false : retrieve = 1   result = creature ( )    result . template = object/mobile/shared piket longhorn.iff  print ( lwrite ) if false : gout = 1 result . attribute template id = 9    result . stf name ( monster name piket )    return result  print ( stf ) if false : orm = 1 "

    # # correct label: test_healthcheck, idx:4
    # src = "( self ) : response = self . fetch ( /healthcheck ) expect ( response . body ) . to equal ( working )"
    # adv = "( self ) : if false : sets = 1 print ( invoke )  parsexml = self . fetch ( /healthcheck ) if false : contribute = 1   expect ( parsexml . body ) . to equal ( working ) if false : aaaa = 1 "
    # adv_cverted = "def code(self):\n    parsexml = self.fetch('/healthcheck')\n    expect(parsexml.body).to.equal(working)\n"
    
    # correct label: init, idx:40
    # src = "( self db file path verbose stream ) : self . verbose = logger ( database verbose stream ) self . connection = self . connect ( db file path ) self . configure ( )"
    # adv = "( self asserts whether ) :    self . verbose = logger ( database whether ) if false : field2 = 1   self . connection = self . connect ( asserts ) if false : save = 1 print ( has )  self . configure ( ) "
    # adv_cverted = "self.verbose = logger(database)\nself.connection = self.connect(asserts)\nself.configure()"

    # # correct label: init, idx:27
    # src = "( self address ) : self . version = 6 self . max prefixlen = ipv6length"
    # adv = "( self address ) : if false : asserts = 1 print ( whether ) if false : converts = 1 self . version = 6 if false : raises = 1   self . virtualname = ipv6length "


    with open(file_name+'.dict', 'r') as d:
        dict = json.load(d)

    print("=============== src =================")
    prompt = prompt.format(dict)+src
    response = chatbot(prompt,dict=dict,temperature=0,max_tokens=30,example=True)
    print(response)

    print("=============== adv =================")
    prompt = prompt.format(dict)+adv_cverted
    response = chatbot(prompt,dict=dict,temperature=0,max_tokens=30,example=False)
    print(response)

    print("=============== advd =================")
    prompt_w_d = prompt_w_d.format(dict)+adv_cverted
    response = chatbot(prompt_w_d,dict=dict,temperature=0,max_tokens=30,example=False)
    print(response)

    prompt = 'Remove the dead code and the print statement in the code, code:'+adv
    # prompt = 'Remove the if false statement and the print statement in the code, code:'+adv
    response = chatbot(prompt,dict=dict,temperature=0.1,example=True)
    print(response)
