import openai
import requests
import json


openai.api_key = '[your openai key]'

class ChatGPT():
    def __init__(self) -> None:
        self.few_shot_src_examples = [
            {"role": "user", "content": "( self ) : self . res1 = adfuller ( self . x regression = ct autolag = none maxlag = 4 ) self . teststat = - 1.8566374 self . pvalue = 0.67682968 self . critvalues = [ - 4.007 - 3.437 - 3.137 ]"},
            {"role": "assistant", "content": "init"},
            {"role": "user", "content": "( kernel ) : result = weapon ( ) result . template = ( object/weapon/melee/sword/shared sword lightsaber ben.iff ) result . attribute template id = 10 result . stf name ( weapon name sword lightsaber ben ) return result"},
            {"role": "assistant", "content": "create"},
            {"role": "user", "content": "( self other ) : return elemwise ( operator . add self other )"},
            {"role": "assistant", "content": "add"},
            # {"role": "user", "content": "( self * args ** kwargs ) : if self . thumbnail : from django . core . files . images import get image dimensions import django if django . version [ 1 ] < 2 : width height = get image dimensions ( self . thumbnail . file ) else : width height = get image dimensions ( self . thumbnail . file close = true ) else : width height = none none self . thumbnail width = width self . thumbnail height = height super ( category self ) . save ( * args ** kwargs )"},
            # {"role": "assistant", "content": "save"},
        ]
        self.few_shot_adv_examples = [
            {"role": "user", "content": "( self ) : if false : bool = 1   self . res1 = adfuller ( self . x regression = ct autolag = none maxlag = 4 )   if false : disassemble = 1 self . gpu = - 1.8566374    self . pvalue = 0.67682968 if false : field2 = 1   self . critvalues = [ - 4.007 - 3.437 - 3.137 ] if false : disasm = 1"},
            {"role": "assistant", "content": "init"},
            {"role": "user", "content": "( self * args ** coeffs ) :  print ( homogeneity ) if false : lwrite = 1 delete the selected metadata type.   if false : init = 1 return super ( api metadata type view self ) . delete ( * args ** coeffs )   if false : nfev = 1"},
            {"role": "assistant", "content": "delete"},
            # {"role": "user", "content": "( self other ) : if false : partitioning = 1 print ( set1 )  return ( self . pid == other . pid and self . sid == other . sid and self . doing == other . score and self . p start == other . p start and self . p end == other . p end and self . mor == other . p end and self . s start == other . s start and self . sy == other . s end and self . s len == other . s len )"},
            # {"role": "assistant", "content": "eq"},
            # {"role": "user", "content": "( kernel ) :    result = weapon ( )  print ( lwrite )  result . template = ( object/weapon/melee/sword/shared sword lightsaber ben.iff )  print ( stf ) if false : exported = 1 result . attribute template id = 10    result . stf name ( weapon name sword lightsaber ben )  print ( virtualname )  return result   if false : orm = 1"},
            # {"role": "assistant", "content": "create"},
        ]


    def __call__(self, input_msg, dict, defense=True,use_GPT=35, single_example=False,model = "gpt-4-0613", **kwargs):
        if use_GPT == 35:
            model = "gpt-3.5-turbo"
        sys_msg = "the possible output should be one of the following dictionary: "+str(dict)
        msg = [
            {"role": "system", "content": "act as a code summarization model that only outputs one word"},
            {"role": "system", "content": sys_msg}
        ]
        if not single_example: 
            if not defense:
                msg += self.few_shot_src_examples
                msg += [{"role": "user", "content": input_msg}]
                raw_response = openai.ChatCompletion.create(
                        model=model,
                        messages=msg,
                        **kwargs)
                return [str(m.message.content) for m in raw_response['choices']]
            else:
                msg += self.few_shot_src_examples
                msg += self.few_shot_adv_examples
                msg += [{"role": "user", "content": input_msg}]
                if use_GPT:
                    raw_response = openai.ChatCompletion.create(
                            model=model,
                            messages=msg,
                            **kwargs)
                    return [str(m.message.content) for m in raw_response['choices']]
                else:
                    inputs = [msg]
                    cls_params = {'model': model, **kwargs}

                    outputs = cls.generate(inputs, cls_params, use_tqdm=False)
                    responses = [o.outputs[0].text for o in outputs]

                    return responses

        else: 
            raw_response = openai.ChatCompletion.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "act as a code model"},
                            # {"role": "system", "content": sys_msg},
                            {"role": "user", "content": msg}
                        ],
                        **kwargs)
            return [str(m.message.content) for m in raw_response['choices']]


if __name__ == '__main__':
    chatbot = ChatGPT()
    attack = False
    prompt = "Choose one word from the provided dictionary to summarize the given piece of code. The code are as follows:\n code:"
    prompt_w_d = "Choose one word from the provided dictionary to summarize the given piece of code. Remove the if false statement and the print statement in the code before the summarization. \n code:"
    file_name = 'result/role_system/GPT_result'

    # correct label: create, idx:39
    src = "( kernel ) : result = creature ( ) result . template = object/mobile/shared piket longhorn.iff result . attribute template id = 9 result . stf name ( monster name piket ) return result"
    adv = "( kernel ) : if false : retrieve = 1   result = creature ( )    result . template = object/mobile/shared piket longhorn.iff  print ( lwrite ) if false : gout = 1 result . attribute template id = 9    result . stf name ( monster name piket )    return result  print ( stf ) if false : orm = 1 "

    # correct label: test_healthcheck, idx:4
    src = "( self ) : response = self . fetch ( /healthcheck ) expect ( response . body ) . to equal ( working )"
    adv = "( self ) : if false : sets = 1 print ( invoke )  parsexml = self . fetch ( /healthcheck ) if false : contribute = 1   expect ( parsexml . body ) . to equal ( working ) if false : aaaa = 1 "
    adv_cverted = "def code(self):\n    parsexml = self.fetch('/healthcheck')\n    expect(parsexml.body).to.equal(working)\n"
    
    # correct label: init, idx:40
    src = "( self db file path verbose stream ) : self . verbose = logger ( database verbose stream ) self . connection = self . connect ( db file path ) self . configure ( )"
    adv = "( self asserts whether ) :    self . verbose = logger ( database whether ) if false : field2 = 1   self . connection = self . connect ( asserts ) if false : save = 1 print ( has )  self . configure ( ) "
    adv_cverted = "self.verbose = logger(database)\nself.connection = self.connect(asserts)\nself.configure()"

    # correct label: init, idx:27
    src = "( self address ) : self . version = 6 self . max prefixlen = ipv6length"
    adv = "( self address ) : if false : asserts = 1 print ( whether ) if false : converts = 1 self . version = 6 if false : raises = 1   self . virtualname = ipv6length "


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
