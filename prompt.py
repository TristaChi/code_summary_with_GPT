prompt = "Choose one word from the provided dictionary to summarize the given piece of code. \n code:"
prompt_w_d = "Choose one word from the provided dictionary to summarize the given piece of code. Remove the if false statement and the print statement in the code before the summarization. \n code:"
prompt_idk = "Choose one word from the provided dictionary to summarize the given piece of code. Reply 'I don't know' if not sure. \n code:"
prompt_idk_w_d = "Choose one word from the provided dictionary to summarize the given piece of code. Remove the if false statement and the print statement in the code before the summarization. Reply 'I don't know' if not sure. \n code:"

prompt_self_d1 = "Choose one word from the provided dictionary to summarize the given piece of code. Before the summarization, given a perturbed version of a code snippet, your task is to convert it back to its original, clean, and functional form by removing any extraneous and unnecessary lines or elements. Make sure the output is syntactically correct and maintains the original logic and structure of the code. "
prompt_self_d2 = "Choose one word from the provided dictionary to summarize the given piece of code. Before the summarization, restore the perturbed code to its original form. Remove added print statements, eliminate dead code, correct replaced literals, and restore renamed variables, parameters, and fields to their original names. Ensure the output is syntactically correct and retains the original logic."


prompt_init = "Choose one word from the provided dictionary to summarize the given piece of code. The dictionary and code are as follows:\n dictionary: {} \n code:"
prompt_w_d_init = "Choose one word from the provided dictionary to summarize the given piece of code. Remove the dead code and the print statement in the code before the summarization. The dictionary and code are as follows:\n dictionary: {} \n code:"
