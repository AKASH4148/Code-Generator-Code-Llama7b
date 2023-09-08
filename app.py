from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain import PromptTemplate
import os
import io
import gradio as gr
import time

custom_prompt_template=""" 
You're an AI coding assstant and your task is to solve coding problems and return code snippet based on given user query.
below is the user query
Query : {query} 

You just return the correct code and it should be robust and detailed.

"""

def set_custom_prompt():
    prompt=PromptTemplate(
        template=custom_prompt_template,
        input_variables=["query"]
    )
    return prompt

def load_model():
    llm=CTransformers(
        model="codellama-7b-instruct.ggmlv3.Q4_0.bin",
        model_type='llama',
        max_new_tokens=1096,
        temperature=0.2,
        repetition_penality=1.13
    )
    return llm

def chain_pipeline():
    llm=load_model()
    qa_prompt=set_custom_prompt()
    qa_chain=LLMChain(
        prompt=qa_prompt,
        llm=llm
    )
    return qa_chain
llm_chain=chain_pipeline()

def bot(query):
    llm_response=llm_chain.run({"query":query})
    return llm_response

with gr.Blocks(title="Code Llama Demo") as demo:
    gr.Markdown("# Code Llama Demo")
    chatbot=gr.Chatbot([], elem_id="chatbot", height=700)
    msg=gr.Textbox()
    clear=gr.ClearButton([msg, chatbot])

    def response(message, chat_history):
        bot_message=bot(message)
        chat_history.append((message, bot_message))
        time.sleep(2)
        return "", chat_history
    msg.submit(response, [msg, chatbot], [msg, chatbot])
demo.launch()