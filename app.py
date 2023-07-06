#Libraries
import os
from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI
st.set_page_config(layout="wide")

from langchain.llms import OpenAI
from langchain import FewShotPromptTemplate, PromptTemplate, SQLDatabaseChain, SQLDatabase
# from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.chains import LLMChain, LLMMathChain, TransformChain, SequentialChain
from langchain.callbacks import get_openai_callback
from langchain.agents import Tool, initialize_agent, AgentType
import inspect
import re

#Model based Librabries
import pandas as pd
import numpy as np
import re
import openai
import os
import time
# from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
# import tiktoken

#OpenAI setup
os.environ['OPENAI_API_KEY'] = apikey
# openai.api_key = openai_key

model_list = {
    'davinci':'text-davinci-003',
    'curie': 'text-curie-001',
    'babbage': 'text-babbage-001',
    'ada': 'text-ada-001'}

openai_llm = OpenAI(
    model_name = model_list['davinci'],
    openai_api_key = st.secrets["openai_key"],
    temperature = 0
)

#Functions
def extract_sql_query_and_result(string):
    # Extract SQL query
    query_pattern = r'SQLQuery:(.*?)SQLResult:'
    query_match = re.search(query_pattern, string, re.DOTALL)
    query = query_match.group(1).strip()

    # Extract SQL result
    result_pattern = r'SQLResult:(.*?)\n'
    result_match = re.search(result_pattern, string, re.DOTALL)
    result = eval(result_match.group(1).strip())

    return query, result

#Creating Database connection
db = SQLDatabase.from_uri("sqlite:///customer_data.db"
                          ,include_tables = ['Customer_State', 'Customer_Transactions']
                          ,sample_rows_in_table_info = 3
                          # ,custom_table_info = custom_table_info
                          )

#Setting up SQL Chain
llm_sql_chain = SQLDatabaseChain.from_llm(openai_llm, db, verbose=True, return_intermediate_steps = True, use_query_checker=True)

#Setting up Python Chain
prefix = """Given a SQL query and result, write a python code which uses SQL query and result to perform below tasks:
1. Use SQL query to get column names and result to enter the rows in a pandas dataframe and call it 'df'.
2. Create a chart to show this pandas dataframe effectively using 'plotly' library.
3. Keep the width of the chart 800 and height of the chart 600.

Below is an example and use the same format:

SQLQuery: SELECT "Product_Category", SUM("Sales_USD") AS "Total_Sales" FROM "Customer_Transactions" 
WHERE "Transaction_Date" BETWEEN date('2023-11-01') AND date('2023-11-30') 
AND "Product_Category" IS NOT NULL
GROUP BY "Product_Category" 
ORDER BY "Total_Sales" DESC 
LIMIT 5;

SQLResult: [('Medical', 9085), ('Sanitation', 8560), ('Fitness', 6271), ('Furniture', 6179), ('Computers', 5090)]

Answer: ```
#Libraries
import pandas as pd
import plotly.graph_objects as go

# Create dataframe
df = pd.DataFrame({'Product Category': ['Medical', 'Sanitation', 'Fitness', 'Furniture', 'Computers'],
                   'Sales': [9085, 8560, 6271, 6179, 5090]})

# Create bar chart using Plotly
fig = go.Figure(data=[go.Bar(x=df['Product Category'], y=df['Sales'])])

# Set chart title and axis labels
fig.update_layout(
  title='Top Product Categories for October 2023',
  xaxis_title='Product Category',
  yaxis_title='Sales',
  # Set the width of the figure  
  width=800,
  # Set the height of the figure
  height=600)
```
"""
suffix = """
SQLQuery: {query}
SQLResult: {result}
Answer: 
"""
suffix_prompt = PromptTemplate(
    input_variables=["query", "result"], template=suffix
)

##App framework
st.image('exl_logo.png', width = 150)
st.title('Coversational BI')

# slider_left, slider_right = st.columns([3,10])
# with slider_left:
#     option = st.select_slider('',
#         options=['Q/A', 'Visualization'])

option = st.selectbox(
    '',
    ('Q/A', 'Vizualization'))


prompt = st.text_area('Please enter your query', height = 5)


def model_call(prompt, option_selected):

    return_items = []
    #Running SQL Chain
    sql_chain_output = llm_sql_chain(prompt)
    sql_query = sql_chain_output['intermediate_steps'][-2]['input']
    query, result = extract_sql_query_and_result(sql_query)
    return_items.extend([query, sql_chain_output])

    if option_selected == 'Vizualization':
    #Running Python Chain
        code_llm_prompt = prefix + suffix_prompt.format(query = query, result = result)
        code_output = openai_llm(code_llm_prompt)
        code_output = code_output.strip('`')
        return_items.append(code_output)
    
    return return_items



if st.button('Run'):
    if option == 'Vizualization':
        progress_text = "Analyzing the query, getting back with a chart"
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(2)
            my_bar.progress(percent_complete + 1, text=progress_text)
            model_output =  model_call(prompt, option)
            prompt_query, prompt_nl_output, prompt_python_code = model_output[0], model_output[1], model_output[2]
            if type(prompt_query) == type('Check'):
                my_bar.progress(100)
                break
        
        exec(prompt_python_code)
        left_side, center, right_side = st.columns([2,10,2])
        with left_side:
            st.text("")
        with right_side:
            st.text("")
        with center:
            st.plotly_chart(fig)

        with st.expander("Table"):
            st.dataframe(df)
        
        with st.expander("SQL Query"):
            st.code(prompt_query)

        with st.expander("Python Code"):
            st.code(prompt_python_code)
            
    else:
        progress_text = "Analyzing the query, getting back with the answer"
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(2)
            my_bar.progress(percent_complete + 1, text=progress_text)
            
            model_output =  model_call(prompt, option)
            prompt_query, prompt_nl_output = model_output[0], model_output[1]
            if type(prompt_query) == type('Check'):
                my_bar.progress(100)
                break
        
        st.write(prompt_nl_output['result'])

        with st.expander("SQL Query"):
            st.code(prompt_query)
