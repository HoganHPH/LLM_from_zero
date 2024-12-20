<h1>Classify Text into Labels - Function calling</h1>

<ul>
    Guide:
    <li><a href='https://python.langchain.com/docs/tutorials/classification/'>LangChain Classification Function Calling</a></li>
    <li><a href='https://python.langchain.com/docs/integrations/providers/huggingface/'>LangChain HuggingFace</a></li>
</ul>


<p>
    <b>
    Objective: Use a LLM model of HuggingFace for Function Calling in LangChain
    </b>
</p>

<h2>Steps:</h2>

<h3>Step 1: Define LangChain chat model from a LLM with checkpoint of Hugging Face</h3>
<ul>
    <li>Using HuggingFaceEndpoint and ChatHuggingFace to create a chat model using checkpoint from HuggingFace</li>
</ul>
<h3>Step 2: Define chat promt (looks like an instruction for LLM)</h3>
<ul>
    <li>Following the exact prompt of HunggingFace model that can be found in the page of the model</li>
    <li>Apply <b>ChatPromptTemplate.from_template()</b> function to create a prompt in LangChain</li>
</ul>
<h3>Step 3: Define model (properties LLM need to return)</h3>
<ul>
    <li>Define model using Pydantic as properties and restricted values that LLM need to return</li>
</ul>
<h3>Step 4: Assign/Bind task for LLM</h3>
<ul>
    <li>Binding the task for LLM using <b>chat_model.bind_tools(task)</b> function</li>
</ul>
<h3>Step 5: Function calling and Show result</h3>
<ul>
    <li>Using <b>invoke()</b> function to feed prompt into LLM</li>
    <li>Need to cast response into a dict ("response.dict()") and then access the answer as an attribute of the dict ("dict['tool_calls']")</li>
</ul>

