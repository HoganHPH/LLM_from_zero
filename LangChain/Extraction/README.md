<h1>Classify Text into Labels - Function calling</h1>

<ul>
    Guide:
    <li><a href='https://python.langchain.com/docs/tutorials/classification/'>LangChain Classification Function Calling</a></li>
    <li><a href='https://python.langchain.com/docs/integrations/providers/huggingface/'>LangChain HuggingFace</a></li>
</ul>


<p>
    <b>
    Objective: Familiar with Extraction using Function Calling in LangChain. The model used is from HuggingFace checkpoint.
    </b>
</p>

<h2>Guilds:</h2>
<ul>
    <li><a href="https://python.langchain.com/docs/tutorials/extraction/">LangChain Extraction</a></li>
    <li><a href="https://python.langchain.com/docs/concepts/tool_calling/">LangChain Function Calling</a></li>
    <li><a href="https://python.langchain.com/docs/integrations/providers/huggingface/">LangChain and HuggingFace</a></li>
</ul>
<h2>Main components of Extraction procedure (or Function/ Tools Calling):</h2>

<h3>Schema</h3>
<ul>
    <li>Describe what information we want to extract from the text</li>
</ul>
<h3>Prompt template</h3>
<ul>
    <li>instructions for extractor (LLM model) in a specific form</li>
</ul>
<h3>LLM model</h3>
<ul>
    <li>From pretrained HuggingFace checkpoint</li>
</ul>
<h3>Bind tool/func for LLM</h3>
<ul>
    <li>Binding the task for LLM using <b>chat_model.bind_tools(task)</b> function</li>
</ul>
<h3>Function calling and Show result</h3>
<ul>
    <li>Using <b>invoke()</b> function to feed prompt into LLM</li>
    <li>Need to cast response into a dict ("response.dict()") and then access the answer as an attribute of the dict ("dict['tool_calls']")</li>
</ul>

