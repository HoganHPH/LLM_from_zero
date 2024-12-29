<h1>Comprehensive LangChain Function Calling</h1>
<h2>Key concepts</h2>
<ol>
    <b>5 main steps when using tool calling:</b>
    <li>Tool Creation</li>
    <li>Tool Binding</li>
    <li>Tool Calling</li>
    <li>Tool Execution</li>
    <li>Passing Tool outputs to Chat model</li>
</ol>
<h2>Comprehensive:</h2>
<h3>Tool Creation</h3>
<p><b>A tool is an association between a function and its schema</b></p>
<p><b>There are 3 ways to define a tool:</b></p>
<ul>
    <li>1) Using function:</li>
        <ul>
            <li>With Decorator</li>
            <li>With StructuredTool</li>
            Provides a bit more configurability than the @tool decorator
        </ul>
    <li>2) Using LangChain Runnables</li>
    <li>3) Using sub-classing from BaseTool</li>
    Provides maximal control over the tool definition, but requires writing more code
</ul>
<h3>Tool Binding</h3>
<p><b>Defining tool schemas and Bind to the Chat Model</b></p>
<p><b>How to bind? -> Use chat_model.bind_tools([tool_1, tool_2,...]) function</b></p>
<p><b>There are 4 ways to define tool schemas:</b></p>
<ul>
    <li>1) Python functions</li>
    <li>2) LangChain Tool (@tool decorator)</li>
    <li>3) Pydantic class</li>
    <li>4) TypedDict class</li>
</ul>
<h3>Tool Calling</h3>
<p><b>The Chat model return the tools in its response as a .tool_calls attribute and also arguments</b></p>
<p>Example:
    <code>
        response = llm_with_tools.invoke(query)
        returned_tools = response.tool_calls
    </code>
</p>
<h3>Tool Execution</h3>
<p><b>Execute the function with arguemnts corresponding to the tools and return the results of the called function</b></p>
<p>Example:
    <code>
        if len(returned_tools) > 0:
            for tool in returned_tools:
                args = tool['args']
                matching_tool= [t for t in tools if t.name == tool['name']]
                response = matching_tool[0].invoke(args)
                print(response)
        else:
            print("No tool call")
    </code>
</p>
<h3>Passing Tool outputs to Chat model</h3>
<p><b>Pass the results of the called function back to Chat model so that Chat model can generate a message for user</b></p>
<p><b>
    !Note:
    ToolMessage must include a tool_call_id that matches an id in the original tool calls 
    that the model generates. This helps the model match tool responses with tool calls.
</b></p>
<p>Example:
    <code>
        messages = [HumanMessage(query)]
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        for tool_call in ai_msg.tool_calls:
            selected_tool = {"addition-tool": add, "multiplication-tool": multiply}[tool_call["name"].lower()]
            tool_msg = selected_tool.invoke(tool_call)
            messages.append(tool_msg)
        response = llm_with_tools.invoke(messages)
        print(response.content)
    </code>
</p>
<h2>References</h2>
<ul>
    <li><a href='https://python.langchain.com/docs/how_to/custom_tools/#creating-tools-from-functions'>How to create tools</a></li>
    <li><a href='https://python.langchain.com/docs/concepts/tool_calling/'>Tool calling</a></li>
    <li><a href='https://python.langchain.com/docs/how_to/tool_calling/'>How to use chat models to call tools</a></li>
    <li><a href='https://python.langchain.com/docs/how_to/tool_results_pass_to_model/'>How to pass tool outputs to chat models</a></li>
    <li><a href='https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/'>Defining Custom Tools</a></li>
    <li><a href='https://python.langchain.com/docs/how_to/tool_choice/'>How to force models to call a tool</a></li>
    <li><a href='https://python.langchain.com/docs/how_to/tool_runtime/'>How to pass run time values to tools</a</li>
</ul>
<h2>What's next?</h2>
<ul>
    <li><a href='https://langchain-ai.github.io/langgraph/tutorials/introduction/'>LangGraph Quickstart</a</li>
</ul>

