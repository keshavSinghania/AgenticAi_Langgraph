from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model="llama3.2:latest")

#flow
#User -> llm (generate template) -> another llm (template + topic == final blog) -> another llm(evaluation) -> output (question, template , blog and eval)

#MEMORY (STATE)
class blog_state(TypedDict):
    topic: str
    template : str
    blog : str
    evaluation: str

#nodes(function)

#function to generate template for blog using topic
def generate_template(state:blog_state) -> blog_state:
    print("generating template")
    prompt = PromptTemplate(
        template="Generate a template for topic: {topic} ,so that i can use that template to generate blog",
        input_variables=["topic"]
    )

    chain = prompt | llm
    topic = state["topic"]
    template = chain.invoke({"topic" : topic})
    print("template generation done")
    state["template"] = template.content
    return {"template": template.content}
    
#function to generate blog using template
def generate_blog(state: blog_state) -> blog_state:
    print("generating blog using tempplate")
    prompt = PromptTemplate(
        template="Generate a detailed blog of topic : {topic} using this template: {template}",
        input_variables=["topic", "template"]
    )
    chain = prompt | llm
    topic = state["topic"]
    template = state["template"]
    blog = chain.invoke({"topic" :topic, "template": template})
    print("blog genertated")
    state["blog"] = blog.content
    return {"blog" : blog.content}

#function to evaluate out blog using topic and blog
def evaluate_blog(state:blog_state) -> blog_state:
    print("evaluating your blog")
    prompt = PromptTemplate(
        template="Evaluate this blog : {blog} based on this topic : {topic}, please note you must have to rank this btw 1 to 10 , and given answer directly btw 1 to 10 , just integer value as response",
        input_variables= ["blog", "topic"]
    )

    chain = prompt | llm
    print("evaluated your blog")
    evaluation = chain.invoke({
        "blog" : state["blog"],
        "topic" : state["topic"]
    })
    state["evaluation"] = evaluation.content
    return {"evaluation" : evaluation.content}



#create graph
graph = StateGraph(blog_state)


#Nodes
graph.add_node("generate_template", generate_template)
graph.add_node("generate_blog", generate_blog)
graph.add_node("evaluate_blog", evaluate_blog)

#Edges
graph.add_edge(START, "generate_template")
graph.add_edge("generate_template", "generate_blog")
graph.add_edge("generate_blog", "evaluate_blog")
graph.add_edge("evaluate_blog", END)


app = graph.compile()

# print(app.get_graph().draw_ascii())

topic = input("Enter your topic")
result = app.invoke({
    "topic": topic
})
print("\nTOPIC:", result["topic"])
print("\nTEMPLATE:", result["template"])
print("\nBLOG:", result["blog"])
print("\nEVALUATION:", result["evaluation"])