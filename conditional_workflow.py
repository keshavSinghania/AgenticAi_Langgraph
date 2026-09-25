from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph , START , END
from typing import Literal, TypedDict
from pydantic import BaseModel , Field
from langchain_core.prompts import PromptTemplate

#open source llm
llm = ChatOllama(model="llama3.2:latest")

#Pydantic    
class QueryClassification(BaseModel):
    category : Literal["billing", "technical", "order", "general"]

class FinalResponse(BaseModel):
    category : Literal["billing", "technical", "order", "general"]
    response : str = Field(description="This will be the final response user will get")
    improve : str = Field(description="This will be the final response our team will see to get the idea of what to improve")

#Prompt
classifier_llm_prompt = PromptTemplate.from_template("""
Classify the user's query into exactly one of these categories:

- billing: payment, refund, transaction, charges, billing issues
- technical: bugs, errors, login problems, app/website technical issues
- order: order status, delivery, shipping, cancellation, tracking
- general: questions that don't clearly belong to the above categories

Choose the category that best matches the user's main problem.

User query:
{user_query}
""")
billing_prompt = PromptTemplate.from_template("""
You are a billing support specialist.

The classifier has already identified this query as a billing-related issue.
Do not change the category.

Analyze the customer's feedback and generate a structured response with:

1. response:
Write a clear, polite, and helpful response that will be shown directly
to the customer. Address the customer's billing problem as clearly as
possible.

2. improve:
Analyze the customer's feedback from the company's perspective.
Identify what the support team, billing system, or company should improve
based specifically on this feedback.

The improve field must contain a concrete and relevant improvement.
Do not give generic suggestions.

Customer feedback:
{user_query}
""")
technical_prompt = PromptTemplate.from_template("""
You are a technical support specialist.

The classifier has already identified this query as a technical issue.
Do not change the category.

Analyze the customer's feedback and generate a structured response with:

1. response:
Write a clear, polite, and easy-to-understand response that will be
shown directly to the customer. Help them understand or troubleshoot
their technical problem.

2. improve:
Analyze the customer's feedback from the company's perspective.
Identify what the technical team, product, system, documentation, or
support process should improve based specifically on this feedback.

The improve field must contain a concrete and relevant improvement.
Do not give generic suggestions.

Customer feedback:
{user_query}
""")
order_prompt = PromptTemplate.from_template("""
You are an order support specialist.

The classifier has already identified this query as an order-related issue.
Do not change the category.

Analyze the customer's feedback and generate a structured response with:

1. response:
Write a clear, polite, and helpful response that will be shown directly
to the customer. Address their order-related problem.

2. improve:
Analyze the customer's feedback from the company's perspective.
Identify what the company or support team should improve in the ordering,
delivery, tracking, or order-status process based specifically on this
feedback.

The improve field must contain a concrete and relevant improvement.
Do not give generic suggestions.

Customer feedback:
{user_query}
""")
general_prompt = PromptTemplate.from_template("""
You are a general customer support specialist.

The classifier has already identified this query as a general support
query. Do not change the category.

Analyze the customer's feedback and generate a structured response with:

1. response:
Write a clear, polite, and helpful response that will be shown directly
to the customer. Address their question or concern.

2. improve:
Analyze the customer's feedback from the company's perspective.
Identify what the support team, company, product, or customer experience
should improve based specifically on this feedback.

The improve field must contain a concrete and relevant improvement.
Do not give generic suggestions.

Customer feedback:
{user_query}
""")

#state for my workflow
class WorkflowState(TypedDict):
    user_query : str
    category : Literal["billing", "technical", "order", "general"]
    response : FinalResponse

# Nodes

# this llm will classify the feedback of user - >  topic "billing", "technical", "order", "general"
def classifier_llm(state : WorkflowState):
    user_query = state["user_query"]
    structured_llm = llm.with_structured_output(QueryClassification)
    chain = classifier_llm_prompt | structured_llm

    response = chain.invoke({"user_query": user_query})
    return {
        "category" : response.category
    }

def billing_response(state : WorkflowState):
    user_query = state["user_query"]
    structured_llm = llm.with_structured_output(FinalResponse)
    chain = billing_prompt | structured_llm
    response = chain.invoke({"user_query" : user_query})
    return {
        "response" : response
    }

def general_response(state : WorkflowState):
    user_query = state["user_query"]
    structured_llm = llm.with_structured_output(FinalResponse)
    chain = general_prompt | structured_llm
    response = chain.invoke({"user_query" : user_query})
    return {
        "response" : response
    }

def technical_response(state : WorkflowState):
    user_query = state["user_query"]
    structured_llm = llm.with_structured_output(FinalResponse)
    chain = technical_prompt | structured_llm
    response = chain.invoke({"user_query" : user_query})
    return {
        "response" : response
    }

def order_response(state : WorkflowState):
    user_query = state["user_query"]
    structured_llm = llm.with_structured_output(FinalResponse)
    chain = order_prompt | structured_llm
    response = chain.invoke({"user_query" : user_query})
    return {
        "response" : response
    }

def route_query(state : WorkflowState):
    return state["category"]

graph = StateGraph(WorkflowState)

graph.add_node("classifier_llm", classifier_llm)
graph.add_node("general_response", general_response)
graph.add_node("billing_response", billing_response)
graph.add_node("technical_response", technical_response)
graph.add_node("order_response", order_response)

graph.add_edge(START, "classifier_llm")
graph.add_conditional_edges("classifier_llm", route_query, {"billing" : "billing_response", "technical" : "technical_response", "order": "order_response", "general": "general_response"})
graph.add_edge("billing_response", END)
graph.add_edge("technical_response", END)
graph.add_edge("order_response", END)
graph.add_edge("general_response", END)

app = graph.compile()

result = app.invoke({
    "user_query": "My payment was deducted but my order is still pending."
})

print(result)