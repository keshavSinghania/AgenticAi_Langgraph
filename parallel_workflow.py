from langgraph.graph import StateGraph, START, END
from typing import TypedDict , Annotated
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
import operator

llm = ChatOllama(model = "llama3.2:latest")

#prommpt templates
skill_analysis_prompt = PromptTemplate.from_template("""
You are a technical recruiter.

Analyze how well the candidate's skills match the skills required for the job.

Candidate Skills:
{candidate_skills}

Required Job Skills:
{required_skills}

Evaluate:
- Which skills match
- Which required skills are missing
- Give a score from 0 to 100
- Briefly explain the reasoning

Return the analysis according to the required structured output.
""")
experience_analysis_prompt = PromptTemplate.from_template("""
You are a technical recruiter.

Analyze how well the candidate's experience matches the requirements of the job.

Candidate Experience:
{candidate_experience}

Job Description:
{job_description}

Evaluate:
- Whether the candidate's experience is relevant to the job
- Whether the candidate has enough experience for the role
- Give a score from 0 to 100
- Briefly explain the reasoning

Return the analysis according to the required structured output.
""")
project_analysis_prompt = PromptTemplate.from_template("""
You are a technical recruiter.

Analyze how relevant the candidate's projects are to the given job.

Candidate Projects:
{candidate_projects}

Job Description:
{job_description}

Evaluate:
- How relevant each project is to the job
- Which projects demonstrate the required technical skills
- Give a score from 0 to 100
- Briefly explain the reasoning

Return the analysis according to the required structured output.
""")
final_analysis_prompt = PromptTemplate.from_template("""
You are a technical recruiter.

You have received the following analyses of a candidate:

{analysis_results}

Combine these analyses into one final candidate assessment.

Evaluate:
- Overall suitability for the job
- Strengths of the candidate
- Main areas where the candidate is lacking
- Overall score from 0 to 100
- Brief reasoning for the final assessment

Do not perform the individual skill, experience, or project analysis again.
Use the provided analyses to create the final assessment.

Return the analysis according to the required structured output.
""")

#pydantic 
class Candidate_Profile(BaseModel):
    name: str
    skills : list[str]
    experience : str
    projects: list[str]

class Job_Desciption(BaseModel):
    role : str
    required_skills : list[str]
    description : str

class Skill_Analysis(BaseModel):
    score : int
    matched_skills : list[str]
    missing_skills: list[str]
    reason : str

class Experience_Analysis(BaseModel):
    score: int
    reason : str

class Project_Analysis(BaseModel):
    score : int
    relevent_projects : list[str]
    reason : str        

class Final_Analysis(BaseModel):
    overall_score: int
    strengths: list[str]
    weaknesses: list[str]
    overall_assessment: str

#State for workflow
class Resume_Analysis_State(TypedDict):
    candidate_profile : Candidate_Profile
    job_description : Job_Desciption

    analysis_result : Annotated[list, operator.add]

    final_analysis : str


#NODES
# giving candidate skills and candidate required skills to analysis skill
def skill_analyser_llm(state : Resume_Analysis_State):
    candidate_skills = state["candidate_profile"].skills
    required_skills = state["job_description"].required_skills

    structured_llm = llm.with_structured_output(Skill_Analysis)

    chain = skill_analysis_prompt | structured_llm

    response = chain.invoke({
        "candidate_skills" : candidate_skills,
        "required_skills" : required_skills
    })

    return{
        "analysis_result" : [
            {
            "category" : "skill",
            "result" : response
            }
        ]
    }

#giving candidate_experience and job_desciption to analysis experince
def experience_analyser_llm(state : Resume_Analysis_State):
    candidate_experience = state["candidate_profile"].experience
    job_description = state["job_description"].description

    structured_llm = llm.with_structured_output(Experience_Analysis)

    chain = experience_analysis_prompt | structured_llm
    response = chain.invoke({
        "candidate_experience" : candidate_experience,
        "job_description" : job_description
    })

    return {
        "analysis_result":[
            {
                "category" : "experience",
                "result" : response
            }
        ]
    }

#giving candidate_project and job_description to analysis based on project
def project_analyser_llm(state : Resume_Analysis_State):
    
    candidate_projects = state["candidate_profile"].projects
    job_description = state["job_description"].description

    structured_llm = llm.with_structured_output(Project_Analysis)

    chain = project_analysis_prompt | structured_llm
    response = chain.invoke({
        "candidate_projects" : candidate_projects,
        "job_description" : job_description
    })

    return {
        "analysis_result": [
            {
                "category" : "project",
                "result" : response
            }
        ]
    }

#llm for final analysis based on all analysis result
def final_analyser_llm(state : Resume_Analysis_State):
    analysis_results = state["analysis_result"]

    structured_llm = llm.with_structured_output(Final_Analysis)
    chain = final_analysis_prompt | structured_llm
    response = chain.invoke({
        "analysis_results" : analysis_results
    })

    return {
        "final_analysis" : response
    }

graph  = StateGraph(Resume_Analysis_State)

# Node integration
graph.add_node("skill_analyser_llm", skill_analyser_llm)
graph.add_node("experience_analyser_llm", experience_analyser_llm)
graph.add_node("project_analyser_llm", project_analyser_llm)
graph.add_node("final_analyser_llm", final_analyser_llm)

# Edges integration
graph.add_edge(START, "skill_analyser_llm")
graph.add_edge(START, "experience_analyser_llm")
graph.add_edge(START, "project_analyser_llm")

graph.add_edge("skill_analyser_llm", "final_analyser_llm")
graph.add_edge("experience_analyser_llm", "final_analyser_llm")
graph.add_edge("project_analyser_llm", "final_analyser_llm")

graph.add_edge("final_analyser_llm", END)

app = graph.compile()

candidate = Candidate_Profile(
    name=input("Enter name: "),
    skills=input("Enter skills: ").split(","),
    experience=input("Enter experience: "),
    projects=input("Enter projects: ").split(",")
)

job = Job_Desciption(
    role=input("Enter role: "),
    required_skills=input("Enter required skills: ").split(","),
    description=input("Enter job description: ")
)

result = app.invoke({
    "candidate_profile": candidate,
    "job_description": job,
    "analysis_result": [],
    "final_analysis": ""
})
print(result["final_analysis"])