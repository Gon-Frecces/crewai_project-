import os
import requests
from bs4 import BeautifulSoup
from crewai import Agent, Task, Crew
from langchain.tools import tool
from decouple import config
from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
import ollama
from langchain import ollama

llm = ollama(model = "openhermes")



class WebBrowserTool():

    @tool("internet_search", return_direct=False)
    def internet_search(query: str) -> str:
        """Useful for quering content on the internet using DuckDuckGo"""
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
            return results if results else "No results found"

    @tool("process_search_results", return_direct=False)
    def process_search_results(url: str) -> str:
        """Process Content From Webpage"""
        response = requests.get(url=url)
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()


tools = [
    WebBrowserTool().internet_search,
    WebBrowserTool().process_search_results
]


# Define your agents with roles and goals
researcher = Agent(
    role='Researcher',
    goal='Develop ideas for teaching someone new to the subject',
    backstory="""Your primary role is to function as an health eduction researcher, adept at scouring 
    the internet for the latest and most relevant trending stories across various health sectors like community health, 
    first aid, health education, safety practices, emergency response lines, fitness and accident management. You possess the 
    capability to access a wide range of online news sources, 
    blogs, and social media platforms to gather real-time information.""",
    verbose=True,
    # Agent not allowed to delegate tasks to any other agent
    allow_delegation=False,
    tools=tools,
    llm=llm
)


writer = Agent(
    role='Writer',
    goal=" Use the Researcher's ideas to write a piece of text to explain the topic",
    backstory="""To craft compelling and informative eductional reports using insights provided by the researcher, 
    focusing on delivering high-quality health educational texts which students can read to get informed .""",
    verbose=True,
    allow_delegation=False,
    tools=tools,
    llm=llm
)


examiner = Agent(
    role='Examiner',
    goal=" Craft two or three test questions to evalute understanding of the created text, along with the correct answers. In other words test whether a student has fully understood the text",
    backstory="""To craft compelling and relevant test questions to  evaluate the student's understnding of the 
    created text, along with the correct answers.
    """,
    verbose=True,
    allow_delegation=False,
    tools=tools,
    llm=llm
)

task1 = Task(
    description="""Conduct a comprehensive research on Health Education, specifically focusing on first aid, 
    safety practices, emergency response lines,and accident management.
    Your final response should provide a detailed explanation on basic Health Education summarized into 5 paragraphs.
    Also include any advancements made in the space industry. 
    Make sure you include links to the sources (websites) from which you obtained the facts from.""",
    agent=researcher
)


task2 = Task(
    description="""Using the research findings of the researcher, write an educational publication or report. 
    Your final answer MUST be the full blog post of at least 3 paragraphs.""",
    agent=writer
)

task3 = Task(
    description="""Using the educational report from the writer, craft a test that has 3 questions to test the
    understanding of student who read the report""",
    agent=examiner
)




agents = [researcher, writer, examiner]

# Instantiate your crew with a sequential process
crew = Crew(
    agents=agents,
    tasks=[task1, task2, task3],
    verbose=2
)

# # Get your crew to work!
result = crew.kickoff()


print(result)