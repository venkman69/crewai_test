import os
from crewai import LLM
from lib import utils
from pathlib import Path
from crewai.tools import BaseTool
from crewai import Agent, Task, Crew, Process, CrewOutput
import sys
import datetime
# from crewai.memory import LongTermMemory

# Read your API key from the environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Use Gemini 2.5 Pro Experimental model
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=gemini_api_key,
    temperature=0.0,  # Lower temperature for more consistent results.
)


class TextExtractor(BaseTool):
    name: str = "text_extractor"
    description: str = "Extracts the text from a any PDF, text file, raw string or url."

    def _run(self, text_source: str) -> str:
        try:
            if Path(text_source).exists():
                if text_source.endswith(".pdf"):
                    print(f"Extracting text from PDF: {text_source}")
                    return utils.extract_text_from_pdf(text_source)
                elif text_source.endswith(".txt"):
                    print(f"Extracting text from TXT: {text_source}")
                    return utils.extract_text_from_txt(text_source)
            elif text_source.startswith("http"):
                print(f"Extracting text from URL: {text_source}")
                return utils.get_text_from_url(text_source)
            else:
                return text_source
        except Exception as e:
            # exception is thrown if it is already text and Path will throw an exception
            print("Assuming the text source is already raw text")
            return text_source


class JobSourceIdentifier(BaseTool):
    name: str = "job_source_identifier"
    description: str = "Identifies the source of a URL and identifies a reconstructable job url and ability to create filename"

    def _run(self, job_url: str) -> str:
        return utils.identify_job_source(job_url)


text_extractor = TextExtractor()
job_source_identifier = JobSourceIdentifier()

resume_text_extractor_agent = Agent(
    role="Resume Text extractor",
    goal="""Extract the text from the file or string provided.""",
    backstory="""You are a text extractor and can process raw text, urls, text files and PDFs.""",
    llm=gemini_llm,
    tools=[text_extractor],
    verbose=True,
    allow_delegation=False,
    memory=True,
)
resume_skill_analyzer_agent = Agent(
    role="Resume skill analyzer",
    goal="""Extract the skills from the file or string. """,
    backstory="""You are an expert HR professional specializing in identifying skills within a resume.""",
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False,
    memory=True,
)
resume_skill_analyzer_task = Task(
    description="""Extract the skills and years of work experience from the resume raw string data within {resume_url_or_text}
    work experience is overall number of years from the first job to the last job and round up.
    In addition, extract the name and contact information from the resume.""",
    agent=resume_skill_analyzer_agent,
    expected_output="""Output should be in JSON format, 
    for example : 
        {
            "resume_skills":["Python","WAF","Jira"], 
            "years_of_experience":5,
            "name":"John Doe",
            "email":"john.doe@example.com",
            "phone":"123-456-7890"
        }""",
)
job_text_extractor_agent = Agent(
    role="Job Text extractor",
    goal="""Extract the text from the file, url or string from provided input. """,
    backstory="""You are a text extractor and can process raw text, urls, text files and PDFs.
    You can also identify the source of a URL and identify a reconstructable job url and ability to create filename.""",
    llm=gemini_llm,
    tools=[text_extractor, job_source_identifier],
    verbose=True,
    allow_delegation=False,
    memory=True,
)
job_text_extraction_task = Task(
    description="""Extract the text from the job description from {job_posting_url_or_string}.""",
    agent=job_text_extractor_agent,
    expected_output="""Output should be in JSON format, 
    for example : 
        {
            "job_description":"Job description text",
            "job_posting_details":
            {
                "job_source": "Indeed", 
                "job_id": job_id, 
                "job_url": "https://www.indeed.com/viewjob?jk=1234567890"
            }
        }
    """,
)

job_skill_analyzer_agent = Agent(
    role="Job skill analyzer",
    goal="""Extract the skills from the job description. """,
    backstory="""You are an expert HR professional specializing in identifying skills 
    within a job description. You can also extract the organization name from the job description.
    You can also extract the job posting details such as job_source, job_id and job_url from the url
    In addition you can pass the job posting details.""",
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False,
    memory=True,
    tools=[job_source_identifier],
)
job_skill_analyzer_task = Task(
    description="""Extract the organization name, summary of the role, years of industry experience, and skills from provided inputs in: {job_text} and {job_url}.
    Skills should be categorized into required and preferred.
    if a particular skill mentions a minimum years of experience, 
    then represent it within brackets for example "Python (2 years)".
    Add the organization name and years of experience to the job_posting_details and pass it.""",
    agent=job_skill_analyzer_agent,
    expected_output="""Output should be in JSON format, 
    for example : 
        {
            "required_skills":["Python (2 years)","WAF"],
            "preferred_skills":["Jira","TOGAF"],
            "job_posting_details":
            {
                "role_summary":"Role summary text",
                "organization":"Google",
                "years_of_experience": "5",
                "job_source": "Indeed", 
                "job_id": "1234567890", 
                "job_url": "https://www.indeed.com/viewjob?vjk=1234567890"
            }
        }""",
)

job_vs_resume_skill_matching_agent = Agent(
    role="HR Expert who can review the job skills and candidate skills and assign a score",
    goal="""Review the job skills and candidate skills and assign a score.""",
    backstory="""You are an expert HR professional who can determine how a candidate skills match the required and preferred skills in a job description
    """,
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False,
    memory=True,
)
job_vs_resume_skill_matching_task = Task(
    description="""Identify a match score for the job and candidate skills contained within the JSON formatted strings.
    Job skils are in: {job_analysis} and candi date skills are in: {resume_analysis}
    Compare the required and preferred skills for the job with the candidate skills and assign a score.
    Use this criteria to calculate the score:
    1. Calculate 'required_skill_match_score': Count the number of required skills in the job that are present in the candidate and divide by the total number of required skills in the job to get the required skill match score.
    2. Calculate 'preferred_skill_match_score': Count the number of preferred skills in the job that are present in the candidate and divide by the total number of preferred skills in the job to get the preferred skill match score.
    3. Calculate 'final_score': Final score = required_skill_match_score * 0.7 + preferred_skill_match_score * 0.3  
    for example if job has 10 required skills and 5 preferred skills and 
       candidate has 8 required skills and 3 preferred skills then final score = 8/10 * 0.7 + 3/5 * 0.3 = 0.74
    4. Score data should be presented as keys in JSON output as below:
       a. A tag named "score" which is a dictionary with the following keys:
          i. final_score
          ii. required_skill_match_score
          iii. preferred_skill_match_score
          iv. matching_required_skills_count, 
          v. missing_required_skills_count, 
          vi. matching_preferred_skills_count, 
          vii. missing_preferred_skills_count, 
          viii. total_required_skills_count,
          ix.  total_preferred_skills_count, 
    5. Pass the job_posting_details as is
    6. If a skill mentions a minimum years of experience, then compare the years of experience 
        in the resume with the minimum years of experience in the job description. if the years of experience 
        in the resume is less than the minimum years of experience in the job description then consider 
        it as a missing skill.
    
    """,
    agent=job_vs_resume_skill_matching_agent,
    expected_output="""Output should be in JSON format, 
    for example : 
        {
            "matching_required_skills":["Java","Python"], 
            "missing_required_skills":["Ruby", "TOGAF"], 
            "matching_preferred_skills":["CISSP","WAF"],
            "missing_preferred_skills":["Jira","management"],
            "organization":"Google",
            "job_posting_details": {
                "job_source": "Indeed", 
                "job_id": "1234567890", 
                "job_url": "https://www.indeed.com/viewjob?vjk=1234567890"
            },
            "score": {
                "final_score": 0.2139,
                "required_skill_match_score": 0.2667,
                "preferred_skill_match_score": 0.0909,
                "matching_required_skills_count": 8,
                "missing_required_skills_count": 22,
                "matching_preferred_skills_count": 1,
                "missing_preferred_skills_count": 10,
                "total_required_skills_count": 30,
                "total_preferred_skills_count": 11
            }
        }""",
)

resume_vs_job_decision_agent = Agent(
    role="""HR Expert who can review the analysis and render a final summary, 
    specifically whether job is a good fit for the candidate""",
    goal="""Check the missing skills to determine if it may disqualify the candidate.""",
    backstory="""You are an expert HR professional who can decide whether a candidate passes
    to the next stage of the hiring process or not.
    """,
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False,
    memory=True,
)
resume_vs_job_decision_task = Task(
    description="""
    1. Look at the required missing skills, if any of the following appear then candidate will not pass:
        a. Active Security Clearance.
        b. Specific certifications that are required.
        c. If a skill mentions a minimum years of experience, then compare 
    2. Pass all the data from the job_skill_analyzer_task output.
    3. Add a new key named "resume_vs_job_decision" to the output.
    4. The "resume_vs_job_decision" key should contain a dictionary with the following keys:
        a. decision
        b. reason
    5. If the candidate does not fail due to any of the above criteria, and the final score is above 70%,
    then the decision should be "Pass" and the reason should be "Score is above 70% and candidate has sufficient skills and years of experience".
    Otherwise the decision should be "Fail" along with the appropriate reason(s).
    """,
    agent=resume_vs_job_decision_agent,
    expected_output="""Output should be in JSON format, adding to the output from the resume_vs_job_decision_task. 
    for example this should be added to the output from the resume_vs_job_decision_task within the key "resume_vs_job_decision" :
         {
         "decision":"Pass", 
         "reason":"Score is above 70% and candidate has sufficient skills and years of experience"}
         or
         {
         "decision":"Fail", 
         "reason":"Candidate is missing required skills or Active Security Clearance"}
    """,
)


def resume_skill_analyser_crew(resume_url_or_text: str, cache: bool = True):
    # get the resume text
    resume_text = text_extractor.run(resume_url_or_text)
    if cache and resume_text in utils.dc:
        print(f"Returning cached result for resume {resume_url_or_text[:100]}...")
        return utils.dc[resume_text]
    resume_crew = Crew(
        agents=[
            resume_skill_analyzer_agent,
        ],
        tasks=[
            resume_skill_analyzer_task,
        ],
        process=Process.sequential,
        verbose=True,
    )
    result = resume_crew.kickoff(inputs={"resume_url_or_text": resume_text})
    print("Caching result for resume")
    utils.dc[resume_text] = result
    save_resume_skill_analysis(result)
    return result


def job_skill_analyser_crew(job_url: str, cache: bool = True):
    # get the job text
    job_text = text_extractor.run(job_url)
    if cache and job_text in utils.dc:
        print("Returning cached result for job")
        return utils.dc[job_text]
    job_crew = Crew(
        agents=[
            job_skill_analyzer_agent,
        ],
        tasks=[
            job_skill_analyzer_task,
        ],
        process=Process.sequential,
        verbose=True,
    )
    result = job_crew.kickoff(inputs={"job_text": job_text, "job_url": job_url})
    print("Caching result for job")
    utils.dc[job_text] = result
    save_job_skill_analysis(result)
    return result


def compare_and_decide_crew(
    resume_analysis: str, job_analysis: str, cache: bool = True
):
    if cache and (resume_analysis, job_analysis) in utils.dc:
        print("Returning cached result for compare and decide")
        return utils.dc[(resume_analysis, job_analysis)]
    compare_and_decide_crew = Crew(
        agents=[
            job_vs_resume_skill_matching_agent,
            resume_vs_job_decision_agent,
        ],
        tasks=[
            job_vs_resume_skill_matching_task,
            resume_vs_job_decision_task,
        ],
        process=Process.sequential,
        verbose=True,
    )
    result = compare_and_decide_crew.kickoff(
        inputs={"resume_analysis": resume_analysis, "job_analysis": job_analysis}
    )
    print("Caching result for compare and decide")
    utils.dc[(resume_analysis, job_analysis)] = result
    save_final_crew_result(result)
    return result


class FakeResult:
    def __init__(self, raw: str = ""):
        if raw != "":
            self.raw = raw
        else:
            if Path("fake_result_data.txt").exists():
                with open("fake_result_data.txt", "r") as f:
                    self.raw = f.read()
            else:
                self.raw = ""


def save_final_crew_result(crew_result: CrewOutput):
    print("Writing Job task outputs to files")
    os.makedirs("logs", exist_ok=True)
    complete_flow_result_json = utils.extract_json_from_crew_output(crew_result.raw)
    job_id = complete_flow_result_json["job_posting_details"]["job_id"]
    job_source = complete_flow_result_json["job_posting_details"]["job_source"]
    job_filename = f"final_decision_{job_source}_{job_id}.txt"
    save_result(crew_result, job_filename)
    return job_filename


def save_job_skill_analysis(crew_result: CrewOutput):
    print("Writing Job task outputs to files")
    os.makedirs("logs", exist_ok=True)
    job_skill_analysis_result_json = utils.extract_json_from_crew_output(
        crew_result.raw
    )
    job_id = job_skill_analysis_result_json["job_posting_details"]["job_id"]
    job_source = job_skill_analysis_result_json["job_posting_details"]["job_source"]
    job_filename = f"job_skills_analysis_{job_source}_{job_id}.txt"
    save_result(crew_result, job_filename)
    return job_filename


def save_resume_skill_analysis(crew_result: CrewOutput):
    print("Writing Resume task outputs to files")
    os.makedirs("logs", exist_ok=True)
    resume_skill_analysis_result_json = utils.extract_json_from_crew_output(
        crew_result.raw
    )
    candidate_name = resume_skill_analysis_result_json["name"].replace(" ", "_")
    datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    resume_analysis_filename = f"resume_skills_analysis_{candidate_name}_{datestr}.txt"
    save_result(crew_result, resume_analysis_filename)
    return resume_analysis_filename


def save_result(crew_result: CrewOutput, file_name: str):
    if isinstance(crew_result, FakeResult):
        return "logs/fake_result.txt"
    with open(f"logs/{file_name}", "w") as f:
        for i, task_output in enumerate(crew_result.tasks_output):
            f.write(f"-------------- {task_output.agent} --------------\n")
            f.write(f"Description:\n{task_output.description}\n")
            f.write(f"Summary:\n{task_output.summary}\n")
            for mesg in task_output.messages:
                f.write(f"{mesg['role']}: {mesg['content']}\n")
            f.write(f"Raw:\n{task_output.raw}\n")
            f.write("\n")
    return file_name


if __name__ == "__main__":
    # complete_flow_result = run_flow(
    #     "/mnt/g/My Drive/Personal/Resume/Srpincipal/NarayanNatarajan Resume.pdf",
    #     sys.argv[1],
    # )

    # print("--- Crew Execution Finished ---")
    # save_result(complete_flow_result)

    # read from command line prompt:
    job_description = input("Enter job URL: ")

    resume_analysis = resume_skill_analyser_crew(
        "/mnt/g/My Drive/Personal/Resume/Srpincipal/NarayanNatarajan Resume.pdf",
        cache=True,
    )
    job_analysis = job_skill_analyser_crew(job_description, cache=False)

    final_decision = compare_and_decide_crew(
        resume_analysis.raw, job_analysis.raw, cache=True
    )

    print(final_decision.raw)
